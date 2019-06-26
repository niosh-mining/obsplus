"""
Bank ABC
"""
import gc
import os
import threading
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from os.path import join
from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import pandas as pd
import tables
from pandas.io.sql import DatabaseError

import obsplus
from obsplus.exceptions import BankDoesNotExistError, BankIndexLockError
from obsplus.interfaces import ProgressBar
from obsplus.utils import iter_files, get_progressbar


BankType = TypeVar("BankType", bound="_Bank")


class _Bank(ABC):
    """
    The abstract base class for ObsPlus' banks. Used to access local
    archives in a client-like fashion.
    """

    # hdf5 compression defaults
    _complib = "blosc"
    _complevel = 9
    # attributes subclasses need to define
    ext = ""
    bank_path = ""
    namespace = ""
    index_name = ".index.h5"  # name of index file
    # indexing process down.
    # concurrency handling features
    _lock_file_name = ".~obsplus_hdf5.lock"
    _owns_lock = False
    _concurrent = False
    _index_lock = threading.RLock()  # lock for updating index thread
    # optional str defining the directory structure and file name schemes
    path_structure = None
    name_structure = None
    # the minimum obsplus version. If not met delete index and re-index
    # bump when database schema change.
    _min_version = "0.0.0"
    # status bar attributes
    _bar_update_interval = 50  # number of files before updating bar
    _min_files_for_bar = 100  # min number of files before using bar enabled

    @abstractmethod
    def read_index(self, **kwargs) -> pd.DataFrame:
        """ read the index filtering on various params """

    @abstractmethod
    def update_index(self: BankType) -> BankType:
        """ update the index """

    @abstractmethod
    def last_updated(self) -> Optional[float]:
        """ get the last modified time stored in the index. If
        Not available return None
        """

    @abstractmethod
    def _read_metadata(self) -> pd.DataFrame:
        """ Return a dictionary of metadata. """

    # --- path/node related objects

    @property
    def index_path(self):
        """
        The expected path to the index file.
        """
        return join(self.bank_path, self.index_name)

    @property
    def lock_file_path(self):
        """
        The expected path for the lock file.
        """
        return join(self.bank_path, self._lock_file_name)

    @property
    def _index_node(self):
        """
        The node, or table, the index information is stored in the database.
        """
        return "/".join([self.namespace, "index"])

    @property
    def _index_version(self) -> str:
        """
        Get the version of obsplus used to create the index.
        """
        return self._read_metadata()["obsplus_version"].iloc[0]

    @property
    def _time_node(self):
        """
        The node, or table, the update time information is stored in the database.
        """
        return "/".join([self.namespace, "last_updated"])

    @property
    def _meta_node(self):
        """
        The node, or table, the update metadata is stored in the database.
        """
        return "/".join([self.namespace, "metadata"])

    def _enforce_min_version(self):
        """
        Check version of obsplus used to create index and delete index if the
        minimum version requirement is not met.
        """
        try:
            version = self._index_version
        except (FileNotFoundError, DatabaseError):
            return
        else:
            if self._min_version > version:
                msg = (
                    f"the indexing schema has changed since {self._min_version} "
                    f"the index will be recreated"
                )
                warnings.warn(msg)
                os.remove(self.index_path)

    def _unindexed_file_iterator(self):
        """ return an iterator of potential unindexed waveform files """
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        last_updated = self.last_updated  # this needs db so only call once
        if last_updated is not None:
            mtime = last_updated - 0.001
        # return file iterator
        return iter_files(self.bank_path, ext=self.ext, mtime=mtime)

    def _make_meta_table(self):
        """ get a dataframe of meta info """
        meta = dict(
            path_structure=self.path_structure,
            name_structure=self.name_structure,
            obsplus_version=obsplus.__version__,
        )
        return pd.DataFrame(meta, index=[0])

    def get_service_version(self):
        """ Return the version of obsplus used to create index. """
        return self._index_version

    def ensure_bank_path_exists(self, create=False):
        """
        Ensure the bank_path exists else raise an BankDoesNotExistError.

        If create is True, simply create the bank.
        """
        path = Path(self.bank_path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            msg = f"{path} is not a directory, cant read bank"
            raise BankDoesNotExistError(msg)

    def block_on_index_lock(self, wait_interval=0.2, max_retry=10):
        """
        Blocks until the lock is released.

        Will wait a certain number of seconds a certain number of times
        before raising a BankIndexLockError.

        Parameters
        ----------
        wait_interval
            The number of seconds to wait between each try
        max_retry
            The number of times to retry before raising.
        """
        # if the concurrent feature is not selected just bail out
        if not self._concurrent:
            return
        # if there is no lock, or this bank owns the lock return
        if not os.path.exists(self.lock_file_path) or self._owns_lock:
            return
        # get random state, based on process id, for waiting random seconds
        rand = np.random.RandomState(os.getpid() or 13)
        # wait until the lock file is gone or timeout occurs (then raise)
        count = 0
        while count < max_retry:
            if not os.path.exists(self.lock_file_path):
                return
            time.sleep(wait_interval + rand.rand() * wait_interval)
            count += 1
        else:
            duration = max_retry * wait_interval
            msg = (
                f"{self.lock_file_path} was not released after about "
                f"{duration * 1.5} seconds. It may need to be manually deleted"
                f" if the index is not in the process of being updated."
            )
            raise BankIndexLockError(msg)

    @contextmanager
    def lock_index(self):
        """
        Acquire lock for work inside context manager.
        """
        if self._concurrent:
            self.block_on_index_lock()  # ensure lock isn't already in use
            assert Path(self.bank_path).exists()
            open(self.lock_file_path, "w").close()  # create lock file
            self._owns_lock = True
            # close all open files (nothing should have tables at this point)
            gc.collect()  # must call GC to ensure everything is cleaned up (ugly)
            tables.file._open_files.close_all()
        yield
        if self._concurrent:
            with suppress(FileNotFoundError):
                os.unlink(self.lock_file_path)
            self._owns_lock = False

    def get_progress_bar(self, bar=None) -> Optional[ProgressBar]:
        """
        Return a progress bar instance based on bar parameter.

        If bar is False, return None.
        If bar is None return default Bar
        If bar is a subclass of ProgressBar, init class and set max_values.
        If bar is an instance of ProgressBar, return it.
        """
        # conditions to bail out early
        if bar is False:  # False indicates no bar is to be used
            return None
        elif isinstance(bar, ProgressBar):  # bar is already instantiated
            return bar
        # next, count number of files
        num_files = sum([1 for _ in self._unindexed_file_iterator()])
        if num_files < self._min_files_for_bar:  # not enough files to use bar
            return None
        # instantiate bar and return
        print(f"updating or creating event index for {self.bank_path}")
        kwargs = {"min_value": self._min_files_for_bar, "max_value": num_files}
        # an instance should be init'ed
        if isinstance(bar, type) and issubclass(bar, ProgressBar):
            return bar(**kwargs)
        elif bar is None:
            return get_progressbar(**kwargs)
        else:
            msg = f"{bar} is not a valid input for get_progress_bar"
            raise ValueError(msg)
