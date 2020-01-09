"""
Bank ABC
"""
import os
import warnings
from abc import ABC, abstractmethod
from os.path import join
from pathlib import Path
from typing import Optional, TypeVar, Mapping
from types import MappingProxyType as MapProxy

import pandas as pd
from pandas.io.sql import DatabaseError

import obsplus
from obsplus.constants import CPU_COUNT, bank_subpaths_type
from obsplus.exceptions import BankDoesNotExistError
from obsplus.interfaces import ProgressBar
from obsplus.bank.utils import _IndexCache
from obsplus.utils import iter_files, get_progressbar, iterate

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
    executor = None  # an executor for using parallelism
    # optional str defining the directory structure and file name schemes
    path_structure = None
    name_structure = None
    # the minimum obsplus version. If not met delete index and re-index
    # bump when database schema change.
    _min_version = "0.0.3"
    # status bar attributes
    _bar_update_interval = 50  # number of files before updating bar
    _min_files_for_bar = 100  # min number of files before using bar enabled
    _read_func: callable  # function for reading datatype
    # required dypes for input to storage layer
    _dtypes_input: Mapping = MapProxy({})
    # required dtypes for output from bank
    _dtypes_output: Mapping = MapProxy({})
    # the index cache (can greatly reduce IO efforts)
    _index_cache: Optional[_IndexCache] = None

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

    def _unindexed_iterator(self, sub_paths: Optional[bank_subpaths_type] = None):
        """ return an iterator of potential unindexed files """
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        last_updated = self.last_updated  # this needs db so only call once
        if last_updated is not None:
            mtime = last_updated - 0.001
        # get paths to iterate
        bank_path = self.bank_path
        if sub_paths is None:
            paths = self.bank_path
        else:
            # TODO move this to bank utils
            paths = [
                f"{self.bank_path}/{x}" if str(bank_path) not in str(x) else str(x)
                for x in iterate(sub_paths)
            ]
        # return file iterator
        return iter_files(paths, ext=self.ext, mtime=mtime)

    def _measured_unindexed_iterator(
        self,
        bar: Optional[ProgressBar] = None,
        sub_paths: Optional[bank_subpaths_type] = None,
    ):
        """
        A generator to yield un-indexed files and update progress bar.

        Parameters
        ----------
        bar
            Any object with an update method.

        Returns
        -------

        """
        # get progress bar
        bar = self.get_progress_bar(bar)
        # get the iterator
        for num, path in enumerate(self._unindexed_iterator(sub_paths)):
            # update bar if count is in update interval
            if bar is not None and num % self._bar_update_interval == 0:
                bar.update(num)
            yield path
        # finish progress bar
        getattr(bar, "finish", lambda: None)()  # call finish if bar exists

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
        num_files = sum([1 for _ in self._unindexed_iterator()])
        if num_files < self._min_files_for_bar:  # not enough files to use bar
            return None
        # instantiate bar and return
        kwargs = {"min_value": self._min_files_for_bar, "max_value": num_files}
        # an instance should be init'ed
        if isinstance(bar, type) and issubclass(bar, ProgressBar):
            return bar(**kwargs)
        elif bar is None:
            return get_progressbar(**kwargs)
        else:
            msg = f"{bar} is not a valid input for get_progress_bar"
            raise ValueError(msg)

    def clear_cache(self):
        """
        Clear the index cache if the bank is using one.
        """
        if self._index_cache is not None:
            self._index_cache.clear_cache()

    @property
    def _max_workers(self):
        """
        Return the max number of workers allowed by the executor.

        If the Executor has no attribute `_max_workers` use the number of
        CPUs instead. If there is no executor assigned to bank instance
        return 1.
        """
        executor = getattr(self, "executor", None)
        if executor is not None:
            return getattr(executor, "_max_workers", CPU_COUNT)
        return 1

    def _map(self, func, args, chunksize=None):
        """
        Map the args to function, using executor if defined else perform
        in serial.
        """
        if self.executor is not None:
            return self.executor.map(func, args, chunksize=chunksize)
        else:
            return (func(x) for x in args)
