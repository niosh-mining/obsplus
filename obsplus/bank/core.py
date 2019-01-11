"""
Bank ABC
"""
import os
import threading
import warnings
from abc import ABC, abstractmethod
from os.path import join
from pathlib import Path

import pandas as pd
from pandas.io.sql import DatabaseError

import obsplus
from obsplus.bank.utils import iter_files
from obsplus.exceptions import BankDoesNotExistError


class _Bank(ABC):
    """
    The abstract base class for ObsPlus' banks. Used to access local
    archives in a client-like fashion.
    """

    # hdf5 compression defaults
    _complib = "blosc"
    _complevel = 9

    # lock for updating index
    _index_lock = threading.RLock()
    # attributes subclasses need to define
    ext = ""
    bank_path = ""
    namespace = ""
    index_name = ".index.h5"  # name of index file
    # optional str defining the directory structure and file name schemes
    path_structure = None
    name_structure = None
    # the minimum obsplus version. If not met delete index and re-index
    # bump when database schema change.
    _min_version = "0.0.0"

    @abstractmethod
    def read_index(self, **kwargs) -> pd.DataFrame:
        """ read the index filtering on various params """

    @abstractmethod
    def update_index(self):
        """ update the index """

    @abstractmethod
    def last_updated(self):
        """ get the last modified time stored in the index. If
        Not available return None
        """

    #
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

    def _unindexed_file_iterator(self):
        """ return an iterator of potential unindexed waveform files """
        # on rare occasions mtimes
        mtime = self.last_updated - 0.001 if self.last_updated is not None else None
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

    def assert_bank_path_exists(self, create=False):
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
