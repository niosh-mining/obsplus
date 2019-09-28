"""
A collection of Obsplus Exceptions.
"""


class BankDoesNotExistError(FileNotFoundError):
    """
    Exception raised when the user tries to read data from a bank but the
    bank directory does not exist.
    """


class BankIndexLockError(OSError):
    """
    Raised when a bank tries to read an index but the lock file is not deleted.
    """


class FileHashChangedError(ValueError):
    """
    Raised when the expected md5 hash of a file has changed.
    """


class MissingDataFileError(FileNotFoundError):
    """
    Raised when the expected md5 hash of a file has changed.
    """


class DataVersionError(ValueError):
    """
    Raised when the version of a dataset doesn't match what is expected
    """


class ValidationError(ValueError):
    """
    Raised when something goes wrong with object validation.
    """


class ValidationNameError(ValidationError, KeyError):
    """
    Raised when a namespace with no validators is used.
    """
