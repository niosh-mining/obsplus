"""A collection of ObsPlus Exceptions and Warnings."""


# --- Exceptions


class BankDoesNotExistError(FileNotFoundError):
    """Exception raised when the bank directory does not exist."""


class FileHashChangedError(ValueError):
    """Raised when the expected md5 hash of a file has changed."""


class MissingDataFileError(FileNotFoundError):
    """Raised when the expected md5 hash of a file has changed."""


class DataVersionError(ValueError):
    """Raised when the version of a dataset doesn't match what is expected."""


class ValidationError(ValueError):
    """Raised when something goes wrong with object validation."""


class ValidationNameError(ValidationError, KeyError):
    """Raised when a namespace with no validators is used."""


class DataFrameContentError(ValueError):
    """Raised when something is unexpected in a dataframe's contents."""


class AmbiguousResponseError(ValueError):
    """
    Raised when trying to get a response for an inventory but more than
    one response meets the criteria.
    """


# --- Warnings


class TimeOverflowWarning(UserWarning):
    """Displayed when a large time value is cast into a 64bit ns time stamp."""
