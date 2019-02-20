"""
A collection of Obsplus Exceptions.
"""


class BankDoesNotExistError(FileNotFoundError):
    """
    Exception raised when the user tries to read data from a bank but the
    bank directory does not exist.
    """
