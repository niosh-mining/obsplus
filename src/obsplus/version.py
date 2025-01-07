"""Module for reporting the package version."""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

__version__ = "0.0.0"

# try to get version from installed metadata
with suppress(PackageNotFoundError):
    __version__ = version("obsplus")

__last_version__ = ".".join(__version__.split("dev")[0].split("+")[0].split(".")[:3])
