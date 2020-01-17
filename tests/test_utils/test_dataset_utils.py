"""
Simple tests for the dataset utils.
"""
from pathlib import Path

from obsplus.utils.dataset import _create_opsdata


class TestDirectoryCreated:
    """ tests for creating the ops_data directory. """

    def test_directory_created(self, tmpdir):
        path = Path(tmpdir) / "_ops_data_test"
        _create_opsdata(path)
        assert path.exists()
        assert (path / "README.txt").exists()
