"""
Recreates the hash files for the datasets.
"""
import obsplus

dataset_names = ("bingham_test", "crandall_test", "ta_test")

for name in dataset_names:
    ds = obsplus.load_dataset(name)
    ds.create_sha256_hash()
