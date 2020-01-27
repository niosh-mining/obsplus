"""
Recreates the hash files for the datasets.
"""
import obsplus

dataset_names = ("bingham", "crandall", "TA")

for name in dataset_names:
    ds = obsplus.load_dataset(name)
    ds.create_md5_hash()
