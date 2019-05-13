"""
Recreates the hash files for the datasets.
"""
import obspy
import obsplus

dataset_names = ('bingham', 'crandall', 'TA', 'kemmerer')

for name in dataset_names:
    ds = obsplus.load_dataset(name)
    ds.create_md5_hash()


