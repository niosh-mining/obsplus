:orphan:

.. _quickref:

Quick References
################

A brief glance at ObsPlus' main features.

Events
======

.. autosummary::
     :toctree: stubs

     obsplus.get_event_client
     obsplus.EventBank
     obsplus.events_to_df
     obsplus.picks_to_df
     obsplus.json_to_cat
     obsplus.cat_to_json


Stations
========

.. autosummary::
     :toctree: stubs

     obsplus.get_station_client
     obsplus.stations_to_df



Waveforms
=========

.. autosummary::
     :toctree: stubs

     obsplus.get_waveform_client
     obsplus.WaveBank


Datasets
========

.. autosummary::
     :toctree: stubs

     obsplus.datasets.dataset.DataSet
     obsplus.load_dataset
     obsplus.Fetcher


Utils
=====

.. autosummary::
     :toctree: stubs

     obsplus.DataFrameExtractor
     obsplus.utils.yield_obj_parent_attr
     obsplus.utils.time.to_utc
     obsplus.utils.time.to_datetime64
