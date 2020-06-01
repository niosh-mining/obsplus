"""
Misc. ObsPlus utilities.
"""
import contextlib
import fnmatch
import hashlib
import os
import warnings
from functools import wraps, partial, singledispatch
from os.path import join
from pathlib import Path, PurePosixPath
from typing import (
    Generator,
    Tuple,
    Any,
    Set,
    Optional,
    Callable,
    Union,
    Sequence,
    TypeVar,
    Collection,
    Dict,
    Iterable,
)

import numpy as np
import obspy
import pandas as pd
from obspy.core import event as ev
from obspy.core.inventory import Station, Channel
from obspy.io.mseed.core import _read_mseed as mread
from obspy.io.quakeml.core import _read_quakeml

from obsplus.constants import NULL_SEED_CODES, NSLC

BASIC_NON_SEQUENCE_TYPE = (int, float, str, bool, type(None))
READ_DICT = dict(mseed=mread, quakeml=_read_quakeml)


def _get_progressbar():
    """Suppress ProgressBar's warning."""
    # TODO remove this when progress no longer issues warning
    with suppress_warnings():
        from progressbar import ProgressBar
    return ProgressBar


def deprecated_callable(func=None, replacement_str=None):
    """
    Mark a function as deprecated.

    Whenever it is used a userwarning will be issued. You can optionally
    provide a string to indicate which function should be used in its place.

    Parameters
    ----------
    func
    replacement_str

    Returns
    -------

    """
    fname = str(getattr(func, "__name__", func))

    if callable(func):

        @wraps(func)
        def _wrap(*args, **kwargs):
            msg = f"{fname} is deprecated and will be removed in a future release."
            if replacement_str:
                msg += f" Please use {replacement_str} instead."
            warnings.warn(msg)
            return func(*args, **kwargs)

        return _wrap
    else:
        return partial(deprecated_callable, replacement_str=replacement_str)


def yield_obj_parent_attr(
    obj, cls=None, is_attr=None, has_attr=None, basic_types=False
) -> Generator[Tuple[Any, Any, str], None, None]:
    """
    Recurse an object, yield a tuple of object, parent, attr.

    Useful when data need to be changed or the provided DataFrame extractors
    don't quite perform the desired task. Can also be used to extract
    relationships between entities in object trees to build a connecting graph.

    Parameters
    ----------
    obj
        The object to recurse through attributes of lists, tuples, and other
        instances.
    cls
        Only return instances of cls if not None, dont filter on types.
    is_attr
        Only return objects stored as attr_name, if None return all.
    has_attr
        Only return objects that have attribute has_attr, if None return all.
    basic_types
        If True, yield non-sequence basic types (int, float, str, bool).

    Examples
    --------
    >>> # --- get all picks from a catalog object
    >>> import obsplus
    >>> import obspy.core.event as ev
    >>> cat = obsplus.load_dataset('bingham_test').event_client.get_events()
    >>> picks = []  # put all the picks in a list.
    >>> for pick, _, _ in yield_obj_parent_attr(cat, cls=ev.Pick):
    ...     picks.append(pick)
    >>> assert len(picks)

    >>> # --- yield all objects which have resource identifiers
    >>> objects = []  # list of (rid, parent)
    >>> RID = ev.ResourceIdentifier
    >>> for rid, parent, attr in yield_obj_parent_attr(cat, cls=RID):
    ...     objects.append((str(rid), parent))
    >>> assert len(objects)

    >>> # --- Create a dict of {resource_id: [(attr, parent), ...]}
    >>> from collections import defaultdict
    >>> rid_mapping = defaultdict(list)
    >>> for rid, parent, attr in yield_obj_parent_attr(cat, cls=RID):
    ...     rid_mapping[str(rid)].append((attr, parent))
    >>> # count how many times each resource_id is referred to
    >>> count = {i: len(v) for i, v in rid_mapping.items()}
    """
    ids: Set[int] = set()  # id cache to avoid circular references

    def func(obj, attr=None, parent=None):
        id_tuple = (id(obj), id(parent))

        # If object/parent combo have not been yielded continue.
        if id_tuple in ids:
            return
        ids.add(id_tuple)
        # Check if this object is stored as the desired attribute.
        is_attribute = is_attr is None or attr == is_attr
        # Check if the object has the desired attribute.
        has_attribute = has_attr is None or hasattr(obj, has_attr)
        # Check if isinstance of desired class.
        is_instance = cls is None or isinstance(obj, cls)
        # Check if basic type (dont
        is_basic = basic_types or not isinstance(obj, BASIC_NON_SEQUENCE_TYPE)
        # Iterate through basic built-in types.
        if isinstance(obj, (list, tuple)):
            for val in obj:
                yield from func(val, attr=attr, parent=parent)
        elif isinstance(obj, dict):
            for item, val in obj.items():
                yield from func(val, attr=item, parent=obj)
        # Yield object, parent, and attr if desired conditions are met.
        elif is_attribute and has_attribute and is_instance and is_basic:
            yield (obj, parent, attr)
        # Iterate through non built-in object attributes.
        if hasattr(obj, "__slots__"):
            for attr in obj.__slots__:
                val = getattr(obj, attr)
                yield from func(val, attr=attr, parent=obj)
        if hasattr(obj, "__dict__"):
            for item, val in obj.__dict__.items():
                yield from func(val, attr=item, parent=obj)

    return func(obj)


def get_instances_from_tree(object, cls):
    """
    Get all instances in an object tree.

    Simply uses :func:`~obsplus.utils.misc.yield_obj_parent_attr` under the
    hood.
    """
    return [x for x, _, _ in yield_obj_parent_attr(object, cls=cls)]


def try_read_catalog(catalog_path, **kwargs):
    """ Try to read a events from file, if it raises return None """
    read = READ_DICT.get(kwargs.pop("format", None), obspy.read_events)
    try:
        cat = read(catalog_path, **kwargs)
    except Exception:
        warnings.warn(f"obspy failed to read {catalog_path}")
    else:
        if cat is not None and len(cat):
            return cat
    return None


def read_file(file_path, funcs=(pd.read_csv,)) -> Optional[Any]:
    """
    For a given file_path, try reading it with each function in funcs.

    Parameters
    ----------
    file_path
        The path to the file to read
    funcs
        A tuple of functions to try to read the file (starting with first)

    """
    for func in funcs:
        assert callable(func)
        try:
            return func(file_path)
        except Exception:
            pass
    raise IOError(f"failed to read {file_path}")


def apply_to_files_or_skip(func: Callable, directory: Union[str, Path]):
    """
    Generator for applying func to all files in directory.

    Skip any files that raise an exception.

    Parameters
    ----------
    func
        Any callable that takes a file path as the only input.

    directory
        A directory that exists.

    Yields
    ------
    outputs of func
    """
    path = Path(directory)
    assert path.is_dir(), f"{directory} is not a directory"
    for fi in path.rglob("*"):
        if os.path.isfile(fi):
            try:
                yield func(fi)
            except Exception:
                pass


def get_progressbar(max_value, min_value=None, *args, **kwargs) -> Optional:
    """
    Get a progress bar object using the ProgressBar2 library.

    Fails gracefully if bar cannot be displayed (eg if no std out).
    Args and kwargs are passed to ProgressBar constructor.

    Parameters
    ----------
    max_value
        The highest number expected
    min_value
        The minimum number of updates required to show the bar
    """

    def _new_update(bar):
        """ A new update function that swallows attribute and index errors """
        old_update = bar.update

        def update(value=None, force=False, **kwargs):
            with contextlib.suppress((IndexError, ValueError, AttributeError)):
                old_update(value=value, force=force, **kwargs)

        return update

    if min_value and max_value < min_value:
        return None  # no progress bar needed, return None
    try:
        ProgressBar = _get_progressbar()
        bar = ProgressBar(max_value=max_value, *args, **kwargs)
        bar.start()
        bar.update = _new_update(bar)
        bar.update(1)
    except Exception:  # this can happen when stdout is being redirected
        return None  # something went wrong, return None
    return bar


def iterate(obj):
    """
    Return an iterable from any object.

    If string, do not iterate characters, return str in tuple .
    """
    if obj is None:
        return ()
    if isinstance(obj, str):
        return (obj,)
    return obj if isinstance(obj, Sequence) else (obj,)


class DummyFile(object):
    """ Dummy class to mock std out interface but go nowhere. """

    def write(self, x):
        """ do nothing """

    def flush(self):
        """ do nothing """


def getattrs(obj: object, col_set: Collection, default_value: object = np.nan) -> dict:
    """
    Parse an object for a collection of attributes, return a dict of values.

    If obj does not have a requested attribute, or if its value is None, fill
    with the default value.

    Parameters
    ----------
    obj
        Any object.
    col_set
        A sequence of attributes to extract from obj.
    default_value
        If not attribute is found fill with this value.
    """
    out = {}
    if obj is None:  # return empty dict if None
        return out
    for item in col_set:
        try:
            val = getattr(obj, item)
        except (ValueError, AttributeError):
            val = default_value
        if val is None:
            val = default_value
        out[item] = val
    return out


any_type = TypeVar("any_type")


@singledispatch
def replace_null_nlsc_codes(
    obspy_object: any_type, null_codes=NULL_SEED_CODES, replacement_value=""
) -> any_type:
    """
    Iterate an obspy object and replace nullish nslc codes with some value.

    Operates in place, but also returns the original object.

    Parameters
    ----------
    obspy_object
        An obspy catalog, event, (or any sub element), stream, trace,
        inventory, etc.
    null_codes
        The codes that are considered null values and should be replaced.
    replacement_value
        The value with which to replace the null_codes.
    """
    wid_codes = tuple(x + "_code" for x in NSLC)
    for wid, _, _ in yield_obj_parent_attr(obspy_object, cls=ev.WaveformStreamID):
        for code in wid_codes:
            if getattr(wid, code) in null_codes:
                setattr(wid, code, replacement_value)
    return obspy_object


@replace_null_nlsc_codes.register(obspy.Stream)
def _replace_null_stream(st, null_codes=NULL_SEED_CODES, replacement_value=""):
    for tr in st:
        _replace_null_trace(tr, null_codes, replacement_value)
    return st


@replace_null_nlsc_codes.register(obspy.Trace)
def _replace_null_trace(tr, null_codes=NULL_SEED_CODES, replacement_value=""):
    for code in NSLC:
        val = getattr(tr.stats, code)
        if val in null_codes:
            setattr(tr.stats, code, replacement_value)
    return tr


@replace_null_nlsc_codes.register(obspy.Inventory)
@replace_null_nlsc_codes.register(Station)
@replace_null_nlsc_codes.register(Channel)
def _replace_inv_nulls(inv, null_codes=NULL_SEED_CODES, replacement_value=""):
    for code in ["location_code", "code"]:
        for obj, _, _ in yield_obj_parent_attr(inv, has_attr=code):
            if getattr(obj, code) in null_codes:
                setattr(obj, code, replacement_value)
    return inv


def iter_files(
    paths: Union[str, Iterable[str]],
    ext: Optional[str] = None,
    mtime: Optional[float] = None,
    skip_hidden: bool = True,
) -> Iterable[str]:
    """
    use os.scan dir to iter files, optionally only for those with given
    extension (ext) or modified times after mtime

    Parameters
    ----------
    paths
        The path to the base directory to traverse. Can also use a collection
        of paths.
    ext : str or None
        The extensions to map.
    mtime : int or float
        Time stamp indicating the minimum mtime.
    skip_hidden : bool
        If True skip files or folders (they begin with a '.')

    Yields
    ------
    Paths, as strings, meeting requirements.
    """
    try:  # a single path was passed
        for entry in os.scandir(paths):
            if entry.is_file() and (ext is None or entry.name.endswith(ext)):
                if mtime is None or entry.stat().st_mtime >= mtime:
                    if entry.name[0] != "." or not skip_hidden:
                        yield entry.path
            elif entry.is_dir() and not (skip_hidden and entry.name[0] == "."):
                yield from iter_files(
                    entry.path, ext=ext, mtime=mtime, skip_hidden=skip_hidden
                )
    except TypeError:  # multiple paths were passed
        for path in paths:
            yield from iter_files(path, ext, mtime, skip_hidden)
    except NotADirectoryError:  # a file path was passed, just return it
        yield paths


def hash_file(path: Union[str, Path]):
    """
    Calculate the sha256 hash of a file.

    Reads the file in chunks to allow using large files. Taken from this stack
    overflow answer: http://bit.ly/2Jqb1Jr

    Parameters
    ----------
    path
        The path to the file to read.

    Returns
    -------
    A str of hex for file hash

    """
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_directory(
    path: Union[Path, str],
    match: str = "*",
    exclude: Optional[Union[str, Collection[str]]] = None,
    hidden=False,
) -> Dict[str, str]:
    """
    Calculate the sha256 hash of all files in a directory.

    Parameters
    ----------
    path
        The path to the directory
    match
        A unix-style matching string
    exclude
        A list of unix style strings to exclude
    hidden
        If True skip all files starting with a .

    Returns
    -------
    A dict containing paths and md5 hashes.
    """
    path = Path(path)
    out = {}
    excludes = iterate(exclude)
    for sub_path in path.rglob(match):
        keep = True
        # skip directories
        if sub_path.is_dir():
            continue
        # skip if matches on exclusion
        for exc in excludes:
            if fnmatch.fnmatch(sub_path.name, exc):
                keep = False
                break
        if not hidden and sub_path.name.startswith("."):
            keep = False
        if keep:
            relative_path = sub_path.relative_to(path)
            out[str(PurePosixPath(relative_path))] = hash_file(sub_path)
    return out


def _get_path(info, path, name, path_struct, name_strcut):
    """return a dict with path, and file name"""
    if path is None:  # if the path needs to be created
        ext = info.get("ext", "")
        # get name
        fname = name or name_strcut.format_map(info)
        fname = fname if fname.endswith(ext) else fname + ext  # add ext
        # get structure
        psplit = path_struct.format_map(info).split("/")
        path = join(*psplit, fname)
        out_name = fname
    else:  # if the path is already known
        out_name = os.path.basename(path)
    return dict(path=path, filename=out_name)


@contextlib.contextmanager
def suppress_warnings(category=Warning):
    """
    Context manager for suppressing warnings.

    Parameters
    ----------
    category
        The types of warnings to suppress. Must be a subclass of Warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=category)
        yield
    return None


def register_func(list_or_dict: Union[list, dict], key: Optional[str] = None):
    """
    Decorator for registering a function name in a list or dict.

    If list_or_dict is a list only append the name of the function. If it is
    as dict append name (as key) and function as the value.

    Parameters
    ----------
    list_or_dict
        A list or dict to which the wrapped function will be added.
    key
        The name to use, if different than the name of the function.
    """

    def wrapper(func):
        name = key or func.__name__
        if hasattr(list_or_dict, "append"):
            list_or_dict.append(name)
        else:
            list_or_dict[name] = func
        return func

    return wrapper


def validate_version_str(version_str: str):
    """
    Check the version string is of the form x.y.z.

    If the version string is not valid raise a ValueError.
    """
    is_str = isinstance(version_str, str)
    # If version_str is not a str or doesnt have a len of 3
    if not (is_str and len(version_str.split(".")) == 3):
        msg = f"version must be a string of the form x.y.z, not {version_str}"
        raise ValueError(msg)


def get_version_tuple(version_str: str) -> Tuple[int, int, int]:
    """
    Convert a semantic version string to a tuple.

    Parameters
    ----------
    version_str
        A version of the form "x.y.z". Google semantic versioning for more
        details.
    """
    validate_version_str(version_str)
    split = version_str.split(".")
    return int(split[0]), int(split[1]), int(split[2])


class FunctionCacheDescriptor:
    """
    A Simple descriptor for making cached function calls.

    Outputs are stored on instance _{name} where name is assigned to the
    descriptor.
    """

    def __init__(self, func: Callable, func_kwargs: Optional[dict] = None):
        self._func = func
        self._func_kwargs = func_kwargs if func_kwargs is not None else {}

    def __set_name__(self, owner, name):
        self._name = f"_{name}"

    def __get__(self, instance, owner):
        value = getattr(instance, self._name, None)
        if value is None:
            value = self._func(**self._func_kwargs)
            setattr(instance, self._name, value)
        return value
