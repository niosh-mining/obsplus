"""
Utilities for working with Dict and Json like things.
"""
import collections.abc
from typing import Union, Sequence, Mapping, Optional, Dict, Tuple, Any

from obsplus.interfaces import TreeSpecCallable


class FrozenDict(collections.abc.Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete
    :py:class:`collections.Mapping` interface. It can be used as a drop-in
    replacement for dictionaries where faux-immutability is desired.

    Notes
    -----
    This implementation was Inspired by the no-longer maintained package
    frozen-dict (https://github.com/slezica/python-frozendict)

    By design, changes in the original dict are not reflected in the frozen
    dict so that the hash doesn't break.

    We can't simply use types.MappingProxyType because it can't be pickled.
    """

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        """Perform a shallow copy on the dictionaries contents."""
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._dict)

    def _hash_contents(self):
        """Returns a hash of the dictionary"""
        out = 0
        for key, value in self._dict.items():
            out ^= hash((key, value))
        return out

    def __hash__(self):
        if self._hash is None:
            self._hash = self._hash_contents()
        return self._hash


def resolve_tree_specs(
    data: Union[dict, list],
    specs: Sequence[Union[int, str, TreeSpecCallable]],
    operation_stash: Optional[Mapping[Tuple, Any]] = None,
) -> Dict[tuple, Union[dict, list]]:
    """
    Get the nodes of a tree-like structure given an address of tuples.

    Parameters
    ----------
    data
        Json-like structure of dicts, arrays, etc.
    specs
        A single tuple or list of specifications. A specification is a
        str or int (for get_item) or a callable which returns a dict of
        {address: object}.
    operation_stash
        A stash for storing the results of operations on addresses.

    Notes
    -----
    The spec can be ambiguous if any nested dicts have int keys.
    """

    def _get_addresses():
        """Return address as a list of addresses."""
        if len(specs) and isinstance(specs[0], (str, int, float)):
            return [specs]
        return specs

    def _handle_callables(address, current_op, new_spec, obj):
        """Handle callables in spec."""
        op_key = tuple(address + [current_op])
        if op_key not in operation_stash:
            operation_stash[op_key] = current_op(new_spec, address, obj)
        out = [(new_spec, a, o) for (_, a, o) in operation_stash[op_key]]
        return out

    def _handle_get_item(address, current_op, obj, new_spec):
        new_address = list(address) + [current_op]
        try:
            attr = obj[current_op]
        except (KeyError, IndexError):
            return []  # nothing to return
        else:
            return [(new_spec, new_address, attr)]

    def _recursive_getter(spec_address_obj_list):
        """Recurse data, return addresses"""
        out = []
        for spec, address, obj in spec_address_obj_list:
            if not len(spec):  # This spec has been exhausted, store and continue
                out.append((spec, address, obj))
                continue
            current_op, new_spec = spec[0], spec[1:]
            # current operation is a function
            if callable(current_op):
                result = _handle_callables(address, current_op, new_spec, obj)
            # if this is list/tuple we need to apply op to each element
            elif not isinstance(current_op, int) and isinstance(obj, Sequence):
                result = [([num] + list(spec), address, obj) for num in range(len(obj))]
            # otherwise this should be a simple get item
            else:
                result = _handle_get_item(address, current_op, obj, new_spec)
            out += _recursive_getter(result)
        return out

    operation_stash = {} if operation_stash is None else operation_stash
    specs = _get_addresses()
    inputs = [(spec, (), data) for spec in specs]
    return {tuple(add): obj for _, add, obj in _recursive_getter(inputs)}
