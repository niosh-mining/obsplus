"""
Core modules for validate.
"""
import inspect
from collections import defaultdict
from contextlib import contextmanager
from functools import singledispatch
from itertools import product
from typing import Optional, Union

import obspy
import obspy.core.event as ev
import pandas as pd

from obsplus.exceptions import ValidationNameError
from obsplus.utils import yield_obj_parent_attr, iterate

# The validator state is of the form:
# {"namespace": {cls: [validator1, validator2, ...], ...}, ...}


_VALIDATOR_STATE = dict(validators=defaultdict(lambda: defaultdict(list)))


def _create_decompose_func():
    """ Create a single dispatch function for decomposing objects. """

    @singledispatch
    def decompose(obj):
        """
        Decompose an object into its constitutive parts.

        This function should return a dict of the form:
            {cls: [instance1, instance2, instance3, ...]}
        """
        return {type(obj): [obj]}

    return decompose


_VALIDATOR_STATE["decomposer"] = _create_decompose_func()


@contextmanager
def _temp_validate_namespace():
    """
    A contextmanager for setting temporary validator state for testing.
    """
    old_validators = _VALIDATOR_STATE["validators"]
    old_decomposer = _VALIDATOR_STATE["decomposer"]
    _VALIDATOR_STATE["validators"] = defaultdict(lambda: defaultdict(list))
    _VALIDATOR_STATE["decomposer"] = _create_decompose_func()
    yield
    _VALIDATOR_STATE["validators"] = old_validators
    _VALIDATOR_STATE["decomposer"] = old_decomposer


def decomposer(cls):
    """
    Register a function as a decomposer for a given class.

    The decomposer simply splits the object into its constitutive parts
    that may need to be tested. It should return a dict of the following form:
        {cls: [instance1, instance2, ...], ... }

    Parameters
    ----------
    cls

    Returns
    -------

    """

    def _wrap(func):
        _decomposer = _VALIDATOR_STATE["decomposer"]
        for cls_ in iterate(cls):
            _decomposer.register(cls_)(func)
        return func

    return _wrap


def validator(namespace: str, cls: type):
    """
    Register a callable to a given namespace to operate on type cls.

    Parameters
    ----------
    namespace

    cls

    Returns
    -------

    """

    def _wrap(func):
        # register function and return it
        state = _VALIDATOR_STATE["validators"]
        state[namespace][cls].append(func)
        return func

    return _wrap


def _get_validators(namespace):
    """ Get validators for a specified namespace if sound else raise. """
    validator_namespaces = _VALIDATOR_STATE["validators"]
    if namespace not in validator_namespaces:
        msg = f"{namespace} has no registered validators"
        raise ValidationNameError(msg)
    return validator_namespaces[namespace]


def _get_validate_obj_intersection(validators, obj_tree):
    """ get the intersection between validators and obj_tree. """
    validators_to_run = defaultdict(list)
    for cls1, cls2 in product(obj_tree, validators):
        if issubclass(cls1, cls2):
            validators_to_run[cls1].extend(validators[cls2])
    return validators_to_run


def _make_validator_report(validator, obj, kwargs):
    """ run a validator against an object and make a report. """
    val_name = getattr(validator, "__name__", validator)
    out = {"validator": val_name, "object": obj, "message": ""}
    try:
        _run_validator(validator, obj, kwargs)
    except AssertionError as e:
        msg = f"validator {val_name} failed object: {obj} raising message:\n {e}"
        out["passed"] = False
        out["message"] = msg
    else:
        out["passed"] = True
    return out


def _run_validator(validator, obj, kwargs):
    """ Run the validator. """
    if kwargs:
        validator(obj, **kwargs)
    else:
        validator(obj)


def validate(
    obj, namespace: str, report: bool = False, **kwargs
) -> Optional[pd.DataFrame]:
    """
    Validate an object using validators in specified namespace.

    Parameters
    ----------
    obj
        Any function whose type has registered validators.
    namespace
        The validation namespace specified by validators.
    report
        If True, return a dataframe which passed/failed validators and objects
        and suppress all failures.

    Notes
    -----
    Parameters can be passed individual validators using kwargs.
    """
    # get validators and decompose object to testable parts.
    validators_raw = _get_validators(namespace)
    obj_tree = _VALIDATOR_STATE["decomposer"](obj)
    # get validators to run on obj_tree
    validators = _get_validate_obj_intersection(validators_raw, obj_tree)
    # iterate over each validator run it.
    reports = []
    for cls, validate_funcs in validators.items():
        for validator, obj in product(validate_funcs, obj_tree.get(cls, [])):
            # get the main arg and any addition requested kwargs
            arg, *wanted_args = list(inspect.signature(validator).parameters)
            vkwargs = {x: kwargs[x] for x in set(wanted_args) & set(kwargs)}
            # make report and shallow assertion errors if report requested
            if report:
                reports.append(_make_validator_report(validator, obj, vkwargs))
            else:  # else just run the validator
                _run_validator(validator, obj, vkwargs)
    return pd.DataFrame(reports)


# register common decomposers


@decomposer((obspy.Catalog, ev.Event))
def _decompose_event_or_catalog(events):
    """ Decompose an event or a catalog. """
    out = defaultdict(list)
    for obj, parent, attr in yield_obj_parent_attr(events):
        out[type(obj)].append(obj)
    return out
