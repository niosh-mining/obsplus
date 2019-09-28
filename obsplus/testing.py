"""
Simple utilities used for testing.
"""
from collections import Counter
from contextlib import contextmanager


@contextmanager
def instrument_methods(obj):
    """
    Temporarily instrument an objects methods.

    This allows the calls to each of the executors methods to be counted. A
    Counter object is attached to the executor and each of the methods is
    wrapped to count how many times they are called.


    Parameters
    ----------
    obj
    """
    old_methods = {}
    counter = Counter()
    setattr(obj, "_counter", counter)

    for attr in dir(obj):
        # skip dunders
        if attr.startswith("__"):
            continue
        method = getattr(obj, attr, None)
        # skip anything that isnt callable
        if not callable(method):
            continue
        # append old method to old_methods and create new method
        old_methods[attr] = method

        def func(*args, __method_name=attr, **kwargs):
            counter.update({__method_name: 1})
            return old_methods[__method_name](*args, **kwargs)

        setattr(obj, attr, func)

    # yield monkey patched object
    yield obj
    # reset methods
    for attr, method in old_methods.items():
        setattr(obj, attr, method)
    # delete counter
    delattr(obj, "_counter")
