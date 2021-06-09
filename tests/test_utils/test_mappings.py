"""
Simple tests for FrozenDict.
"""
from collections.abc import Mapping

import pytest
import numpy as np

from obsplus.utils.mapping import FrozenDict, resolve_tree_specs


@pytest.fixture(scope="session")
def frozen_dict():
    """Return an example frozen dict."""
    return FrozenDict({"bob": 1, "bill": 2})


class TestFrozenDict:
    """Test chases for frozen dict"""

    def test_is_mapping(self, frozen_dict):
        """Frozen dict should follow mapping ABC."""
        assert isinstance(frozen_dict, Mapping)

    def test_init_on_dict(self):
        """Ensure a dict can be used to init frozendict."""
        out = FrozenDict({"bob": 1})
        assert isinstance(out, FrozenDict)
        assert "bob" in out

    def test_len(self, frozen_dict):
        """Ensure len works."""
        assert len(frozen_dict) == 2

    def test_contains(self, frozen_dict):
        """Ensure contains works"""
        assert "bob" in frozen_dict
        assert "bill" in frozen_dict

    def test_hash(self, frozen_dict):
        """A frozen dict should be a valid key in a dict/set."""
        out = {frozen_dict: 1}
        assert frozen_dict in out

    def test_init_on_keys(self):
        """Ensure dict can be inited with keys as well."""
        out = FrozenDict(bob=1, bill=2)
        assert isinstance(out, FrozenDict)

    def test_cant_add_keys(self, frozen_dict):
        """Ensure keys can't be added to the dict."""
        with pytest.raises(TypeError, match="not support item assignment"):
            frozen_dict["bob"] = 1

        with pytest.raises(TypeError, match="not support item assignment"):
            frozen_dict["new"] = 1

    def test_cant_mutate_original(self, frozen_dict):
        """
        Ensure the original dict can be changed and this does not affect frozen's
        contents.
        """
        original = {"one": 1, "two": 2}
        froz = FrozenDict(original)
        # test adding new key
        assert "three" not in froz
        original["three"] = 3
        assert "three" not in froz
        # test modifying existing key
        original["one"] = 11
        assert froz["one"] == 1


class TestResolveTreeSpec:
    """Tests for transversing tree structures."""

    @pytest.fixture(scope="class")
    def example_dict(self):
        """Return a simple dictionary."""
        out = {
            "scalar": 1,
            "array1": [1, 3, 5],
            "array2": {"array3": [{"a": 2}, {"a": 2}, {"a": 2, "b": 3}]},
            "bob1": [
                {
                    "bob2": [
                        {"bob3": []},
                        {"bob3": {"bob4": 1}},
                        {
                            "bob3": [
                                {"bob4": 1},
                                {"bob4": 2},
                                {"bob4": 3},
                            ]
                        },
                    ]
                }
            ],
        }
        return out

    def test_simplest_case(self, example_dict):
        """Tests for getting scalar addresses."""
        out = resolve_tree_specs(example_dict, ("scalar",))
        assert out == {("scalar",): example_dict["scalar"]}

    def test_array(self, example_dict):
        """Ensure a spec can return an array."""
        out = resolve_tree_specs(example_dict, ("array1",))
        assert out == {("array1",): example_dict["array1"]}

    def test_expand_address(self, example_dict):
        """Tests for more branch fetching."""
        spec = ("array2", "array3", "a")
        expected = {
            ("array2", "array3", 0, "a"): 2,
            ("array2", "array3", 1, "a"): 2,
            ("array2", "array3", 2, "a"): 2,
        }
        out = resolve_tree_specs(example_dict, spec)
        assert out == expected

    def test_nested_resolve_address(self, example_dict):
        """Tests for resolving nested attributes mixed w/ arrays."""
        spec = ("bob1", "bob2", "bob3")
        expected = {
            ("bob1", 0, "bob2", 0, "bob3"): [],
            ("bob1", 0, "bob2", 1, "bob3"): {"bob4": 1},
            ("bob1", 0, "bob2", 2, "bob3"): [
                {"bob4": 1},
                {"bob4": 2},
                {"bob4": 3},
            ],
        }
        out = resolve_tree_specs(example_dict, spec)
        assert out == expected

    def test_function_in_spec(self, example_dict):
        """Tests for a function in the spec tuple."""

        def _mean(spec, address, obj):
            """Take the mean of an address and object."""
            assert isinstance(obj, list)
            return [(spec, list(address) + [np.mean], np.mean(obj))]

        spec = ("array1", _mean)
        expected = {("array1", np.mean): np.mean(example_dict["array1"])}
        assert resolve_tree_specs(example_dict, spec) == expected

    def test_op_key_stash(self, example_dict):
        """Tests for operations getting cached."""

        def _return_self(x):
            """Just return input args."""
            return x

        def _mirror(spec, address, obj):
            """Take the mean of an address and object."""
            assert isinstance(obj, list)
            return [(spec, list(address) + [_return_self], _return_self(obj))]

        specs = [
            ("bob1", "bob2", _mirror, 0, "bob3"),
            ("bob1", "bob2", _mirror, 1, "bob3"),
        ]
        expected = {
            ("bob1", 0, "bob2", _return_self, 0, "bob3"): [],
            ("bob1", 0, "bob2", _return_self, 1, "bob3"): {"bob4": 1},
        }
        out = resolve_tree_specs(example_dict, specs)
        assert out == expected
