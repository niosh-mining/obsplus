"""
Tests for the validate module.
"""
from collections import defaultdict

import pytest

from obsplus.validate import _temp_validate_namespace, validator, validate
from obsplus.exceptions import ValidationNameError, ValidationError


class Thing1:
    """ The first dummy class for testing. """


class Thing2:
    """ The second dummy class for testing. """


class Thing3(Thing1, Thing2):
    """ The third dummy class for testing. """


class Thing4:
    """ The fourth dummy class. """


@pytest.fixture()
def temporary_validate_space():
    with _temp_validate_namespace() as state:
        yield state


class TestValidateBasics:
    validate_namespace = "_test_validators"

    @pytest.fixture
    def registered_validators(self, temporary_validate_space):
        outdist = defaultdict(int)

        @validator(self.validate_namespace, Thing1)
        def first_validator(obj):
            outdist["test1"] += 1

        @validator(self.validate_namespace, Thing2)
        def second_validator(obj, some_kwarg=None):
            outdist["test2"] += 1
            if some_kwarg:
                raise ValueError(f"some_kwarg: {some_kwarg}")

        @validator(self.validate_namespace, Thing3)
        def third_validator(obj):
            outdist["test3"] += 1

        @validator(self.validate_namespace, Thing4)
        def fourth_validator(obj):
            """ This validator should fail. """
            assert False

        yield outdist

    def test_thing_one(self, registered_validators):
        """ Ensure the validator is triggered on thing one instance. """
        validate(Thing1(), self.validate_namespace)
        assert registered_validators["test1"] == 1

    def test_thing_two(self, registered_validators):
        """
        An instance of thing 3 should trigger all the validators since it
        is a subclass of thing1 and thing2.
        """
        validate(Thing3(), self.validate_namespace)
        for a in range(1, 4):
            assert registered_validators[f"test{a}"] == 1

    def test_failed_validator(self, registered_validators):
        """ Ensure an assertion is raised. """
        with pytest.raises(AssertionError):
            validate(Thing4(), self.validate_namespace)

    def test_failed_validator_report(self, registered_validators):
        """ Ensure a dataframe is returned with a report. """
        df = validate(Thing4(), self.validate_namespace, report=True)
        assert len(df) == 1
        assert not df["passed"].iloc[0]

    def test_kwargs_passed(self, registered_validators):
        """ Ensure the kwargs get passed to individual validators. """
        with pytest.raises(ValueError) as e:
            validate(Thing2(), self.validate_namespace, some_kwarg=True)
