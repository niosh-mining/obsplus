"""
Tests for the validate module.
"""
from collections import defaultdict

import pytest

from obsplus.validate import validate, decompose, temp_validate_namespace, validator


class TestThing1:
    def __init__(self, a):
        self.a = a


class TestThing2:
    def __init__(self, b):
        self.b = b


class TestThing3(TestThing1, TestThing2):
    def __init_(self, a, b):
        self.a = a
        self.b = b


@pytest.fixture(scope="class")
def temporary_validate_space():
    with temp_validate_namespace() as state:
        yield state


class TestValidateBasics:
    @pytest.fixture
    def register_validators(self):

        outdist = defaultdict(int)

        @validator("test1", TestThing1)
        def first_validator(obj):
            outdist["test1"] += 1

        @validator("test1", TestThing2)
        def second_validor(obj):
            outdist["test1"] += 1

            pass

        @validator("test1", TestThing3)
        def third_validator(obj):
            outdist["test1"] += 1
