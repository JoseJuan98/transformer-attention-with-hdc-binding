# -*- coding: utf-8 -*-
"""Unit tests for ArgFormatter"""
# Pytest imports
import pytest

# Standard imports
import unittest

# First party imports
from models.arg_formatter import ArgFormatter


class TestFormatter(ArgFormatter):
    component_name = "test"
    catalog = {"test-component": None}


class TestArgFormatter(unittest.TestCase):

    def test_format_arguments_with_str(self):

        expected_arg = "test-component"

        component_type, component_args = TestFormatter.format_arguments(arguments=expected_arg)

        assert component_type == expected_arg
        assert component_args == {}
        assert isinstance(component_type, str)

    def test_format_arguments_with_dict(self):

        args = {"type": "test-component", "arg1": "value1", "arg2": 0.1}

        component_type, component_args = TestFormatter.format_arguments(arguments=args)

        assert component_type == args["type"]
        assert component_args["arg1"] == args["arg1"]
        assert component_args["arg2"] == args["arg2"]

        assert isinstance(component_type, str)
        assert isinstance(component_args, dict)

    def test_format_arguments_raises_value_error(self):

        with pytest.raises(ValueError):
            TestFormatter.format_arguments(arguments="Uknown")
