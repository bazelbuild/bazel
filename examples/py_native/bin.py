# Lint as: python2, python3
# pylint: disable=superfluous-parens
"""A tiny example binary for the native Python rules of Bazel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from examples.py_native.lib import GetNumber
from fib import Fib

print(("The number is %d" % GetNumber()))
print(("Fib(5) == %d" % Fib(5)))
