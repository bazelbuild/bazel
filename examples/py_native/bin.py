# pylint: disable=superfluous-parens
"""A tiny example binary for the native Python rules of Bazel."""
from fib import Fib
from lib import GetNumber

print("The number is %d" % GetNumber())
print("Fib(5) == %d" % Fib(5))
