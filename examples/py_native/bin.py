"""A tiny example binary for the native Python rules of Bazel."""
from examples.py_native.lib import GetNumber
from fib import Fib

print "The number is %d" % GetNumber()
print "Fib(5) == %d" % Fib(5)
