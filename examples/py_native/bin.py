"""A tiny example binary for the native Python rules of Bazel."""
from examples.py_native.lib import GetNumber

print "The number is %d" % GetNumber()
