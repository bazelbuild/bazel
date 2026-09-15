"""A tiny example binary for the native Python rules of Bazel."""
import unittest
from examples.py_native.lib import GetNumber


class TestGetNumber(unittest.TestCase):

  def test_fail(self):
    self.assertEqual(GetNumber(), 0)


if __name__ == '__main__':
  unittest.main()
