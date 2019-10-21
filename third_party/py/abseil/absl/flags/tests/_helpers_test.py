# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unittests for helpers module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl.flags import _helpers
from absl.flags.tests import module_bar
from absl.flags.tests import module_foo
from absl.testing import absltest


class FlagSuggestionTest(absltest.TestCase):

  def setUp(self):
    self.longopts = [
        'fsplit-ivs-in-unroller=',
        'fsplit-wide-types=',
        'fstack-protector=',
        'fstack-protector-all=',
        'fstrict-aliasing=',
        'fstrict-overflow=',
        'fthread-jumps=',
        'ftracer',
        'ftree-bit-ccp',
        'ftree-builtin-call-dce',
        'ftree-ccp',
        'ftree-ch']

  def test_damerau_levenshtein_id(self):
    self.assertEqual(0, _helpers._damerau_levenshtein('asdf', 'asdf'))

  def test_damerau_levenshtein_empty(self):
    self.assertEqual(5, _helpers._damerau_levenshtein('', 'kites'))
    self.assertEqual(6, _helpers._damerau_levenshtein('kitten', ''))

  def test_damerau_levenshtein_commutative(self):
    self.assertEqual(2, _helpers._damerau_levenshtein('kitten', 'kites'))
    self.assertEqual(2, _helpers._damerau_levenshtein('kites', 'kitten'))

  def test_damerau_levenshtein_transposition(self):
    self.assertEqual(1, _helpers._damerau_levenshtein('kitten', 'ktiten'))

  def test_mispelled_suggestions(self):
    suggestions = _helpers.get_flag_suggestions('fstack_protector_all',
                                                self.longopts)
    self.assertEqual(['fstack-protector-all'], suggestions)

  def test_ambiguous_prefix_suggestion(self):
    suggestions = _helpers.get_flag_suggestions('fstack', self.longopts)
    self.assertEqual(['fstack-protector', 'fstack-protector-all'], suggestions)

  def test_misspelled_ambiguous_prefix_suggestion(self):
    suggestions = _helpers.get_flag_suggestions('stack', self.longopts)
    self.assertEqual(['fstack-protector', 'fstack-protector-all'], suggestions)

  def test_crazy_suggestion(self):
    suggestions = _helpers.get_flag_suggestions('asdfasdgasdfa', self.longopts)
    self.assertEqual([], suggestions)

  def test_suggestions_are_sorted(self):
    sorted_flags = sorted(['aab', 'aac', 'aad'])
    misspelt_flag = 'aaa'
    suggestions = _helpers.get_flag_suggestions(misspelt_flag,
                                                reversed(sorted_flags))
    self.assertEqual(sorted_flags, suggestions)


class GetCallingModuleTest(absltest.TestCase):
  """Test whether we correctly determine the module which defines the flag."""

  def test_get_calling_module(self):
    self.assertEqual(_helpers.get_calling_module(), sys.argv[0])
    self.assertEqual(module_foo.get_module_name(),
                     'absl.flags.tests.module_foo')
    self.assertEqual(module_bar.get_module_name(),
                     'absl.flags.tests.module_bar')

    # We execute the following exec statements for their side-effect
    # (i.e., not raising an error).  They emphasize the case that not
    # all code resides in one of the imported modules: Python is a
    # really dynamic language, where we can dynamically construct some
    # code and execute it.
    code = ('from absl.flags import _helpers\n'
            'module_name = _helpers.get_calling_module()')
    exec(code)  # pylint: disable=exec-used

    # Next two exec statements executes code with a global environment
    # that is different from the global environment of any imported
    # module.
    exec(code, {})  # pylint: disable=exec-used
    # vars(self) returns a dictionary corresponding to the symbol
    # table of the self object.  dict(...) makes a distinct copy of
    # this dictionary, such that any new symbol definition by the
    # exec-ed code (e.g., import flags, module_name = ...) does not
    # affect the symbol table of self.
    exec(code, dict(vars(self)))  # pylint: disable=exec-used

    # Next test is actually more involved: it checks not only that
    # get_calling_module does not crash inside exec code, it also checks
    # that it returns the expected value: the code executed via exec
    # code is treated as being executed by the current module.  We
    # check it twice: first time by executing exec from the main
    # module, second time by executing it from module_bar.
    global_dict = {}
    exec(code, global_dict)  # pylint: disable=exec-used
    self.assertEqual(global_dict['module_name'],
                     sys.argv[0])

    global_dict = {}
    module_bar.execute_code(code, global_dict)
    self.assertEqual(global_dict['module_name'],
                     'absl.flags.tests.module_bar')

  def test_get_calling_module_with_iteritems_error(self):
    # This test checks that get_calling_module is using
    # sys.modules.items(), instead of .iteritems().
    orig_sys_modules = sys.modules

    # Mock sys.modules: simulates error produced by importing a module
    # in paralel with our iteration over sys.modules.iteritems().
    class SysModulesMock(dict):

      def __init__(self, original_content):
        dict.__init__(self, original_content)

      def iteritems(self):
        # Any dictionary method is fine, but not .iteritems().
        raise RuntimeError('dictionary changed size during iteration')

    sys.modules = SysModulesMock(orig_sys_modules)
    try:
      # _get_calling_module should still work as expected:
      self.assertEqual(_helpers.get_calling_module(), sys.argv[0])
      self.assertEqual(module_foo.get_module_name(),
                       'absl.flags.tests.module_foo')
    finally:
      sys.modules = orig_sys_modules


class IsBytesOrString(absltest.TestCase):

  def test_bytes(self):
    self.assertTrue(_helpers.is_bytes_or_string(b'bytes'))

  def test_str(self):
    self.assertTrue(_helpers.is_bytes_or_string('str'))

  def test_unicode(self):
    self.assertTrue(_helpers.is_bytes_or_string(u'unicode'))

  def test_list(self):
    self.assertFalse(_helpers.is_bytes_or_string(['str']))


if __name__ == '__main__':
  absltest.main()
