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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.flags import _helpers
from absl.testing import absltest

FLAGS = flags.FLAGS


class FlagsUnitTest(absltest.TestCase):
  """Flags formatting Unit Test."""

  def test_get_help_width(self):
    """Verify that get_help_width() reflects _help_width."""
    default_help_width = _helpers._DEFAULT_HELP_WIDTH  # Save.
    self.assertEqual(80, _helpers._DEFAULT_HELP_WIDTH)
    self.assertEqual(_helpers._DEFAULT_HELP_WIDTH, flags.get_help_width())
    _helpers._DEFAULT_HELP_WIDTH = 10
    self.assertEqual(_helpers._DEFAULT_HELP_WIDTH, flags.get_help_width())
    _helpers._DEFAULT_HELP_WIDTH = default_help_width  # restore

  def test_text_wrap(self):
    """Test that wrapping works as expected.

    Also tests that it is using global flags._help_width by default.
    """
    default_help_width = _helpers._DEFAULT_HELP_WIDTH
    _helpers._DEFAULT_HELP_WIDTH = 10

    # Generate a string with length 40, no spaces
    text = ''
    expect = []
    for n in range(4):
      line = str(n)
      line += '123456789'
      text += line
      expect.append(line)

    # Verify we still break
    wrapped = flags.text_wrap(text).split('\n')
    self.assertEqual(4, len(wrapped))
    self.assertEqual(expect, wrapped)

    wrapped = flags.text_wrap(text, 80).split('\n')
    self.assertEqual(1, len(wrapped))
    self.assertEqual([text], wrapped)

    # Normal case, breaking at word boundaries and rewriting new lines
    input_value = 'a b c d e f g h'
    expect = {1: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
              2: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
              3: ['a b', 'c d', 'e f', 'g h'],
              4: ['a b', 'c d', 'e f', 'g h'],
              5: ['a b c', 'd e f', 'g h'],
              6: ['a b c', 'd e f', 'g h'],
              7: ['a b c d', 'e f g h'],
              8: ['a b c d', 'e f g h'],
              9: ['a b c d e', 'f g h'],
              10: ['a b c d e', 'f g h'],
              11: ['a b c d e f', 'g h'],
              12: ['a b c d e f', 'g h'],
              13: ['a b c d e f g', 'h'],
              14: ['a b c d e f g', 'h'],
              15: ['a b c d e f g h']}
    for width, exp in expect.items():
      self.assertEqual(exp, flags.text_wrap(input_value, width).split('\n'))

    # We turn lines with only whitespace into empty lines
    # We strip from the right up to the first new line
    self.assertEqual('', flags.text_wrap('  '))
    self.assertEqual('\n', flags.text_wrap('  \n  '))
    self.assertEqual('\n', flags.text_wrap('\n\n'))
    self.assertEqual('\n\n', flags.text_wrap('\n\n\n'))
    self.assertEqual('\n', flags.text_wrap('\n '))
    self.assertEqual('a\n\nb', flags.text_wrap('a\n  \nb'))
    self.assertEqual('a\n\n\nb', flags.text_wrap('a\n  \n  \nb'))
    self.assertEqual('a\nb', flags.text_wrap('  a\nb  '))
    self.assertEqual('\na\nb', flags.text_wrap('\na\nb\n'))
    self.assertEqual('\na\nb\n', flags.text_wrap('  \na\nb\n  '))
    self.assertEqual('\na\nb\n', flags.text_wrap('  \na\nb\n\n'))

    # Double newline.
    self.assertEqual('a\n\nb', flags.text_wrap(' a\n\n b'))

    # We respect prefix
    self.assertEqual(' a\n b\n c', flags.text_wrap('a\nb\nc', 80, ' '))
    self.assertEqual('a\n b\n c', flags.text_wrap('a\nb\nc', 80, ' ', ''))

    # tabs
    self.assertEqual('a\n b   c',
                     flags.text_wrap('a\nb\tc', 80, ' ', ''))
    self.assertEqual('a\n bb  c',
                     flags.text_wrap('a\nbb\tc', 80, ' ', ''))
    self.assertEqual('a\n bbb c',
                     flags.text_wrap('a\nbbb\tc', 80, ' ', ''))
    self.assertEqual('a\n bbbb    c',
                     flags.text_wrap('a\nbbbb\tc', 80, ' ', ''))
    self.assertEqual('a\n b\n c\n d',
                     flags.text_wrap('a\nb\tc\td', 3, ' ', ''))
    self.assertEqual('a\n b\n c\n d',
                     flags.text_wrap('a\nb\tc\td', 4, ' ', ''))
    self.assertEqual('a\n b\n c\n d',
                     flags.text_wrap('a\nb\tc\td', 5, ' ', ''))
    self.assertEqual('a\n b   c\n d',
                     flags.text_wrap('a\nb\tc\td', 6, ' ', ''))
    self.assertEqual('a\n b   c\n d',
                     flags.text_wrap('a\nb\tc\td', 7, ' ', ''))
    self.assertEqual('a\n b   c\n d',
                     flags.text_wrap('a\nb\tc\td', 8, ' ', ''))
    self.assertEqual('a\n b   c\n d',
                     flags.text_wrap('a\nb\tc\td', 9, ' ', ''))
    self.assertEqual('a\n b   c   d',
                     flags.text_wrap('a\nb\tc\td', 10, ' ', ''))

    # multiple tabs
    self.assertEqual('a       c',
                     flags.text_wrap('a\t\tc', 80, ' ', ''))

    _helpers._DEFAULT_HELP_WIDTH = default_help_width  # restore

  def test_doc_to_help(self):
    self.assertEqual('', flags.doc_to_help('  '))
    self.assertEqual('', flags.doc_to_help('  \n  '))
    self.assertEqual('a\n\nb', flags.doc_to_help('a\n  \nb'))
    self.assertEqual('a\n\n\nb', flags.doc_to_help('a\n  \n  \nb'))
    self.assertEqual('a b', flags.doc_to_help('  a\nb  '))
    self.assertEqual('a b', flags.doc_to_help('\na\nb\n'))
    self.assertEqual('a\n\nb', flags.doc_to_help('\na\n\nb\n'))
    self.assertEqual('a b', flags.doc_to_help('  \na\nb\n  '))
    # Different first line, one line empty - erm double new line.
    self.assertEqual('a b c\n\nd', flags.doc_to_help('a\n  b\n  c\n\n  d'))
    self.assertEqual('a b\n      c d', flags.doc_to_help('a\n  b\n  \tc\n  d'))
    self.assertEqual('a b\n      c\n      d',
                     flags.doc_to_help('a\n  b\n  \tc\n  \td'))

  def test_doc_to_help_flag_values(self):
    # !!!!!!!!!!!!!!!!!!!!
    # The following doc string is taken as is directly from flags.py:FlagValues
    # The intention of this test is to verify 'live' performance
    # !!!!!!!!!!!!!!!!!!!!
    """Used as a registry for 'Flag' objects.

    A 'FlagValues' can then scan command line arguments, passing flag
    arguments through to the 'Flag' objects that it owns.  It also
    provides easy access to the flag values.  Typically only one
    'FlagValues' object is needed by an application:  flags.FLAGS

    This class is heavily overloaded:

    'Flag' objects are registered via __setitem__:
         FLAGS['longname'] = x   # register a new flag

    The .value member of the registered 'Flag' objects can be accessed as
    members of this 'FlagValues' object, through __getattr__.  Both the
    long and short name of the original 'Flag' objects can be used to
    access its value:
         FLAGS.longname          # parsed flag value
         FLAGS.x                 # parsed flag value (short name)

    Command line arguments are scanned and passed to the registered 'Flag'
    objects through the __call__ method.  Unparsed arguments, including
    argv[0] (e.g. the program name) are returned.
         argv = FLAGS(sys.argv)  # scan command line arguments

    The original registered Flag objects can be retrieved through the use
    """
    doc = flags.doc_to_help(self.test_doc_to_help_flag_values.__doc__)
    # Test the general outline of the converted docs
    lines = doc.splitlines()
    self.assertEqual(17, len(lines))
    empty_lines = [index for index in range(len(lines)) if not lines[index]]
    self.assertEqual([1, 3, 5, 8, 12, 15], empty_lines)
    # test that some starting prefix is kept
    flags_lines = [index for index in range(len(lines))
                   if lines[index].startswith('     FLAGS')]
    self.assertEqual([7, 10, 11], flags_lines)
    # but other, especially common space has been removed
    space_lines = [index for index in range(len(lines))
                   if lines[index] and lines[index][0].isspace()]
    self.assertEqual([7, 10, 11, 14], space_lines)
    # No right space was kept
    rspace_lines = [index for index in range(len(lines))
                    if lines[index] != lines[index].rstrip()]
    self.assertEqual([], rspace_lines)
    # test double spaces are kept
    self.assertEqual(True, lines[2].endswith('application:  flags.FLAGS'))

  def test_text_wrap_raises_on_excessive_indent(self):
    """Ensure an indent longer than line length raises."""
    self.assertRaises(ValueError,
                      flags.text_wrap, 'dummy', length=10, indent=' ' * 10)

  def test_text_wrap_raises_on_excessive_first_line(self):
    """Ensure a first line indent longer than line length raises."""
    self.assertRaises(
        ValueError,
        flags.text_wrap, 'dummy', length=80, firstline_indent=' ' * 80)


if __name__ == '__main__':
  absltest.main()
