# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from scripts.docs import clr_converter


def _read_data_file(basename):
  path = os.path.join(
      os.getenv("TEST_SRCDIR"),
      os.getenv("TEST_WORKSPACE"),
      "scripts/docs/testdata",
      basename)
  with open(path, "rt", encoding="utf-8") as f:
    return f.read()


class ConvertTest(unittest.TestCase):

  def test_full_conversion(self):
    html_content = _read_data_file("clr_input.html")
    expected = _read_data_file("clr_expected.mdx")
    actual = clr_converter.convert(html_content)
    self.assertEqual(actual, expected)

  def test_boolean_option(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--batch"><code><a href="#opts-flag--batch">--[no]batch</a></code> default: "false"</dt>
    <dd><p>Run in batch mode.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn('path="--[no]batch"', result)
    self.assertIn('type="boolean"', result)
    self.assertIn('default="false"', result)

  def test_typed_option(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--jobs"><code><a href="#opts-flag--jobs">--jobs</a>=&lt;an integer&gt;</code> default: "auto"</dt>
    <dd><p>Number of jobs.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn('path="--jobs"', result)
    self.assertIn('type="an integer"', result)
    self.assertIn('default="auto"', result)

  def test_void_expansion_option(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--debug"><code><a href="#opts-flag--debug">--debug</a></code></dt>
    <dd>
    <p>Enable debug mode.</p>
    <p>Expands to:
    <br/>&nbsp;&nbsp;<code><a href="#flag--verbose">--verbose</a></code>
    </p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn('type="void"', result)
    self.assertIn("Expands to:", result)
    self.assertIn("`--verbose`", result)
    self.assertNotIn("default=", result)

  def test_allow_multiple_option(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--rc"><code><a href="#opts-flag--rc">--rc</a>=&lt;path&gt;</code> multiple uses are accumulated</dt>
    <dd><p>RC file location.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn("May be used multiple times", result)
    self.assertNotIn("default=", result)

  def test_deprecated_option(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--old"><code><a href="#opts-flag--old">--[no]old</a></code> default: "false"</dt>
    <dd><p>Old flag.</p>
    <p>Tags:
    <a href="#metadata_tag_DEPRECATED"><code>deprecated</code></a>
    </p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn("deprecated", result)

  def test_abbreviation(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--jobs"><code><a href="#opts-flag--jobs">--jobs</a>=&lt;an integer&gt;</code> [<code>-j</code>] default: "8"</dt>
    <dd><p>Number of jobs.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn("Short form: `-j`", result)

  def test_inherits_from(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2><a name="build">Build Options</a></h2>
    <p>Inherits all options from <a href="#common">common</a> and <a href="#test">test</a>.</p>
    <dl>Cat:
    <dt id="build-flag--f"><code><a href="#build-flag--f">--[no]f</a></code> default: "true"</dt>
    <dd><p>A flag.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn("Inherits all options from [common](#common) and [test](#test)", result)
    self.assertIn("{#build}", result)

  def test_enum_type_escaping(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--level"><code><a href="#opts-flag--level">--level</a>={0,1,2}</code> default: "0"</dt>
    <dd><p>Set level.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn("&lcub;0,1,2&rcub;", result)

  def test_quotes_in_type_description(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--jobs"><code><a href="#opts-flag--jobs">--jobs</a>=&lt;an integer, or a keyword ("auto", "HOST_CPUS")&gt;</code> default: "auto"</dt>
    <dd><p>Number of jobs.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertNotIn('("auto"', result)
    self.assertIn("&quot;auto&quot;", result)

  def test_angle_brackets_in_type_description(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h2>Options</h2>
    <dl>Category:
    <dt id="opts-flag--mem"><code><a href="#opts-flag--mem">--mem</a>=&lt;an integer or "HOST_RAM", followed by [-|*]&lt;float&gt;&gt;</code> default: "0"</dt>
    <dd><p>Memory limit.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    for line in result.split("\n"):
      if "<ParamField" in line:
        self.assertNotIn('*]<float', line)
        self.assertIn("&lt;float&gt;", line)
        break

  def test_commands_table(self):
    html_input = """
    <h2>Commands</h2>
    <table>
    <tr>
      <td><a href="#build"><code>build</code></a></td>
      <td>Build targets.</td>
    </tr>
    </table>
    <h2>Options</h2>
    <dl>Cat:
    <dt id="f"><code><a href="#f">--[no]f</a></code> default: "true"</dt>
    <dd><p>Flag.</p></dd>
    </dl>
    """
    result = clr_converter.convert(html_input)
    self.assertIn("## Commands", result)
    self.assertIn("[`build`](#build)", result)
    self.assertIn("Build targets.", result)

  def test_tag_tables(self):
    html_input = """
    <h2>Commands</h2>
    <table></table>
    <h3>Option Effect Tags</h3>
    <table>
    <tr>
    <td id="effect_tag_FOO"><code>foo</code></td>
    <td>Foo description.</td>
    </tr>
    </table>
    """
    result = clr_converter.convert(html_input)
    self.assertIn("### Option Effect Tags", result)
    self.assertIn('<span id="effect_tag_FOO">`foo`</span>', result)
    self.assertIn("Foo description.", result)


if __name__ == "__main__":
  unittest.main()
