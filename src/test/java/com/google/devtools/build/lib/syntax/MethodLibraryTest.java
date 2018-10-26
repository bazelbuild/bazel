// Copyright 2006 The Bazel Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for MethodLibrary.
 */
@RunWith(JUnit4.class)
public class MethodLibraryTest extends EvaluationTestCase {

  private static final String LINE_SEPARATOR = System.lineSeparator();

  @Before
  public final void setFailFast() throws Exception {
    setFailFast(true);
  }

  @Test
  public void testStackTraceLocation() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "Traceback (most recent call last):"
                + LINE_SEPARATOR
                + "\tFile \"\", line 8"
                + LINE_SEPARATOR
                + "\t\tfoo()"
                + LINE_SEPARATOR
                + "\tFile \"\", line 2, in foo"
                + LINE_SEPARATOR
                + "\t\tbar(1)"
                + LINE_SEPARATOR
                + "\tFile \"\", line 7, in bar"
                + LINE_SEPARATOR
                + "\t\t\"test\".index(x)",
            "def foo():",
            "  bar(1)",
            "def bar(x):",
            "  if x == 1:",
            "    a = x",
            "    b = 2",
            "    'test'.index(x)",
            "foo()");
  }

  @Test
  public void testStackTraceWithIf() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "File \"\", line 5"
                + LINE_SEPARATOR
                + "\t\tfoo()"
                + LINE_SEPARATOR
                + "\tFile \"\", line 3, in foo"
                + LINE_SEPARATOR
                + "\t\ts[0]",
            "def foo():",
            "  s = depset()",
            "  if s[0] == 1:",
            "    x = 1",
            "foo()");
  }

  @Test
  public void testStackTraceWithAugmentedAssignment() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "File \"\", line 4"
                + LINE_SEPARATOR
                + "\t\tfoo()"
                + LINE_SEPARATOR
                + "\tFile \"\", line 3, in foo"
                + LINE_SEPARATOR
                + "\t\ts += \"2\""
                + LINE_SEPARATOR
                + "unsupported operand type(s) for +: 'int' and 'string'",
            "def foo():",
            "  s = 1",
            "  s += '2'",
            "foo()");
  }

  @Test
  public void testStackTraceSkipBuiltInOnly() throws Exception {
    // The error message should not include the stack trace when there is
    // only one built-in function.
    new BothModesTest()
        .testIfExactError(
            "expected value of type 'string' for parameter 'sub', "
                + "in method call index(int) of 'string'",
            "'test'.index(1)");
  }

  @Test
  public void testStackTrace() throws Exception {
    // Unlike SkylarintegrationTests#testStackTraceErrorInFunction(), this test
    // has neither a BUILD nor a bzl file.
    new SkylarkTest()
        .testIfExactError(
            "Traceback (most recent call last):"
                + LINE_SEPARATOR
                + "\tFile \"\", line 6"
                + LINE_SEPARATOR
                + "\t\tfoo()"
                + LINE_SEPARATOR
                + "\tFile \"\", line 2, in foo"
                + LINE_SEPARATOR
                + "\t\tbar(1)"
                + LINE_SEPARATOR
                + "\tFile \"\", line 5, in bar"
                + LINE_SEPARATOR
                + "\t\t\"test\".index(x)"
                + LINE_SEPARATOR
                + "expected value of type 'string' for parameter 'sub', "
                + "in method call index(int) of 'string'",
            "def foo():",
            "  bar(1)",
            "def bar(x):",
            "  if 1 == 1:",
            "    'test'.index(x)",
            "foo()");
  }

  @Test
  public void testBuiltinFunctionErrorMessage() throws Exception {
    new BothModesTest()
        .testIfErrorContains("substring \"z\" not found in \"abc\"", "'abc'.index('z')")
        .testIfErrorContains(
            "expected value of type 'string or tuple of strings' for parameter 'sub', "
                + "in method call startswith(int) of 'string'",
            "'test'.startswith(1)")
        .testIfErrorContains(
            "expected value of type 'list(object)' for parameter args in dict(), "
                + "but got \"a\" (string)",
            "dict('a')");
  }

  @Test
  public void testHasAttr() throws Exception {
    new SkylarkTest()
        .testStatement("hasattr(depset(), 'union')", Boolean.TRUE)
        .testStatement("hasattr('test', 'count')", Boolean.TRUE)
        .testStatement("hasattr(dict(a = 1, b = 2), 'items')", Boolean.TRUE)
        .testStatement("hasattr({}, 'items')", Boolean.TRUE);
  }

  @Test
  public void testGetAttrMissingField() throws Exception {
    new SkylarkTest()
        .testIfExactError(
            "object of type 'string' has no attribute 'not_there'",
            "getattr('a string', 'not_there')")
        .testStatement("getattr('a string', 'not_there', 'use this')", "use this")
        .testStatement("getattr('a string', 'not there', None)", Runtime.NONE);
  }

  @SkylarkModule(name = "AStruct", documented = false, doc = "")
  static final class AStruct implements ClassObject {
    @Override
    public Object getValue(String name) {
      switch (name) {
        case "field":
          return "a";
        default:
          return null;
      }
    }

    @Override
    public ImmutableCollection<String> getFieldNames() {
      return ImmutableList.of("field");
    }

    @Override
    public String getErrorMessageForUnknownField(String name) {
      return null;
    }
  }

  @Test
  public void testGetAttrMissingField_typoDetection() throws Exception {
    new SkylarkTest()
        .update("s", new AStruct())
        .testIfExactError(
            "object of type 'AStruct' has no attribute 'feild' (did you mean 'field'?)",
            "getattr(s, 'feild')");
  }

  @Test
  public void testGetAttrWithMethods() throws Exception {
    String msg =
        "object of type 'string' has no attribute 'count', however, "
            + "a method of that name exists";
    new SkylarkTest()
        .testIfExactError(msg, "getattr('a string', 'count')")
        .testStatement("getattr('a string', 'count', 'default')", "default");
  }

  @Test
  public void testDir() throws Exception {
    new SkylarkTest()
        .testStatement(
            "str(dir({}))",
            "[\"clear\", \"get\", \"items\", \"keys\","
                + " \"pop\", \"popitem\", \"setdefault\", \"update\", \"values\"]");
  }

  @Test
  public void testBoolean() throws Exception {
    new BothModesTest().testStatement("False", Boolean.FALSE).testStatement("True", Boolean.TRUE);
  }

  @Test
  public void testBooleanUnsupportedOperationFails() throws Exception {
    new BothModesTest()
        .testIfErrorContains("unsupported operand type(s) for +: 'bool' and 'bool'", "True + True");
  }

  @Test
  public void testListSort() throws Exception {
    new BothModesTest()
        .testEval("sorted([0,1,2,3])", "[0, 1, 2, 3]")
        .testEval("sorted([])", "[]")
        .testEval("sorted([3, 2, 1, 0])", "[0, 1, 2, 3]")
        .testEval("sorted([[1], [], [2], [1, 2]])", "[[], [1], [1, 2], [2]]")
        .testEval("sorted([True, False, True])", "[False, True, True]")
        .testEval("sorted(['a','x','b','z'])", "[\"a\", \"b\", \"x\", \"z\"]")
        .testEval("sorted({1: True, 5: True, 4: False})", "[1, 4, 5]")
        .testEval("sorted(depset([1, 5, 4]))", "[1, 4, 5]")
        .testIfExactError("Cannot compare function with function", "sorted([sorted, sorted])");
  }

  @Test
  public void testDictionaryCopy() throws Exception {
    new BothModesTest()
        .setUp("x = {1 : 2}", "y = dict(x)")
        .testEval("x[1] == 2 and y[1] == 2", "True");
  }

  @Test
  public void testDictionaryCopyKeyCollision() throws Exception {
    new BothModesTest()
        .setUp("x = {'test' : 2}", "y = dict(x, test = 3)")
        .testEval("y['test']", "3");
  }

  @Test
  public void testDictionaryKeyNotFound() throws Exception {
    new BothModesTest()
        .testIfErrorContains("key \"0\" not found in dictionary", "{}['0']")
        .testIfErrorContains("key 0 not found in dictionary", "{'0': 1, 2: 3, 4: 5}[0]");
  }

  @Test
  public void testDictionaryAccess() throws Exception {
    new BothModesTest()
        .testEval("{1: ['foo']}[1]", "['foo']")
        .testStatement("{'4': 8}['4']", 8)
        .testStatement("{'a': 'aa', 'b': 'bb', 'c': 'cc'}['b']", "bb");
  }

  @Test
  public void testDictionaryVariableAccess() throws Exception {
    new BothModesTest().setUp("d = {'a' : 1}", "a = d['a']\n").testLookup("a", 1);
  }

  @Test
  public void testDictionaryCreation() throws Exception {
    String expected = "{'a': 1, 'b': 2, 'c': 3}";

    new BothModesTest()
        .testEval("dict([('a', 1), ('b', 2), ('c', 3)])", expected)
        .testEval("dict(a = 1, b = 2, c = 3)", expected)
        .testEval("dict([('a', 1)], b = 2, c = 3)", expected);
  }

  @Test
  public void testDictionaryCreationInnerLists() throws Exception {
    new BothModesTest().testEval("dict([[1, 2], [3, 4]], a = 5)", "{1: 2, 3: 4, 'a': 5}");
  }

  @Test
  public void testDictionaryCreationEmpty() throws Exception {
    new BothModesTest().testEval("dict()", "{}").testEval("dict([])", "{}");
  }

  @Test
  public void testDictionaryCreationDifferentKeyTypes() throws Exception {
    String expected = "{'a': 1, 2: 3}";

    new BothModesTest()
        .testEval("dict([('a', 1), (2, 3)])", expected)
        .testEval("dict([(2, 3)], a = 1)", expected);
  }

  @Test
  public void testDictionaryCreationKeyCollision() throws Exception {
    String expected = "{'a': 1, 'b': 2, 'c': 3}";

    new BothModesTest()
        .testEval("dict([('a', 42), ('b', 2), ('a', 1), ('c', 3)])", expected)
        .testEval("dict([('a', 42)], a = 1, b = 2, c = 3)", expected);
    new SkylarkTest().testEval("dict([('a', 42)], **{'a': 1, 'b': 2, 'c': 3})", expected);
  }

  @Test
  public void testDictionaryCreationInvalidPositional() throws Exception {
    new BothModesTest()
        .testIfErrorContains(
            "expected value of type 'list(object)' for parameter args in dict(), "
                + "but got \"a\" (string)",
            "dict('a')")
        .testIfErrorContains("cannot convert item #0 to a sequence", "dict(['a'])")
        .testIfErrorContains("cannot convert item #0 to a sequence", "dict([('a')])")
        .testIfErrorContains("too many (3) positional arguments", "dict((3,4), (3,2), (1,2))")
        .testIfErrorContains(
            "item #0 has length 3, but exactly two elements are required",
            "dict([('a', 'b', 'c')])");
  }

  @Test
  public void testDictionaryValues() throws Exception {
    new BothModesTest()
        .testEval("{1: 'foo'}.values()", "['foo']")
        .testEval("{}.values()", "[]")
        .testEval("{True: 3, False: 5}.values()", "[3, 5]")
        .testEval("{'a': 5, 'c': 2, 'b': 4, 'd': 3}.values()", "[5, 2, 4, 3]");
    // sorted by keys
  }

  @Test
  public void testDictionaryKeys() throws Exception {
    new BothModesTest()
        .testEval("{1: 'foo'}.keys()", "[1]")
        .testEval("{}.keys()", "[]")
        .testEval("{True: 3, False: 5}.keys()", "[True, False]")
        .testEval(
            "{1:'a', 2:'b', 6:'c', 0:'d', 5:'e', 4:'f', 3:'g'}.keys()", "[1, 2, 6, 0, 5, 4, 3]");
  }

  @Test
  public void testDictionaryGet() throws Exception {
    new BuildTest()
        .testStatement("{1: 'foo'}.get(1)", "foo")
        .testStatement("{1: 'foo'}.get(2)", Runtime.NONE)
        .testStatement("{1: 'foo'}.get(2, 'a')", "a")
        .testStatement("{1: 'foo'}.get(2, default='a')", "a")
        .testStatement("{1: 'foo'}.get(2, default=None)", Runtime.NONE);
  }

  @Test
  public void testDictionaryItems() throws Exception {
    new BothModesTest()
        .testEval("{'a': 'foo'}.items()", "[('a', 'foo')]")
        .testEval("{}.items()", "[]")
        .testEval("{1: 3, 2: 5}.items()", "[(1, 3), (2, 5)]")
        .testEval("{'a': 5, 'c': 2, 'b': 4}.items()", "[('a', 5), ('c', 2), ('b', 4)]");
  }

  @Test
  public void testDictionaryClear() throws Exception {
    new SkylarkTest()
        .testEval(
            "d = {1: 'foo', 2: 'bar', 3: 'baz'}\n"
                + "len(d) == 3 or fail('clear 1')\n"
                + "d.clear() == None or fail('clear 2')\n"
                + "d",
            "{}");
  }

  @Test
  public void testDictionaryPop() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "KeyError: 1",
            "d = {1: 'foo', 2: 'bar', 3: 'baz'}\n"
                + "len(d) == 3 or fail('pop 1')\n"
                + "d.pop(2) == 'bar' or fail('pop 2')\n"
                + "d.pop(3, 'quux') == 'baz' or fail('pop 3a')\n"
                + "d.pop(3, 'quux') == 'quux' or fail('pop 3b')\n"
                + "d.pop(1) == 'foo' or fail('pop 1')\n"
                + "d == {} or fail('pop 0')\n"
                + "d.pop(1)");
  }

  @Test
  public void testDictionaryPopItem() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "popitem(): dictionary is empty",
            "d = {2: 'bar', 3: 'baz', 1: 'foo'}\n"
                + "len(d) == 3 or fail('popitem 0')\n"
                + "d.popitem() == (2, 'bar') or fail('popitem 2')\n"
                + "d.popitem() == (3, 'baz') or fail('popitem 3')\n"
                + "d.popitem() == (1, 'foo') or fail('popitem 1')\n"
                + "d == {} or fail('popitem 4')\n"
                + "d.popitem()");
  }

  @Test
  public void testDictionaryUpdate() throws Exception {
    new BothModesTest()
        .setUp("foo = {'a': 2}")
        .testEval("foo.update({'b': 4}); foo", "{'a': 2, 'b': 4}");
    new BothModesTest()
        .setUp("foo = {'a': 2}")
        .testEval("foo.update({'a': 3, 'b': 4}); foo", "{'a': 3, 'b': 4}");
  }

  @Test
  public void testDictionarySetDefault() throws Exception {
    new SkylarkTest()
        .testEval(
            "d = {2: 'bar', 1: 'foo'}\n"
                + "len(d) == 2 or fail('setdefault 0')\n"
                + "d.setdefault(1, 'a') == 'foo' or fail('setdefault 1')\n"
                + "d.setdefault(2) == 'bar' or fail('setdefault 2')\n"
                + "d.setdefault(3) == None or fail('setdefault 3')\n"
                + "d.setdefault(4, 'b') == 'b' or fail('setdefault 4')\n"
                + "d",
            "{1: 'foo', 2: 'bar', 3: None, 4: 'b'}");
  }

  @Test
  public void testListIndexMethod() throws Exception {
    new BothModesTest()
        .testStatement("['a', 'b', 'c'].index('a')", 0)
        .testStatement("['a', 'b', 'c'].index('b')", 1)
        .testStatement("['a', 'b', 'c'].index('c')", 2)
        .testStatement("[2, 4, 6].index(4)", 1)
        .testStatement("[2, 4, 6].index(4)", 1)
        .testStatement("[0, 1, [1]].index([1])", 2)
        .testIfErrorContains("item \"a\" not found in list", "[1, 2].index('a')")
        .testIfErrorContains("item 0 not found in list", "[].index(0)");
  }

  @Test
  public void testHash() throws Exception {
    // We specify the same string hashing algorithm as String.hashCode().
    new SkylarkTest()
        .testStatement("hash('skylark')", "skylark".hashCode())
        .testStatement("hash('google')", "google".hashCode())
        .testIfErrorContains(
            "argument 'value' has type 'NoneType', but should be 'string'\n"
                + "in call to builtin function hash(value)",
            "hash(None)");
  }

  @Test
  public void testRange() throws Exception {
    new BothModesTest()
        .testStatement("str(range(5))", "[0, 1, 2, 3, 4]")
        .testStatement("str(range(0))", "[]")
        .testStatement("str(range(1))", "[0]")
        .testStatement("str(range(-2))", "[]")
        .testStatement("str(range(-3, 2))", "[-3, -2, -1, 0, 1]")
        .testStatement("str(range(3, 2))", "[]")
        .testStatement("str(range(3, 3))", "[]")
        .testStatement("str(range(3, 4))", "[3]")
        .testStatement("str(range(3, 5))", "[3, 4]")
        .testStatement("str(range(-3, 5, 2))", "[-3, -1, 1, 3]")
        .testStatement("str(range(-3, 6, 2))", "[-3, -1, 1, 3, 5]")
        .testStatement("str(range(5, 0, -1))", "[5, 4, 3, 2, 1]")
        .testStatement("str(range(5, 0, -10))", "[5]")
        .testStatement("str(range(0, -3, -2))", "[0, -2]")
        .testStatement("str(range(5)[1:])", "[1, 2, 3, 4]")
        .testStatement("len(range(5)[1:])", 4)
        .testStatement("str(range(5)[:2])", "[0, 1]")
        .testStatement("str(range(10)[1:9:2])", "[1, 3, 5, 7]")
        .testStatement("str(range(10)[1:10:2])", "[1, 3, 5, 7, 9]")
        .testStatement("str(range(10)[1:11:2])", "[1, 3, 5, 7, 9]")
        .testStatement("str(range(0, 10, 2)[::2])", "[0, 4, 8]")
        .testStatement("str(range(0, 10, 2)[::-2])", "[8, 4, 0]")
        .testIfErrorContains("step cannot be 0", "range(2, 3, 0)");
  }

  @Test
  public void testRangeIsList() throws Exception {
    // range(), and slices of ranges, may change in the future to return read-only views. But for
    // now it's just a list and can therefore be ordered, mutated, and concatenated. This test
    // ensures we don't break backward compatibility until we intend to, even if range() is
    // optimized to return lazy list-like values.
    runRangeIsListAssertions("range(3)");
    runRangeIsListAssertions("range(4)[:3]");
  }

  @Test
  public void testRangeType() throws Exception {
    new BothModesTest("--incompatible_range_type=true")
        .setUp("a = range(3)")
        .testStatement("len(a)", 3)
        .testStatement("str(a)", "range(0, 3)")
        .testStatement("str(range(1,2,3))", "range(1, 2, 3)")
        .testStatement("repr(a)", "range(0, 3)")
        .testStatement("repr(range(1,2,3))", "range(1, 2, 3)")
        .testStatement("type(a)", "range")
        .testIfErrorContains("unsupported operand type(s) for +: 'range' and 'range'", "a + a")
        .testIfErrorContains("type 'range' has no method append(int)", "a.append(3)")
        .testStatement("str(list(range(5)))", "[0, 1, 2, 3, 4]")
        .testStatement("str(list(range(0)))", "[]")
        .testStatement("str(list(range(1)))", "[0]")
        .testStatement("str(list(range(-2)))", "[]")
        .testStatement("str(list(range(-3, 2)))", "[-3, -2, -1, 0, 1]")
        .testStatement("str(list(range(3, 2)))", "[]")
        .testStatement("str(list(range(3, 3)))", "[]")
        .testStatement("str(list(range(3, 4)))", "[3]")
        .testStatement("str(list(range(3, 5)))", "[3, 4]")
        .testStatement("str(list(range(-3, 5, 2)))", "[-3, -1, 1, 3]")
        .testStatement("str(list(range(-3, 6, 2)))", "[-3, -1, 1, 3, 5]")
        .testStatement("str(list(range(5, 0, -1)))", "[5, 4, 3, 2, 1]")
        .testStatement("str(list(range(5, 0, -10)))", "[5]")
        .testStatement("str(list(range(0, -3, -2)))", "[0, -2]")
        .testStatement("range(3)[-1]", 2)
        .testIfErrorContains(
            "index out of range (index is 3, but sequence has 3 elements)", "range(3)[3]")
        .testStatement("str(range(5)[1:])", "range(1, 5)")
        .testStatement("len(range(5)[1:])", 4)
        .testStatement("str(range(5)[:2])", "range(0, 2)")
        .testStatement("str(range(10)[1:9:2])", "range(1, 9, 2)")
        .testStatement("str(list(range(10)[1:9:2]))", "[1, 3, 5, 7]")
        .testStatement("str(range(10)[1:10:2])", "range(1, 10, 2)")
        .testStatement("str(range(10)[1:11:2])", "range(1, 10, 2)")
        .testStatement("str(range(0, 10, 2)[::2])", "range(0, 10, 4)")
        .testStatement("str(range(0, 10, 2)[::-2])", "range(8, -2, -4)")
        .testStatement("str(range(5)[1::-1])", "range(1, -1, -1)")
        .testIfErrorContains("step cannot be 0", "range(2, 3, 0)")
        .testIfErrorContains("unsupported operand type(s) for *: 'range' and 'int'", "range(3) * 3")
        .testIfErrorContains("Cannot compare range objects", "range(3) < range(5)")
        .testIfErrorContains("Cannot compare range objects", "range(4) > [1]")
        .testStatement("4 in range(1, 10)", true)
        .testStatement("4 in range(1, 3)", false)
        .testStatement("4 in range(0, 8, 2)", true)
        .testStatement("4 in range(1, 8, 2)", false)
        .testStatement("range(0, 5, 10) == range(0, 5, 11)", true)
        .testStatement("range(0, 5, 2) == [0, 2, 4]", false);
  }

  /**
   * Helper function for testRangeIsList that expects a range or range slice expression producing
   * the range value containing [0, 1, 2].
   */
  private void runRangeIsListAssertions(String range3expr) throws Exception {
    // Check stringifications.
    new BothModesTest()
        .setUp("a = " + range3expr)
        .testStatement("str(a)", "[0, 1, 2]")
        .testStatement("repr(a)", "[0, 1, 2]")
        .testStatement("type(a)", "list");

    // Check comparisons.
    new BothModesTest()
        .setUp("a = " + range3expr)
        .setUp("b = range(0, 3, 1)")
        .setUp("c = range(1, 4)")
        .setUp("L = [0, 1, 2]")
        .setUp("T = (0, 1, 2)")
        .testStatement("a == b", true)
        .testStatement("a == c", false)
        .testStatement("a < b", false)
        .testStatement("a < c", true)
        .testStatement("a == L", true)
        .testStatement("a == T", false)
        .testStatement("a < L", false)
        .testIfErrorContains("Cannot compare list with tuple", "a < T");

    // Check mutations.
    new BothModesTest()
        .setUp("a = " + range3expr)
        .testStatement("a.append(3); str(a)", "[0, 1, 2, 3]");
    new SkylarkTest()
        .testStatement(
            "def f():\n"
            + "  a = " + range3expr + "\n"
            + "  b = a\n"
            + "  a += [3]\n"
            + "  return str(b)\n"
            + "f()\n",
            "[0, 1, 2, 3]");

    // Check concatenations.
    new BothModesTest()
        .setUp("a = " + range3expr)
        .setUp("b = range(3, 4)")
        .setUp("L = [3]")
        .setUp("T = (3,)")
        .testStatement("str(a + b)", "[0, 1, 2, 3]")
        .testStatement("str(a + L)", "[0, 1, 2, 3]")
        .testIfErrorContains("unsupported operand type(s) for +: 'list' and 'tuple", "str(a + T)");
  }

  @Test
  public void testEnumerate() throws Exception {
    new BothModesTest()
        .testStatement("str(enumerate([]))", "[]")
        .testStatement("str(enumerate([5]))", "[(0, 5)]")
        .testStatement("str(enumerate([5, 3]))", "[(0, 5), (1, 3)]")
        .testStatement("str(enumerate(['a', 'b', 'c']))", "[(0, \"a\"), (1, \"b\"), (2, \"c\")]")
        .testStatement("str(enumerate(['a']) + [(1, 'b')])", "[(0, \"a\"), (1, \"b\")]");
  }

  @Test
  public void testEnumerateBadArg() throws Exception {
    new BothModesTest()
        .testIfErrorContains(
            "argument 'list' has type 'string', but should be 'sequence'\n"
                + "in call to builtin function enumerate(list)",
            "enumerate('a')");
  }

  @Test
  public void testReassignmentOfPrimitivesNotForbiddenByCoreLanguage() throws Exception {
    new BuildTest()
        .setUp("cc_binary = (['hello.cc'])")
        .testIfErrorContains(
            "'list' object is not callable",
            "cc_binary(name = 'hello', srcs=['hello.cc'], malloc = '//base:system_malloc')");
  }

  @Test
  public void testLenOnString() throws Exception {
    new BothModesTest().testStatement("len('abc')", 3);
  }

  @Test
  public void testLenOnList() throws Exception {
    new BothModesTest().testStatement("len([1,2,3])", 3);
  }

  @Test
  public void testLenOnDict() throws Exception {
    new BothModesTest().testStatement("len({'a' : 1, 'b' : 2})", 2);
  }

  @Test
  public void testLenOnBadType() throws Exception {
    new BothModesTest().testIfErrorContains("int is not iterable", "len(1)");
  }

  @Test
  public void testIndexOnFunction() throws Exception {
    new BothModesTest()
        .testIfErrorContains("type 'function' has no operator [](int)", "len[1]")
        .testIfErrorContains("type 'function' has no operator [:](int, int, NoneType)", "len[1:4]");
  }

  @Test
  public void testBool() throws Exception {
    new BothModesTest()
        .testStatement("bool(1)", Boolean.TRUE)
        .testStatement("bool(0)", Boolean.FALSE)
        .testStatement("bool([1, 2])", Boolean.TRUE)
        .testStatement("bool([])", Boolean.FALSE)
        .testStatement("bool(None)", Boolean.FALSE);
  }

  @Test
  public void testStr() throws Exception {
    new BothModesTest()
        .testStatement("str(1)", "1")
        .testStatement("str(-2)", "-2")
        .testStatement("str([1, 2])", "[1, 2]")
        .testStatement("str(True)", "True")
        .testStatement("str(False)", "False")
        .testStatement("str(None)", "None")
        .testStatement("str(str)", "<built-in function str>");
  }

  @Test
  public void testStrFunction() throws Exception {
    new SkylarkTest().testStatement("def foo(x): return x\nstr(foo)", "<function foo>");
  }

  @Test
  public void testType() throws Exception {
    new SkylarkTest()
        .testStatement("type(1)", "int")
        .testStatement("type('a')", "string")
        .testStatement("type([1, 2])", "list")
        .testStatement("type((1, 2))", "tuple")
        .testStatement("type(True)", "bool")
        .testStatement("type(None)", "NoneType")
        .testStatement("type(str)", "function");
  }

  // TODO(bazel-team): Move this into a new BazelLibraryTest.java file, or at least out of
  // MethodLibraryTest.java.
  @Test
  public void testSelectFunction() throws Exception {
    enableSkylarkMode();
    eval("a = select({'a': 1})");
    SelectorList result = (SelectorList) lookup("a");
    assertThat(((SelectorValue) Iterables.getOnlyElement(result.getElements())).getDictionary())
        .containsExactly("a", 1);
  }

  @Test
  public void testZipFunction() throws Exception {
    new BothModesTest()
        .testStatement("str(zip())", "[]")
        .testStatement("str(zip([1, 2]))", "[(1,), (2,)]")
        .testStatement("str(zip([1, 2], ['a', 'b']))", "[(1, \"a\"), (2, \"b\")]")
        .testStatement("str(zip([1, 2, 3], ['a', 'b']))", "[(1, \"a\"), (2, \"b\")]")
        .testStatement("str(zip([1], [2], [3]))", "[(1, 2, 3)]")
        .testStatement("str(zip([1], {2: 'a'}))", "[(1, 2)]")
        .testStatement("str(zip([1], []))", "[]")
        .testIfErrorContains("type 'int' is not iterable", "zip(123)")
        .testIfErrorContains("type 'int' is not iterable", "zip([1], 1)")
        .testStatement("str(zip([1], depset([2])))", "[(1, 2)]");
  }

  /**
   * Assert that lstrip(), rstrip(), and strip() produce the expected result for a given input
   * string and chars argument. If chars is null no argument is passed.
   */
  private void checkStrip(
      String input, Object chars,
      String expLeft, String expRight, String expBoth) throws Exception {
    if (chars == null) {
      new BothModesTest()
          .update("s", input)
          .testStatement("s.lstrip()", expLeft)
          .testStatement("s.rstrip()", expRight)
          .testStatement("s.strip()", expBoth);
    } else {
      new BothModesTest()
          .update("s", input)
          .update("chars", chars)
          .testStatement("s.lstrip(chars)", expLeft)
          .testStatement("s.rstrip(chars)", expRight)
          .testStatement("s.strip(chars)", expBoth);
    }
  }

  @Test
  public void testStrip() throws Exception {
    // Strip nothing.
    checkStrip("a b c", "", "a b c", "a b c", "a b c");
    checkStrip(" a b c ", "", " a b c ", " a b c ", " a b c ");
    // Normal case, found and not found.
    checkStrip("abcba", "ba", "cba", "abc", "c");
    checkStrip("abc", "xyz", "abc", "abc", "abc");
    // Default whitespace.
    checkStrip(" a b c ", null, "a b c ", " a b c", "a b c");
    checkStrip(" a b c ", Runtime.NONE, "a b c ", " a b c", "a b c");
    // Default whitespace with full range of Latin-1 whitespace chars.
    String whitespace = "\u0009\n\u000B\u000C\r\u001C\u001D\u001E\u001F\u0020\u0085\u00A0";
    checkStrip(
        whitespace + "a" + whitespace, null,
        "a" + whitespace, whitespace + "a", "a");
    checkStrip(
        whitespace + "a" + whitespace, Runtime.NONE,
        "a" + whitespace, whitespace + "a", "a");
    // Empty cases.
    checkStrip("", "", "", "", "");
    checkStrip("abc", "abc", "", "", "");
    checkStrip("", "xyz", "", "", "");
    checkStrip("", null, "", "", "");
  }

  @Test
  public void testFail() throws Exception {
    new SkylarkTest()
        .testIfErrorContains("abc", "fail('abc')")
        .testIfErrorContains("18", "fail(18)");
  }

  @Test
  public void testTupleCoercion() throws Exception {
    new BothModesTest()
        .testStatement("tuple([1, 2]) == (1, 2)", true)
        .testStatement("tuple(depset([1, 2])) == (1, 2)", true)
        // Depends on current implementation of dict
        .testStatement("tuple({1: 'foo', 2: 'bar'}) == (1, 2)", true);
  }
}
