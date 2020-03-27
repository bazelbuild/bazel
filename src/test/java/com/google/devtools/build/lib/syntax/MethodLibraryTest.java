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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for MethodLibrary. */
@RunWith(JUnit4.class)
public final class MethodLibraryTest extends EvaluationTestCase {

  private static final String LINE_SEPARATOR = System.lineSeparator();

  @Before
  public final void setFailFast() throws Exception {
    setFailFast(true);
  }

  @Test
  public void testStackTraceLocation() throws Exception {
    new Scenario()
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
    new Scenario()
        .testIfErrorContains(
            "File \"\", line 5"
                + LINE_SEPARATOR
                + "\t\tfoo()"
                + LINE_SEPARATOR
                + "\tFile \"\", line 3, in foo"
                + LINE_SEPARATOR
                + "\t\ts[0]",
            "def foo():",
            "  s = []",
            "  if s[0] == 1:",
            "    x = 1",
            "foo()");
  }

  @Test
  public void testStackTraceWithAugmentedAssignment() throws Exception {
    new Scenario()
        .testIfErrorContains(
            "File \"\", line 4"
                + LINE_SEPARATOR
                + "\t\tfoo()"
                + LINE_SEPARATOR
                + "\tFile \"\", line 3, in foo"
                + LINE_SEPARATOR
                + "\t\ts += \"2\""
                + LINE_SEPARATOR
                + "unsupported binary operation: int + string",
            "def foo():",
            "  s = 1",
            "  s += '2'",
            "foo()");
  }

  @Test
  public void testStackTraceSkipBuiltInOnly() throws Exception {
    // The error message should not include the stack trace when there is
    // only one built-in function.
    new Scenario()
        .testIfExactError(
            "in call to index(), parameter 'sub' got value of type 'int', want 'string'",
            "'test'.index(1)");
  }

  @Test
  public void testStackTrace() throws Exception {
    // Unlike SkylarintegrationTests#testStackTraceErrorInFunction(), this test
    // has neither a BUILD nor a bzl file.
    new Scenario()
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
                + "in call to index(), parameter 'sub' got value of type 'int', want 'string'",
            "def foo():",
            "  bar(1)",
            "def bar(x):",
            "  if 1 == 1:",
            "    'test'.index(x)",
            "foo()");
  }

  @Test
  public void testBuiltinFunctionErrorMessage() throws Exception {
    new Scenario()
        .testIfErrorContains("substring \"z\" not found in \"abc\"", "'abc'.index('z')")
        .testIfErrorContains(
            "in call to startswith(), parameter 'sub' got value of type 'int', want 'string or"
                + " tuple of strings'",
            "'test'.startswith(1)")
        .testIfErrorContains("in dict, got string, want iterable", "dict('a')");
  }

  @Test
  public void testHasAttr() throws Exception {
    new Scenario()
        .testExpression("hasattr(depset(), 'to_list')", Boolean.TRUE)
        .testExpression("hasattr('test', 'count')", Boolean.TRUE)
        .testExpression("hasattr(dict(a = 1, b = 2), 'items')", Boolean.TRUE)
        .testExpression("hasattr({}, 'items')", Boolean.TRUE);
  }

  @Test
  public void testGetAttrMissingField() throws Exception {
    new Scenario()
        .testIfExactError(
            "'string' value has no field or method 'not_there'", "getattr('a string', 'not_there')")
        .testExpression("getattr('a string', 'not_there', 'use this')", "use this")
        .testExpression("getattr('a string', 'not there', None)", Starlark.NONE);
  }

  @SkylarkModule(name = "AStruct", documented = false, doc = "")
  static final class AStruct implements ClassObject, StarlarkValue {
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
    new Scenario()
        .update("s", new AStruct())
        .testIfExactError(
            "'AStruct' value has no field or method 'feild' (did you mean 'field'?)",
            "getattr(s, 'feild')");
  }

  @Test
  public void testGetAttrWithMethods() throws Exception {
    String msg = "'string' value has no field or method 'cnt'";
    new Scenario()
        .testIfExactError(msg, "getattr('a string', 'cnt')")
        .testExpression("getattr('a string', 'cnt', 'default')", "default");
  }

  @Test
  public void testDir() throws Exception {
    new Scenario()
        .testExpression(
            "str(dir({}))",
            "[\"clear\", \"get\", \"items\", \"keys\","
                + " \"pop\", \"popitem\", \"setdefault\", \"update\", \"values\"]");
  }

  @Test
  public void testBoolean() throws Exception {
    new Scenario().testExpression("False", Boolean.FALSE).testExpression("True", Boolean.TRUE);
  }

  @Test
  public void testBooleanUnsupportedOperationFails() throws Exception {
    new Scenario().testIfErrorContains("unsupported binary operation: bool + bool", "True + True");
  }

  @Test
  public void testListSort() throws Exception {
    new Scenario()
        .testEval("sorted([0,1,2,3])", "[0, 1, 2, 3]")
        .testEval("sorted([])", "[]")
        .testEval("sorted([3, 2, 1, 0])", "[0, 1, 2, 3]")
        .testEval("sorted([[1], [], [2], [1, 2]])", "[[], [1], [1, 2], [2]]")
        .testEval("sorted([True, False, True])", "[False, True, True]")
        .testEval("sorted(['a','x','b','z'])", "[\"a\", \"b\", \"x\", \"z\"]")
        .testEval("sorted({1: True, 5: True, 4: False})", "[1, 4, 5]")
        .testEval("sorted([3, 2, 1, 0], reverse=True)", "[3, 2, 1, 0]")
        .testEval("sorted([[1], [], [1, 2]], key=len, reverse=True)", "[[1, 2], [1], []]")
        .testEval("sorted([[0, 5], [4, 1], [1, 7]], key=max)", "[[4, 1], [0, 5], [1, 7]]")
        .testIfExactError("Cannot compare function with function", "sorted([sorted, sorted])");
  }

  @Test
  public void testDictionaryCopy() throws Exception {
    new Scenario().setUp("x = {1 : 2}", "y = dict(x)").testEval("x[1] == 2 and y[1] == 2", "True");
  }

  @Test
  public void testDictionaryCopyKeyCollision() throws Exception {
    new Scenario().setUp("x = {'test' : 2}", "y = dict(x, test = 3)").testEval("y['test']", "3");
  }

  @Test
  public void testDictionaryKeyNotFound() throws Exception {
    new Scenario()
        .testIfErrorContains("key \"0\" not found in dictionary", "{}['0']")
        .testIfErrorContains("key 0 not found in dictionary", "{'0': 1, 2: 3, 4: 5}[0]");
  }

  @Test
  public void testDictionaryAccess() throws Exception {
    new Scenario()
        .testEval("{1: ['foo']}[1]", "['foo']")
        .testExpression("{'4': 8}['4']", 8)
        .testExpression("{'a': 'aa', 'b': 'bb', 'c': 'cc'}['b']", "bb");
  }

  @Test
  public void testDictionaryVariableAccess() throws Exception {
    new Scenario().setUp("d = {'a' : 1}", "a = d['a']").testLookup("a", 1);
  }

  @Test
  public void testDictionaryCreation() throws Exception {
    String expected = "{'a': 1, 'b': 2, 'c': 3}";

    new Scenario()
        .testEval("dict([('a', 1), ('b', 2), ('c', 3)])", expected)
        .testEval("dict(a = 1, b = 2, c = 3)", expected)
        .testEval("dict([('a', 1)], b = 2, c = 3)", expected);
  }

  @Test
  public void testDictionaryCreationInnerLists() throws Exception {
    new Scenario().testEval("dict([[1, 2], [3, 4]], a = 5)", "{1: 2, 3: 4, 'a': 5}");
  }

  @Test
  public void testDictionaryCreationEmpty() throws Exception {
    new Scenario().testEval("dict()", "{}").testEval("dict([])", "{}");
  }

  @Test
  public void testDictionaryCreationDifferentKeyTypes() throws Exception {
    String expected = "{'a': 1, 2: 3}";

    new Scenario()
        .testEval("dict([('a', 1), (2, 3)])", expected)
        .testEval("dict([(2, 3)], a = 1)", expected);
  }

  @Test
  public void testDictionaryCreationKeyCollision() throws Exception {
    String expected = "{'a': 1, 'b': 2, 'c': 3}";

    new Scenario()
        .testEval("dict([('a', 42), ('b', 2), ('a', 1), ('c', 3)])", expected)
        .testEval("dict([('a', 42)], a = 1, b = 2, c = 3)", expected);
    new Scenario().testEval("dict([('a', 42)], **{'a': 1, 'b': 2, 'c': 3})", expected);
  }

  @Test
  public void testDictionaryCreationInvalidPositional() throws Exception {
    new Scenario()
        .testIfErrorContains("in dict, got string, want iterable", "dict('a')")
        .testIfErrorContains(
            "in dict, dictionary update sequence element #0 is not iterable (string)",
            "dict([('a')])")
        .testIfErrorContains(
            "in dict, dictionary update sequence element #0 is not iterable (string)",
            "dict([('a')])")
        .testIfErrorContains(
            "dict() accepts no more than 1 positional argument but got 3",
            "dict((3,4), (3,2), (1,2))")
        .testIfErrorContains(
            "item #0 has length 3, but exactly two elements are required",
            "dict([('a', 'b', 'c')])");
  }

  @Test
  public void testDictionaryValues() throws Exception {
    new Scenario()
        .testEval("{1: 'foo'}.values()", "['foo']")
        .testEval("{}.values()", "[]")
        .testEval("{True: 3, False: 5}.values()", "[3, 5]")
        .testEval("{'a': 5, 'c': 2, 'b': 4, 'd': 3}.values()", "[5, 2, 4, 3]");
    // sorted by keys
  }

  @Test
  public void testDictionaryKeys() throws Exception {
    new Scenario()
        .testEval("{1: 'foo'}.keys()", "[1]")
        .testEval("{}.keys()", "[]")
        .testEval("{True: 3, False: 5}.keys()", "[True, False]")
        .testEval(
            "{1:'a', 2:'b', 6:'c', 0:'d', 5:'e', 4:'f', 3:'g'}.keys()", "[1, 2, 6, 0, 5, 4, 3]");
  }

  @Test
  public void testDictionaryGet() throws Exception {
    new Scenario()
        .testExpression("{1: 'foo'}.get(1)", "foo")
        .testExpression("{1: 'foo'}.get(2)", Starlark.NONE)
        .testExpression("{1: 'foo'}.get(2, 'a')", "a")
        .testExpression("{1: 'foo'}.get(2, default='a')", "a")
        .testExpression("{1: 'foo'}.get(2, default=None)", Starlark.NONE);
  }

  @Test
  public void testDictionaryItems() throws Exception {
    new Scenario()
        .testEval("{'a': 'foo'}.items()", "[('a', 'foo')]")
        .testEval("{}.items()", "[]")
        .testEval("{1: 3, 2: 5}.items()", "[(1, 3), (2, 5)]")
        .testEval("{'a': 5, 'c': 2, 'b': 4}.items()", "[('a', 5), ('c', 2), ('b', 4)]");
  }

  @Test
  public void testDictionaryClear() throws Exception {
    new Scenario()
        .setUp(
            "d = {1: 'foo', 2: 'bar', 3: 'baz'}",
            "len(d) == 3 or fail('clear 1')",
            "d.clear() == None or fail('clear 2')")
        .testEval("d", "{}");
  }

  @Test
  public void testDictionaryPop() throws Exception {
    new Scenario()
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
    new Scenario()
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
    new Scenario()
        .setUp("foo = {'a': 2}", "foo.update({'b': 4})")
        .testEval("foo", "{'a': 2, 'b': 4}");
    new Scenario()
        .setUp("foo = {'a': 2}", "foo.update({'a': 3, 'b': 4})")
        .testEval("foo", "{'a': 3, 'b': 4}");
  }

  @Test
  public void testDictionarySetDefault() throws Exception {
    new Scenario()
        .setUp(
            "d = {2: 'bar', 1: 'foo'}",
            "len(d) == 2 or fail('setdefault 0')",
            "d.setdefault(1, 'a') == 'foo' or fail('setdefault 1')",
            "d.setdefault(2) == 'bar' or fail('setdefault 2')",
            "d.setdefault(3) == None or fail('setdefault 3')",
            "d.setdefault(4, 'b') == 'b' or fail('setdefault 4')")
        .testEval("d", "{1: 'foo', 2: 'bar', 3: None, 4: 'b'}");
  }

  @Test
  public void testListIndexMethod() throws Exception {
    new Scenario()
        .testExpression("['a', 'b', 'c'].index('a')", 0)
        .testExpression("['a', 'b', 'c'].index('b')", 1)
        .testExpression("['a', 'b', 'c'].index('c')", 2)
        .testExpression("[2, 4, 6].index(4)", 1)
        .testExpression("[2, 4, 6].index(4)", 1)
        .testExpression("[0, 1, [1]].index([1])", 2)
        .testIfErrorContains("item \"a\" not found in list", "[1, 2].index('a')")
        .testIfErrorContains("item 0 not found in list", "[].index(0)");
  }

  @Test
  public void testHash() throws Exception {
    // We specify the same string hashing algorithm as String.hashCode().
    new Scenario()
        .testExpression("hash('skylark')", "skylark".hashCode())
        .testExpression("hash('google')", "google".hashCode())
        .testIfErrorContains(
            "in call to hash(), parameter 'value' got value of type 'NoneType', want 'string'",
            "hash(None)");
  }

  @Test
  public void testRangeType() throws Exception {
    new Scenario()
        .setUp("a = range(3)")
        .testExpression("len(a)", 3)
        .testExpression("str(a)", "range(0, 3)")
        .testExpression("str(range(1,2,3))", "range(1, 2, 3)")
        .testExpression("repr(a)", "range(0, 3)")
        .testExpression("repr(range(1,2,3))", "range(1, 2, 3)")
        .testExpression("type(a)", "range")
        .testIfErrorContains("unsupported binary operation: range + range", "a + a")
        .testIfErrorContains("'range' value has no field or method 'append'", "a.append(3)")
        .testExpression("str(list(range(5)))", "[0, 1, 2, 3, 4]")
        .testExpression("str(list(range(0)))", "[]")
        .testExpression("str(list(range(1)))", "[0]")
        .testExpression("str(list(range(-2)))", "[]")
        .testExpression("str(list(range(-3, 2)))", "[-3, -2, -1, 0, 1]")
        .testExpression("str(list(range(3, 2)))", "[]")
        .testExpression("str(list(range(3, 3)))", "[]")
        .testExpression("str(list(range(3, 4)))", "[3]")
        .testExpression("str(list(range(3, 5)))", "[3, 4]")
        .testExpression("str(list(range(-3, 5, 2)))", "[-3, -1, 1, 3]")
        .testExpression("str(list(range(-3, 6, 2)))", "[-3, -1, 1, 3, 5]")
        .testExpression("str(list(range(5, 0, -1)))", "[5, 4, 3, 2, 1]")
        .testExpression("str(list(range(5, 0, -10)))", "[5]")
        .testExpression("str(list(range(0, -3, -2)))", "[0, -2]")
        .testExpression("range(3)[-1]", 2)
        .testIfErrorContains(
            "index out of range (index is 3, but sequence has 3 elements)", "range(3)[3]")
        .testExpression("str(range(5)[1:])", "range(1, 5)")
        .testExpression("len(range(5)[1:])", 4)
        .testExpression("str(range(5)[:2])", "range(0, 2)")
        .testExpression("str(range(10)[1:9:2])", "range(1, 9, 2)")
        .testExpression("str(list(range(10)[1:9:2]))", "[1, 3, 5, 7]")
        .testExpression("str(range(10)[1:10:2])", "range(1, 10, 2)")
        .testExpression("str(range(10)[1:11:2])", "range(1, 10, 2)")
        .testExpression("str(range(0, 10, 2)[::2])", "range(0, 10, 4)")
        .testExpression("str(range(0, 10, 2)[::-2])", "range(8, -2, -4)")
        .testExpression("str(range(5)[1::-1])", "range(1, -1, -1)")
        .testIfErrorContains("step cannot be 0", "range(2, 3, 0)")
        .testIfErrorContains("unsupported binary operation: range * int", "range(3) * 3")
        .testIfErrorContains("Cannot compare range objects", "range(3) < range(5)")
        .testIfErrorContains("Cannot compare range objects", "range(4) > [1]")
        .testExpression("4 in range(1, 10)", true)
        .testExpression("4 in range(1, 3)", false)
        .testExpression("4 in range(0, 8, 2)", true)
        .testExpression("4 in range(1, 8, 2)", false)
        .testExpression("range(0, 5, 10) == range(0, 5, 11)", true)
        .testExpression("range(0, 5, 2) == [0, 2, 4]", false)
        .testExpression("str(list(range(1, 10, 2)))", "[1, 3, 5, 7, 9]")
        .testExpression("str(range(1, 10, 2)[:99])", "range(1, 11, 2)")
        .testExpression("range(1, 10, 2) == range(1, 11, 2)", true)
        .testExpression("range(1, 10, 2) == range(1, 12, 2)", false)
        // x in range(...), +ve step
        .testExpression("2          in range(3, 0x7ffffffd, 2)", false) // too low
        .testExpression("3          in range(3, 0x7ffffffd, 2)", true) // in range
        .testExpression("4          in range(3, 0x7ffffffd, 2)", false) // even
        .testExpression("5          in range(3, 0x7ffffffd, 2)", true) // in range
        .testExpression("0x7ffffffb in range(3, 0x7ffffffd, 2)", true) // in range
        .testExpression("0x7ffffffc in range(3, 0x7ffffffd, 2)", false) // even
        .testExpression("0x7ffffffd in range(3, 0x7ffffffd, 2)", false) // too high
        // x in range(...), -ve step
        .testExpression("0x7ffffffe in range(0x7ffffffd, 3, -2)", false) // too high
        .testExpression("0x7ffffffd in range(0x7ffffffd, 3, -2)", true) // in range
        .testExpression("0x7ffffffc in range(0x7ffffffd, 3, -2)", false) // even
        .testExpression("0x7ffffffb in range(0x7ffffffd, 3, -2)", true) // in range
        .testExpression("5          in range(0x7ffffffd, 3, -2)", true) // in range
        .testExpression("4          in range(0x7ffffffd, 3, -2)", false) // even
        .testExpression("3          in range(0x7ffffffd, 3, -2)", false); // too low
  }

  @Test
  public void testEnumerate() throws Exception {
    new Scenario()
        .testExpression("str(enumerate([]))", "[]")
        .testExpression("str(enumerate([5]))", "[(0, 5)]")
        .testExpression("str(enumerate([5, 3]))", "[(0, 5), (1, 3)]")
        .testExpression("str(enumerate(['a', 'b', 'c']))", "[(0, \"a\"), (1, \"b\"), (2, \"c\")]")
        .testExpression("str(enumerate(['a']) + [(1, 'b')])", "[(0, \"a\"), (1, \"b\")]");
  }

  @Test
  public void testEnumerateBadArg() throws Exception {
    new Scenario().testIfErrorContains("type 'string' is not iterable", "enumerate('a')");
  }

  @Test
  public void testReassignmentOfPrimitivesNotForbiddenByCoreLanguage() throws Exception {
    new Scenario()
        .setUp("cc_binary = (['hello.cc'])")
        .testIfErrorContains(
            "'list' object is not callable",
            "cc_binary(name = 'hello', srcs=['hello.cc'], malloc = '//base:system_malloc')");
  }

  @Test
  public void testLenOnString() throws Exception {
    new Scenario().testExpression("len('abc')", 3);
  }

  @Test
  public void testLenOnList() throws Exception {
    new Scenario().testExpression("len([1,2,3])", 3);
  }

  @Test
  public void testLenOnDict() throws Exception {
    new Scenario().testExpression("len({'a' : 1, 'b' : 2})", 2);
  }

  @Test
  public void testLenOnBadType() throws Exception {
    new Scenario().testIfErrorContains("int is not iterable", "len(1)");
  }

  @Test
  public void testIndexOnFunction() throws Exception {
    new Scenario()
        .testIfErrorContains("type 'function' has no operator [](int)", "len[1]")
        .testIfErrorContains("invalid slice operand: function", "len[1:4]");
  }

  @Test
  public void testBool() throws Exception {
    new Scenario()
        .testExpression("bool(1)", Boolean.TRUE)
        .testExpression("bool(0)", Boolean.FALSE)
        .testExpression("bool([1, 2])", Boolean.TRUE)
        .testExpression("bool([])", Boolean.FALSE)
        .testExpression("bool(None)", Boolean.FALSE);
  }

  @Test
  public void testStr() throws Exception {
    new Scenario()
        .testExpression("str(1)", "1")
        .testExpression("str(-2)", "-2")
        .testExpression("str([1, 2])", "[1, 2]")
        .testExpression("str(True)", "True")
        .testExpression("str(False)", "False")
        .testExpression("str(None)", "None")
        .testExpression("str(str)", "<built-in function str>");
  }

  @Test
  public void testStrFunction() throws Exception {
    new Scenario().setUp("def foo(x): pass").testExpression("str(foo)", "<function foo>");
  }

  @Test
  public void testType() throws Exception {
    new Scenario()
        .testExpression("type(1)", "int")
        .testExpression("type('a')", "string")
        .testExpression("type([1, 2])", "list")
        .testExpression("type((1, 2))", "tuple")
        .testExpression("type(True)", "bool")
        .testExpression("type(None)", "NoneType")
        .testExpression("type(str)", "function");
  }

  @Test
  public void testZipFunction() throws Exception {
    new Scenario()
        .testExpression("str(zip())", "[]")
        .testExpression("str(zip([1, 2]))", "[(1,), (2,)]")
        .testExpression("str(zip([1, 2], ['a', 'b']))", "[(1, \"a\"), (2, \"b\")]")
        .testExpression("str(zip([1, 2, 3], ['a', 'b']))", "[(1, \"a\"), (2, \"b\")]")
        .testExpression("str(zip([1], [2], [3]))", "[(1, 2, 3)]")
        .testExpression("str(zip([1], {2: 'a'}))", "[(1, 2)]")
        .testExpression("str(zip([1], []))", "[]")
        .testIfErrorContains("type 'int' is not iterable", "zip(123)")
        .testIfErrorContains("type 'int' is not iterable", "zip([1], 1)");
  }

  /**
   * Assert that lstrip(), rstrip(), and strip() produce the expected result for a given input
   * string and chars argument. If chars is null no argument is passed.
   */
  private void checkStrip(
      String input, Object chars,
      String expLeft, String expRight, String expBoth) throws Exception {
    if (chars == null) {
      new Scenario()
          .update("s", input)
          .testExpression("s.lstrip()", expLeft)
          .testExpression("s.rstrip()", expRight)
          .testExpression("s.strip()", expBoth);
    } else {
      new Scenario()
          .update("s", input)
          .update("chars", chars)
          .testExpression("s.lstrip(chars)", expLeft)
          .testExpression("s.rstrip(chars)", expRight)
          .testExpression("s.strip(chars)", expBoth);
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
    checkStrip(" a b c ", Starlark.NONE, "a b c ", " a b c", "a b c");
    // Default whitespace with full range of Latin-1 whitespace chars.
    String whitespace = "\u0009\n\u000B\u000C\r\u001C\u001D\u001E\u001F\u0020\u0085\u00A0";
    checkStrip(
        whitespace + "a" + whitespace, null,
        "a" + whitespace, whitespace + "a", "a");
    checkStrip(
        whitespace + "a" + whitespace, Starlark.NONE, "a" + whitespace, whitespace + "a", "a");
    // Empty cases.
    checkStrip("", "", "", "", "");
    checkStrip("abc", "abc", "", "", "");
    checkStrip("", "xyz", "", "", "");
    checkStrip("", null, "", "", "");
  }

  @Test
  public void testFail() throws Exception {
    new Scenario().testIfErrorContains("abc", "fail('abc')").testIfErrorContains("18", "fail(18)");
  }

  @Test
  public void testTupleCoercion() throws Exception {
    new Scenario()
        .testExpression("tuple([1, 2]) == (1, 2)", true)
        // Depends on current implementation of dict
        .testExpression("tuple({1: 'foo', 2: 'bar'}) == (1, 2)", true);
  }

  @Test
  public void testPositionalOnlyArgument() throws Exception {
    new Scenario()
        .testIfErrorContains(
            "join() got named argument for positional-only parameter 'elements'",
            "','.join(elements=['foo', 'bar'])");
  }

  @Test
  public void testStringJoinRequiresStrings() throws Exception {
    new Scenario()
        .testIfErrorContains(
            "expected string for sequence element 1, got 'int'", "', '.join(['foo', 2])");
  }

  @Test
  public void testDepsetItemsKeywordAndPositional() throws Exception {
    new Scenario("--incompatible_disable_depset_items=false")
        .testIfErrorContains(
            "parameter 'items' cannot be specified both positionally and by keyword",
            "depset([0, 1], 'default', items=[0,1])");
  }

  @Test
  public void testDepsetDirectInvalidType() throws Exception {
    new Scenario()
        .testIfErrorContains(
            "expected type 'sequence' for direct but got type 'string' instead",
            "depset(direct='hello')");
  }

  @Test
  public void testDisableDepsetItems() throws Exception {
    new Scenario("--incompatible_disable_depset_items")
        .setUp("x = depset([0])", "y = depset(direct = [1])")
        .testEval("depset([2, 3], transitive = [x, y]).to_list()", "[0, 1, 2, 3]")
        .testIfErrorContains(
            "parameter 'direct' cannot be specified both positionally and by keyword",
            "depset([0, 1], 'default', direct=[0,1])")
        .testIfErrorContains(
            "in call to depset(), parameter 'items' is deprecated and will be removed soon. "
                + "It may be temporarily re-enabled by setting "
                + "--incompatible_disable_depset_inputs=false",
            "depset(items=[0,1])");
  }

  @Test
  public void testDepsetDepthLimit() throws Exception {
    NestedSet.setApplicationDepthLimit(2000);
    new Scenario()
        .setUp(
            "def create_depset(depth):",
            "  x = depset([0])",
            "  for i in range(1, depth):",
            "    x = depset([i], transitive = [x])",
            "  return x",
            "too_deep_depset = create_depset(3000)",
            "fine_depset = create_depset(900)")
        .testEval("fine_depset.to_list()[0]", "0")
        .testEval("str(fine_depset)[0:6]", "'depset'")
        .testIfErrorContains("depset exceeded maximum depth 2000", "print(too_deep_depset)")
        .testIfErrorContains("depset exceeded maximum depth 2000", "str(too_deep_depset)")
        .testIfErrorContains("depset exceeded maximum depth 2000", "too_deep_depset.to_list()");
  }

  @Test
  public void testDepsetDebugDepth() throws Exception {
    NestedSet.setApplicationDepthLimit(2000);
    new Scenario("--debug_depset_depth=true")
        .setUp(
            "def create_depset(depth):",
            "  x = depset([0])",
            "  for i in range(1, depth):",
            "    x = depset([i], transitive = [x])",
            "  return x")
        .testEval("str(create_depset(900))[0:6]", "'depset'")
        .testIfErrorContains("depset exceeded maximum depth 2000", "create_depset(3000)");
  }
}
