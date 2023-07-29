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

package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import java.util.List;
import net.starlark.java.annot.StarlarkBuiltin;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for MethodLibrary. */
@RunWith(JUnit4.class)
public final class MethodLibraryTest {

  private final EvaluationTestCase ev = new EvaluationTestCase();

  // Asserts that evaluation of src fails with the specified stack.
  private void checkEvalErrorStack(String src, String stack) {
    EvalException ex = assertThrows(EvalException.class, () -> ev.exec(src));
    EvalException.SourceReader reader =
        loc -> {
          // ignore filename
          List<String> lines = Splitter.on('\n').splitToList(src);
          return loc.line() > 0 && loc.line() <= lines.size() ? lines.get(loc.line() - 1) : null;
        };
    assertThat(ex.getMessageWithStack(reader)).isEqualTo(stack);
  }

  private static String join(String... lines) {
    return Joiner.on("\n").join(lines);
  }

  @Test
  public void testStackTrace() throws Exception {
    checkEvalErrorStack(
        join(
            "def foo():", //
            "  bar(1)",
            "def bar(x):",
            "  if x == 1:",
            "    a = x",
            "    b = 2",
            "    'test'.index(x) # hello",
            "foo()"),
        join(
            "Traceback (most recent call last):", //
            "\tFile \"\", line 8, column 4, in <toplevel>",
            "\t\tfoo()",
            "\tFile \"\", line 2, column 6, in foo",
            "\t\tbar(1)",
            "\tFile \"\", line 7, column 17, in bar",
            "\t\t'test'.index(x) # hello",
            "Error in index: in call to index(), parameter 'sub' got value of type 'int', want"
                + " 'string'"));
  }

  @Test
  public void testStackTraceWithIf() throws Exception {
    checkEvalErrorStack(
        join(
            "def foo():", //
            "  s = []",
            "  if s[0] == 1:",
            "    x = 1",
            "foo()"),
        join(
            "Traceback (most recent call last):",
            "\tFile \"\", line 5, column 4, in <toplevel>",
            "\t\tfoo()",
            "\tFile \"\", line 3, column 7, in foo",
            "\t\tif s[0] == 1:",
            "Error: index out of range (index is 0, but sequence has 0 elements)"));
  }

  @Test
  public void testStackTraceWithAugmentedAssignment() throws Exception {
    // Each time the current tree-walking evaluator catches an exception, it computes and sets the
    // frame's error location. Only the first (innermost) such 'set' has any effect. When the frame
    // is popped, its error location is accurate. (In our bytecode future, we'll be able to
    // preemptively set fr.pc cheaply, before every instruction and error, as it's just an int, and
    // thus do away with this.)
    //
    // Assignment statements x=y are special in the evaluator because there's no guarantee that
    // failed evaluation of the subexpressions x or y sets the frame's error location, so
    // Eval(assign) sets the error location to '=', possibly redundantly, to ensure that some
    // location is reported. test exercises that special case.
    checkEvalErrorStack(
        join(
            "def foo():", //
            "  s = 1",
            "  s += '2'",
            "foo()"),
        join(
            "Traceback (most recent call last):", //
            "\tFile \"\", line 4, column 4, in <toplevel>",
            "\t\tfoo()",
            "\tFile \"\", line 3, column 5, in foo",
            "\t\ts += '2'",
            "Error: unsupported binary operation: int + string"));
  }

  @Test
  public void testStackErrorInBuiltinFunction() throws Exception {
    // at top level
    checkEvalErrorStack(
        "len(1)",
        join(
            "Traceback (most recent call last):", //
            "\tFile \"\", line 1, column 4, in <toplevel>",
            "\t\tlen(1)",
            "Error in len: int is not iterable"));

    // in a function
    checkEvalErrorStack(
        join(
            "def f():", //
            "  len(1)",
            "f()"),
        join(
            "Traceback (most recent call last):", //
            "\tFile \"\", line 3, column 2, in <toplevel>",
            "\t\tf()",
            "\tFile \"\", line 2, column 6, in f",
            "\t\tlen(1)",
            "Error in len: int is not iterable"));
  }

  @Test
  public void testStackErrorInOperator() throws Exception {
    // at top level
    checkEvalErrorStack(
        "1//0",
        join(
            "Traceback (most recent call last):", //
            "\tFile \"\", line 1, column 2, in <toplevel>",
            "\t\t1//0",
            "Error: integer division by zero"));

    // in a function
    checkEvalErrorStack(
        join(
            "def f():", //
            "  1//0",
            "f()"),
        join(
            "Traceback (most recent call last):", //
            "\tFile \"\", line 3, column 2, in <toplevel>",
            "\t\tf()",
            "\tFile \"\", line 2, column 4, in f",
            "\t\t1//0",
            "Error: integer division by zero"));

    // in a function callback
    checkEvalErrorStack(
        join(
            "def id(x): return 1//x", //
            "sorted([2, 1, 0], key=id)"),
        join(
            "Traceback (most recent call last):", //
            "\tFile \"\", line 2, column 7, in <toplevel>",
            "\t\tsorted([2, 1, 0], key=id)",
            "\tFile \"<builtin>\", in sorted",
            "\tFile \"\", line 1, column 20, in id",
            "\t\tdef id(x): return 1//x",
            "Error: integer division by zero"));
  }

  @Test
  public void testBuiltinFunctionErrorMessage() throws Exception {
    ev.new Scenario()
        .testIfErrorContains("substring not found", "'abc'.index('z')")
        .testIfErrorContains(
            "in call to startswith(), parameter 'sub' got value of type 'int', want 'string or"
                + " tuple'",
            "'test'.startswith(1)")
        .testIfErrorContains("in dict, got string, want iterable", "dict('a')");
  }

  @Test
  public void testHasAttr() throws Exception {
    ev.new Scenario()
        .testExpression("hasattr([], 'append')", Boolean.TRUE)
        .testExpression("hasattr('test', 'count')", Boolean.TRUE)
        .testExpression("hasattr(dict(a = 1, b = 2), 'items')", Boolean.TRUE)
        .testExpression("hasattr({}, 'items')", Boolean.TRUE);
  }

  @Test
  public void testGetAttrMissingField() throws Exception {
    ev.new Scenario()
        .testIfExactError(
            "'string' value has no field or method 'not_there'", "getattr('a string', 'not_there')")
        .testExpression("getattr('a string', 'not_there', 'use this')", "use this")
        .testExpression("getattr('a string', 'not there', None)", Starlark.NONE);
  }

  @StarlarkBuiltin(name = "AStruct", documented = false, doc = "")
  static final class AStruct implements Structure, StarlarkValue {
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
    ev.new Scenario()
        .update("s", new AStruct())
        .testIfExactError(
            "'AStruct' value has no field or method 'feild' (did you mean 'field'?)",
            "getattr(s, 'feild')");
  }

  @Test
  public void testGetAttrWithMethods() throws Exception {
    String msg = "'string' value has no field or method 'cnt'";
    ev.new Scenario()
        .testIfExactError(msg, "getattr('a string', 'cnt')")
        .testExpression("getattr('a string', 'cnt', 'default')", "default");
  }

  @Test
  public void testDir() throws Exception {
    ev.new Scenario()
        .testExpression(
            "str(dir({}))",
            "[\"clear\", \"get\", \"items\", \"keys\","
                + " \"pop\", \"popitem\", \"setdefault\", \"update\", \"values\"]");
  }

  @Test
  public void testAbs() throws Exception {
    ev.new Scenario()
        // int
        .testEval("abs(4)", "4")
        .testEval("abs(-2)", "2")
        .testEval("abs(0)", "0")
        // float
        .testEval("abs(-2.3)", "2.3")
        .testEval("abs(5.2)", "5.2")
        .testEval("abs(0.0)", "0.0")
        // big int
        .testEval("abs(-12345678901234567890)", "12345678901234567890")
        .testEval("abs(12345678901234567890)", "12345678901234567890");
  }

  @Test
  public void testBoolean() throws Exception {
    ev.new Scenario().testExpression("False", Boolean.FALSE).testExpression("True", Boolean.TRUE);
  }

  @Test
  public void testBooleanUnsupportedOperationFails() throws Exception {
    ev.new Scenario()
        .testIfErrorContains("unsupported binary operation: bool + bool", "True + True");
  }

  @Test
  public void testListSort() throws Exception {
    ev.new Scenario()
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
        .testIfExactError(
            "unsupported comparison: builtin_function_or_method <=> builtin_function_or_method",
            "sorted([sorted, sorted])");
  }

  @Test
  public void testDictionaryCopy() throws Exception {
    ev.new Scenario()
        .setUp("x = {1 : 2}", "y = dict(x)")
        .testEval("x[1] == 2 and y[1] == 2", "True");
  }

  @Test
  public void testDictionaryCopyKeyCollision() throws Exception {
    ev.new Scenario().setUp("x = {'test' : 2}", "y = dict(x, test = 3)").testEval("y['test']", "3");
  }

  @Test
  public void testDictionaryKeyNotFound() throws Exception {
    ev.new Scenario()
        .testIfErrorContains("key \"0\" not found in dictionary", "{}['0']")
        .testIfErrorContains("key 0 not found in dictionary", "{'0': 1, 2: 3, 4: 5}[0]");
  }

  @Test
  public void testDictionaryAccess() throws Exception {
    ev.new Scenario()
        .testEval("{1: ['foo']}[1]", "['foo']")
        .testExpression("{'4': 8}['4']", StarlarkInt.of(8))
        .testExpression("{'a': 'aa', 'b': 'bb', 'c': 'cc'}['b']", "bb");
  }

  @Test
  public void testDictionaryVariableAccess() throws Exception {
    ev.new Scenario().setUp("d = {'a' : 1}", "a = d['a']").testLookup("a", StarlarkInt.of(1));
  }

  @Test
  public void testDictionaryCreation() throws Exception {
    String expected = "{'a': 1, 'b': 2, 'c': 3}";

    ev.new Scenario()
        .testEval("dict([('a', 1), ('b', 2), ('c', 3)])", expected)
        .testEval("dict(a = 1, b = 2, c = 3)", expected)
        .testEval("dict([('a', 1)], b = 2, c = 3)", expected);
  }

  @Test
  public void testDictionaryCreationInnerLists() throws Exception {
    ev.new Scenario().testEval("dict([[1, 2], [3, 4]], a = 5)", "{1: 2, 3: 4, 'a': 5}");
  }

  @Test
  public void testDictionaryCreationEmpty() throws Exception {
    ev.new Scenario().testEval("dict()", "{}").testEval("dict([])", "{}");
  }

  @Test
  public void testDictionaryCreationDifferentKeyTypes() throws Exception {
    String expected = "{'a': 1, 2: 3}";

    ev.new Scenario()
        .testEval("dict([('a', 1), (2, 3)])", expected)
        .testEval("dict([(2, 3)], a = 1)", expected);
  }

  @Test
  public void testDictionaryCreationKeyCollision() throws Exception {
    String expected = "{'a': 1, 'b': 2, 'c': 3}";

    ev.new Scenario()
        .testEval("dict([('a', 42), ('b', 2), ('a', 1), ('c', 3)])", expected)
        .testEval("dict([('a', 42)], a = 1, b = 2, c = 3)", expected);
    ev.new Scenario().testEval("dict([('a', 42)], **{'a': 1, 'b': 2, 'c': 3})", expected);
  }

  @Test
  public void testDictionaryCreationInvalidPositional() throws Exception {
    ev.new Scenario()
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
    ev.new Scenario()
        .testEval("{1: 'foo'}.values()", "['foo']")
        .testEval("{}.values()", "[]")
        .testEval("{True: 3, False: 5}.values()", "[3, 5]")
        .testEval("{'a': 5, 'c': 2, 'b': 4, 'd': 3}.values()", "[5, 2, 4, 3]");
    // sorted by keys
  }

  @Test
  public void testDictionaryKeys() throws Exception {
    ev.new Scenario()
        .testEval("{1: 'foo'}.keys()", "[1]")
        .testEval("{}.keys()", "[]")
        .testEval("{True: 3, False: 5}.keys()", "[True, False]")
        .testEval(
            "{1:'a', 2:'b', 6:'c', 0:'d', 5:'e', 4:'f', 3:'g'}.keys()", "[1, 2, 6, 0, 5, 4, 3]");
  }

  @Test
  public void testDictionaryGet() throws Exception {
    ev.new Scenario()
        .testExpression("{1: 'foo'}.get(1)", "foo")
        .testExpression("{1: 'foo'}.get(2)", Starlark.NONE)
        .testExpression("{1: 'foo'}.get(2, 'a')", "a")
        .testExpression("{1: 'foo'}.get(2, default='a')", "a")
        .testExpression("{1: 'foo'}.get(2, default=None)", Starlark.NONE);
  }

  @Test
  public void testDictionaryItems() throws Exception {
    ev.new Scenario()
        .testEval("{'a': 'foo'}.items()", "[('a', 'foo')]")
        .testEval("{}.items()", "[]")
        .testEval("{1: 3, 2: 5}.items()", "[(1, 3), (2, 5)]")
        .testEval("{'a': 5, 'c': 2, 'b': 4}.items()", "[('a', 5), ('c', 2), ('b', 4)]");
  }

  @Test
  public void testDictionaryClear() throws Exception {
    ev.new Scenario()
        .setUp(
            "d = {1: 'foo', 2: 'bar', 3: 'baz'}",
            "len(d) == 3 or fail('clear 1')",
            "d.clear() == None or fail('clear 2')")
        .testEval("d", "{}");
  }

  @Test
  public void testDictionaryPop() throws Exception {
    ev.new Scenario()
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
    ev.new Scenario()
        .testIfErrorContains(
            "popitem: empty dictionary",
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
    ev.new Scenario()
        .setUp("foo = {'a': 2}", "foo.update({'b': 4})")
        .testEval("foo", "{'a': 2, 'b': 4}");
    ev.new Scenario()
        .setUp("foo = {'a': 2}", "foo.update({'a': 3, 'b': 4})")
        .testEval("foo", "{'a': 3, 'b': 4}");
  }

  @Test
  public void testDictionarySetDefault() throws Exception {
    ev.new Scenario()
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
    ev.new Scenario()
        .testExpression("['a', 'b', 'c'].index('a')", StarlarkInt.of(0))
        .testExpression("['a', 'b', 'c'].index('b')", StarlarkInt.of(1))
        .testExpression("['a', 'b', 'c'].index('c')", StarlarkInt.of(2))
        .testExpression("[2, 4, 6].index(4)", StarlarkInt.of(1))
        .testExpression("[2, 4, 6].index(4)", StarlarkInt.of(1))
        .testExpression("[0, 1, [1]].index([1])", StarlarkInt.of(2))
        .testIfErrorContains("item \"a\" not found in list", "[1, 2].index('a')")
        .testIfErrorContains("item 0 not found in list", "[].index(0)");
  }

  @Test
  public void testHash() throws Exception {
    // We specify the same string hashing algorithm as String.hashCode().
    ev.new Scenario()
        .testExpression("hash('starlark')", StarlarkInt.of("starlark".hashCode()))
        .testExpression("hash('google')", StarlarkInt.of("google".hashCode()))
        .testIfErrorContains(
            "in call to hash(), parameter 'value' got value of type 'NoneType', want 'string'",
            "hash(None)");
  }

  @Test
  public void testRangeType() throws Exception {
    ev.new Scenario()
        .setUp("a = range(3)")
        .testExpression("len(a)", StarlarkInt.of(3))
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
        .testExpression("range(3)[-1]", StarlarkInt.of(2))
        .testIfErrorContains(
            "index out of range (index is 3, but sequence has 3 elements)", "range(3)[3]")
        .testExpression("str(range(5)[1:])", "range(1, 5)")
        .testExpression("len(range(5)[1:])", StarlarkInt.of(4))
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
        .testIfErrorContains("unsupported comparison: range <=> range", "range(3) < range(5)")
        .testIfErrorContains("unsupported comparison: range <=> list", "range(4) > [1]")
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
    ev.new Scenario()
        .testExpression("str(enumerate([]))", "[]")
        .testExpression("str(enumerate([5]))", "[(0, 5)]")
        .testExpression("str(enumerate([5, 3]))", "[(0, 5), (1, 3)]")
        .testExpression("str(enumerate(['a', 'b', 'c']))", "[(0, \"a\"), (1, \"b\"), (2, \"c\")]")
        .testExpression("str(enumerate(['a']) + [(1, 'b')])", "[(0, \"a\"), (1, \"b\")]");
  }

  @Test
  public void testEnumerateBadArg() throws Exception {
    ev.new Scenario().testIfErrorContains("type 'string' is not iterable", "enumerate('a')");
  }

  @Test
  public void testReassignmentOfPrimitivesNotForbiddenByCoreLanguage() throws Exception {
    ev.new Scenario()
        .setUp("cc_binary = (['hello.cc'])")
        .testIfErrorContains(
            "'list' object is not callable",
            "cc_binary(name = 'hello', srcs=['hello.cc'], malloc = '//base:system_malloc')");
  }

  @Test
  public void testLenOnString() throws Exception {
    ev.new Scenario().testExpression("len('abc')", StarlarkInt.of(3));
  }

  @Test
  public void testLenOnList() throws Exception {
    ev.new Scenario().testExpression("len([1,2,3])", StarlarkInt.of(3));
  }

  @Test
  public void testLenOnDict() throws Exception {
    ev.new Scenario().testExpression("len({'a' : 1, 'b' : 2})", StarlarkInt.of(2));
  }

  @Test
  public void testLenOnBadType() throws Exception {
    ev.new Scenario().testIfErrorContains("int is not iterable", "len(1)");
  }

  @Test
  public void testIndexOnFunction() throws Exception {
    ev.new Scenario()
        .testIfErrorContains("type 'builtin_function_or_method' has no operator [](int)", "len[1]")
        .testIfErrorContains("invalid slice operand: builtin_function_or_method", "len[1:4]");
  }

  @Test
  public void testBool() throws Exception {
    ev.new Scenario()
        .testExpression("bool(1)", Boolean.TRUE)
        .testExpression("bool(0)", Boolean.FALSE)
        .testExpression("bool([1, 2])", Boolean.TRUE)
        .testExpression("bool([])", Boolean.FALSE)
        .testExpression("bool(None)", Boolean.FALSE);
  }

  @Test
  public void testStr() throws Exception {
    ev.new Scenario()
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
    ev.new Scenario().setUp("def foo(x): pass").testExpression("str(foo)", "<function foo>");
  }

  @Test
  public void testType() throws Exception {
    ev.new Scenario()
        .setUp("def f(): pass")
        .testExpression("type(1)", "int")
        .testExpression("type('a')", "string")
        .testExpression("type([1, 2])", "list")
        .testExpression("type((1, 2))", "tuple")
        .testExpression("type((1,))", "tuple")
        .testExpression("type(True)", "bool")
        .testExpression("type(None)", "NoneType")
        .testExpression("type(f)", "function")
        .testExpression("type(str)", "builtin_function_or_method");
  }

  @Test
  public void testZipFunction() throws Exception {
    ev.new Scenario()
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
      ev.new Scenario()
          .update("s", input)
          .testExpression("s.lstrip()", expLeft)
          .testExpression("s.rstrip()", expRight)
          .testExpression("s.strip()", expBoth);
    } else {
      ev.new Scenario()
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
    ev.new Scenario()
        .testIfErrorContains("abc", "fail('abc')")
        .testIfErrorContains("18", "fail(18)")
        .testIfErrorContains("1 2 3", "fail(1, 2, 3)")
        .testIfErrorContains("attribute foo: 1 2 3", "fail(1, 2, 3, attr='foo')") // deprecated
        .testIfErrorContains("0 1 2 3", "fail(1, 2, 3, msg=0)"); // deprecated
  }

  @Test
  public void testTupleCoercion() throws Exception {
    ev.new Scenario()
        .testExpression("tuple([1, 2]) == (1, 2)", true)
        .testExpression("tuple({1: 'foo', 2: 'bar'}) == (1, 2)", true);
  }

  @Test
  public void testPositionalOnlyArgument() throws Exception {
    ev.new Scenario()
        .testIfErrorContains(
            "join() got named argument for positional-only parameter 'elements'",
            "','.join(elements=['foo', 'bar'])");
  }

  @Test
  public void testStringJoinRequiresStrings() throws Exception {
    ev.new Scenario()
        .testIfErrorContains(
            "expected string for sequence element 1, got '2' of type int", "', '.join(['foo', 2])");
  }
}
