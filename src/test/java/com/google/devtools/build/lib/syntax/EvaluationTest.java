// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestMode;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test of evaluation behavior.  (Implicitly uses lexer + parser.)
 */
@RunWith(JUnit4.class)
public class EvaluationTest extends EvaluationTestCase {

  @Before
  public final void setBuildMode() throws Exception {
    super.setMode(TestMode.BUILD);
  }

  /**
   * Creates a new instance of {@code ModalTestCase}.
   *
   * <p>If a test uses this method, it allows potential subclasses to run the very same test in a
   * different mode in subclasses
   */
  protected ModalTestCase newTest(String... skylarkOptions) {
    return new BuildTest(skylarkOptions);
  }

  @Test
  public void testExprs() throws Exception {
    newTest()
        .testStatement("'%sx' % 'foo' + 'bar1'", "fooxbar1")
        .testStatement("('%sx' % 'foo') + 'bar2'", "fooxbar2")
        .testStatement("'%sx' % ('foo' + 'bar3')", "foobar3x")
        .testStatement("123 + 456", 579)
        .testStatement("456 - 123", 333)
        .testStatement("8 % 3", 2)
        .testIfErrorContains("unsupported operand type(s) for %: 'int' and 'string'", "3 % 'foo'")
        .testStatement("-5", -5)
        .testIfErrorContains("unsupported operand type for -: 'string'", "-'foo'");
  }

  @Test
  public void testListExprs() throws Exception {
    newTest().testExactOrder("[1, 2, 3]", 1, 2, 3).testExactOrder("(1, 2, 3)", 1, 2, 3);
  }

  @Test
  public void testStringFormatMultipleArgs() throws Exception {
    newTest().testStatement("'%sY%s' % ('X', 'Z')", "XYZ");
  }

  @Test
  public void testConditionalExpressions() throws Exception {
    newTest()
        .testStatement("1 if True else 2", 1)
        .testStatement("1 if False else 2", 2)
        .testStatement("1 + 2 if 3 + 4 else 5 + 6", 3);

    setFailFast(false);
    parseExpression("1 if 2");
    assertContainsError(
        "missing else clause in conditional expression or semicolon before if");
  }

  @Test
  public void testListComparison() throws Exception {
    newTest()
        .testStatement("[] < [1]", true)
        .testStatement("[1] < [1, 1]", true)
        .testStatement("[1, 1] < [1, 2]", true)
        .testStatement("[1, 2] < [1, 2, 3]", true)
        .testStatement("[1, 2, 3] <= [1, 2, 3]", true)

        .testStatement("['a', 'b'] > ['a']", true)
        .testStatement("['a', 'b'] >= ['a']", true)
        .testStatement("['a', 'b'] < ['a']", false)
        .testStatement("['a', 'b'] <= ['a']", false)

        .testStatement("('a', 'b') > ('a', 'b')", false)
        .testStatement("('a', 'b') >= ('a', 'b')", true)
        .testStatement("('a', 'b') < ('a', 'b')", false)
        .testStatement("('a', 'b') <= ('a', 'b')", true)

        .testStatement("[[1, 1]] > [[1, 1], []]", false)
        .testStatement("[[1, 1]] < [[1, 1], []]", true);
  }

  @Test
  public void testSetComparison() throws Exception {
    newTest().testIfExactError("Cannot compare depsets", "depset([1, 2]) < depset([3, 4])");
  }

  @Test
  public void testSumFunction() throws Exception {
    BaseFunction sum = new BaseFunction("sum") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs,
          FuncallExpression ast, Environment env) {
        int sum = 0;
        for (Object arg : args) {
          sum += (Integer) arg;
        }
        return sum;
      }
    };

    newTest().update(sum.getName(), sum).testStatement("sum(1, 2, 3, 4, 5, 6)", 21)
        .testStatement("sum", sum).testStatement("sum(a=1, b=2)", 0);
  }

  @Test
  public void testNotCallInt() throws Exception {
    newTest().setUp("sum = 123456").testLookup("sum", 123456)
        .testIfExactError("'int' object is not callable", "sum(1, 2, 3, 4, 5, 6)")
        .testStatement("sum", 123456);
  }

  @Test
  public void testComplexFunctionCall() throws Exception {
    newTest().setUp("functions = [min, max]", "l = [1,2]")
        .testEval("(functions[0](l), functions[1](l))", "(1, 2)");
  }

  @Test
  public void testKeywordArgs() throws Exception {

    // This function returns the map of keyword arguments passed to it.
    BaseFunction kwargs = new BaseFunction("kwargs") {
      @Override
      public Object call(List<Object> args,
          final Map<String, Object> kwargs,
          FuncallExpression ast,
          Environment env) {
        return SkylarkDict.copyOf(env, kwargs);
      }
    };

    newTest()
        .update(kwargs.getName(), kwargs)
        .testEval(
            "kwargs(foo=1, bar='bar', wiz=[1,2,3]).items()",
            "[('foo', 1), ('bar', 'bar'), ('wiz', [1, 2, 3])]")
        .testEval(
            "kwargs(wiz=[1,2,3], bar='bar', foo=1).items()",
            "[('wiz', [1, 2, 3]), ('bar', 'bar'), ('foo', 1)]");
  }

  @Test
  public void testModulo() throws Exception {
    newTest()
        .testStatement("6 % 2", 0)
        .testStatement("6 % 4", 2)
        .testStatement("3 % 6", 3)
        .testStatement("7 % -4", -1)
        .testStatement("-7 % 4", 1)
        .testStatement("-7 % -4", -3)
        .testIfExactError("integer modulo by zero", "5 % 0");
  }

  @Test
  public void testMult() throws Exception {
    newTest()
        .testStatement("6 * 7", 42)
        .testStatement("3 * 'ab'", "ababab")
        .testStatement("0 * 'ab'", "")
        .testStatement("'1' + '0' * 5", "100000")
        .testStatement("'ab' * -4", "")
        .testStatement("-1 * ''", "");
  }

  @Test
  public void testSlashOperatorIsForbidden() throws Exception {
    newTest("--incompatible_disallow_slash_operator=true")
        .testIfErrorContains("The `/` operator has been removed.", "5 / 2");
  }

  @Test
  public void testDivision() throws Exception {
    newTest("--incompatible_disallow_slash_operator=false")
        .testStatement("6 / 2", 3)
        .testStatement("6 / 4", 1)
        .testStatement("3 / 6", 0)
        .testStatement("7 / -2", -4)
        .testStatement("-7 / 2", -4)
        .testStatement("-7 / -2", 3)
        .testStatement("2147483647 / 2", 1073741823)
        .testIfErrorContains("unsupported operand type(s) for //: 'string' and 'int'", "'str' / 2")
        .testIfExactError("integer division by zero", "5 / 0");
  }

  @Test
  public void testFloorDivision() throws Exception {
    newTest()
        .testStatement("6 // 2", 3)
        .testStatement("6 // 4", 1)
        .testStatement("3 // 6", 0)
        .testStatement("7 // -2", -4)
        .testStatement("-7 // 2", -4)
        .testStatement("-7 // -2", 3)
        .testStatement("2147483647 // 2", 1073741823)
        .testIfErrorContains("unsupported operand type(s) for //: 'string' and 'int'", "'str' // 2")
        .testIfExactError("integer division by zero", "5 // 0");
  }

  @Test
  public void testCheckedArithmetic() throws Exception {
    new SkylarkTest()
        .testIfErrorContains("integer overflow", "2000000000 + 2000000000")
        .testIfErrorContains("integer overflow", "1234567890 * 987654321")
        .testIfErrorContains("integer overflow", "- 2000000000 - 2000000000")

        // literal 2147483648 is not allowed, so we compute it
        .setUp("minint = - 2147483647 - 1")
        .testIfErrorContains("integer overflow", "-minint");
  }

  @Test
  public void testOperatorPrecedence() throws Exception {
    newTest()
        .testStatement("2 + 3 * 4", 14)
        .testStatement("2 + 3 // 4", 2)
        .testStatement("2 * 3 + 4 // -2", 4);
  }

  @Test
  public void testConcatStrings() throws Exception {
    newTest().testStatement("'foo' + 'bar'", "foobar");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testConcatLists() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    // list
    Object x = eval("[1,2] + [3,4]");
    assertThat((Iterable<Object>) x).containsExactly(1, 2, 3, 4).inOrder();
    assertThat(x).isEqualTo(MutableList.of(env, 1, 2, 3, 4));
    assertThat(EvalUtils.isImmutable(x)).isFalse();

    // tuple
    x = eval("(1,2) + (3,4)");
    assertThat(x).isEqualTo(Tuple.of(1, 2, 3, 4));
    assertThat(EvalUtils.isImmutable(x)).isTrue();

    checkEvalError("unsupported operand type(s) for +: 'tuple' and 'list'",
        "(1,2) + [3,4]"); // list + tuple
  }

  @Test
  public void testListComprehensions() throws Exception {
    newTest()
        .testExactOrder("['foo/%s.java' % x for x in []]")
        .testExactOrder("['foo/%s.java' % y for y in ['bar', 'wiz', 'quux']]", "foo/bar.java",
            "foo/wiz.java", "foo/quux.java")
        .testExactOrder("['%s/%s.java' % (z, t) for z in ['foo', 'bar'] "
            + "for t in ['baz', 'wiz', 'quux']]",
            "foo/baz.java",
            "foo/wiz.java",
            "foo/quux.java",
            "bar/baz.java",
            "bar/wiz.java",
            "bar/quux.java")
        .testExactOrder("['%s/%s.java' % (b, b) for a in ['foo', 'bar'] "
            + "for b in ['baz', 'wiz', 'quux']]",
            "baz/baz.java",
            "wiz/wiz.java",
            "quux/quux.java",
            "baz/baz.java",
            "wiz/wiz.java",
            "quux/quux.java")
        .testExactOrder("['%s/%s.%s' % (c, d, e) for c in ['foo', 'bar'] "
            + "for d in ['baz', 'wiz', 'quux'] for e in ['java', 'cc']]",
            "foo/baz.java",
            "foo/baz.cc",
            "foo/wiz.java",
            "foo/wiz.cc",
            "foo/quux.java",
            "foo/quux.cc",
            "bar/baz.java",
            "bar/baz.cc",
            "bar/wiz.java",
            "bar/wiz.cc",
            "bar/quux.java",
            "bar/quux.cc")
        .testExactOrder("[i for i in (1, 2)]", 1, 2)
        .testExactOrder("[i for i in [2, 3] or [1, 2]]", 2, 3);
  }

  @Test
  public void testNestedListComprehensions() throws Exception {
    newTest().testExactOrder("li = [[1, 2], [3, 4]]\n" + "[j for i in li for j in i]", 1, 2,
        3, 4).testExactOrder("input = [['abc'], ['def', 'ghi']]\n"
        + "['%s %s' % (b, c) for a in input for b in a for c in b]",
        "abc a",
        "abc b",
        "abc c",
        "def d",
        "def e",
        "def f",
        "ghi g",
        "ghi h",
        "ghi i");
  }

  @Test
  public void testListComprehensionsMultipleVariables() throws Exception {
    newTest().testEval("[x + y for x, y in [(1, 2), (3, 4)]]", "[3, 7]")
        .testEval("[z + t for (z, t) in [[1, 2], [3, 4]]]", "[3, 7]");
  }

  @Test
  public void testListComprehensionsMultipleVariablesFail() throws Exception {
    newTest().testIfErrorContains(
        "assignment length mismatch: left-hand side has length 3, but right-hand side evaluates to "
            + "value of length 2",
        "[x + y for x, y, z in [(1, 2), (3, 4)]]").testIfExactError(
        "type 'int' is not a collection", "[x + y for x, y in (1, 2)]");
  }

  @Test
  public void testListComprehensionsWithFiltering() throws Exception {
    newTest()
        .setUp("range3 = [0, 1, 2]")
        .testEval("[a for a in (4, None, 2, None, 1) if a != None]", "[4, 2, 1]")
        .testEval("[b+c for b in [0, 1, 2] for c in [0, 1, 2] if b + c > 2]", "[3, 3, 4]")
        .testEval("[d+e for d in range3 if d % 2 == 1 for e in range3]", "[1, 2, 3]")
        .testEval("[[f,g] for f in [0, 1, 2, 3, 4] if f for g in [5, 6, 7, 8] if f * g % 12 == 0 ]",
            "[[2, 6], [3, 8], [4, 6]]")
        .testEval("[h for h in [4, 2, 0, 1] if h]", "[4, 2, 1]");
  }

  @Test
  public void testListComprehensionDefinitionOrder() throws Exception {
    newTest().testIfErrorContains("name 'y' is not defined",
        "[x for x in (1, 2) if y for y in (3, 4)]");
  }

  @Test
  public void testTupleDestructuring() throws Exception {
    newTest()
        .setUp("a, b = 1, 2")
        .testLookup("a", 1)
        .testLookup("b", 2)
        .setUp("c, d = {'key1':2, 'key2':3}")
        .testLookup("c", "key1")
        .testLookup("d", "key2");
  }

  @Test
  public void testSingleTuple() throws Exception {
    newTest().setUp("(a,) = [1]").testLookup("a", 1);
  }

  @Test
  public void testHeterogeneousDict() throws Exception {
    newTest().setUp("d = {'str': 1, 2: 3}", "a = d['str']", "b = d[2]").testLookup("a", 1)
        .testLookup("b", 3);
  }

  @Test
  public void testAccessDictWithATupleKey() throws Exception {
    newTest().setUp("x = {(1, 2): 3}[1, 2]").testLookup("x", 3);
  }

  @Test
  public void testDictWithDuplicatedKey() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "Duplicated key \"str\" when creating dictionary", "{'str': 1, 'x': 2, 'str': 3}");
  }

  @Test
  public void testRecursiveTupleDestructuring() throws Exception {
    newTest()
        .setUp("((a, b), (c, d)) = [(1, 2), (3, 4)]")
        .testLookup("a", 1)
        .testLookup("b", 2)
        .testLookup("c", 3)
        .testLookup("d", 4);
  }

  @Test
  public void testListComprehensionAtTopLevel() throws Exception {
    // It is allowed to have a loop variable with the same name as a global variable.
    newTest().update("x", 42).setUp("y = [x + 1 for x in [1,2,3]]")
        .testExactOrder("y", 2, 3, 4);
  }

  @Test
  public void testDictComprehensions() throws Exception {
    newTest()
        .testStatement("{a : a for a in []}", Collections.emptyMap())
        .testStatement("{b : b for b in [1, 2]}", ImmutableMap.of(1, 1, 2, 2))
        .testStatement("{c : 'v_' + c for c in ['a', 'b']}",
            ImmutableMap.of("a", "v_a", "b", "v_b"))
        .testStatement("{'k_' + d : d for d in ['a', 'b']}",
            ImmutableMap.of("k_a", "a", "k_b", "b"))
        .testStatement("{'k_' + e : 'v_' + e for e in ['a', 'b']}",
            ImmutableMap.of("k_a", "v_a", "k_b", "v_b"))
        .testStatement("{x+y : x*y for x, y in [[2, 3]]}", ImmutableMap.of(5, 6));
  }

  @Test
  public void testDictComprehensionOnNonIterable() throws Exception {
    newTest().testIfExactError("type 'int' is not iterable", "{k : k for k in 3}");
  }

  @Test
  public void testDictComprehension_ManyClauses() throws Exception {
    new SkylarkTest().testStatement(
        "{x : x * y for x in range(1, 10) if x % 2 == 0 for y in range(1, 10) if y == x}",
        ImmutableMap.of(2, 4, 4, 16, 6, 36, 8, 64));
  }

  @Test
  public void testDictComprehensions_MultipleKey() throws Exception {
    newTest().testStatement("{x : x for x in [1, 2, 1]}", ImmutableMap.of(1, 1, 2, 2))
        .testStatement("{y : y for y in ['ab', 'c', 'a' + 'b']}",
            ImmutableMap.of("ab", "ab", "c", "c"));
  }

  @Test
  public void testListConcatenation() throws Exception {
    newTest()
        .testStatement("[1, 2] + [3, 4]", MutableList.of(env, 1, 2, 3, 4))
        .testStatement("(1, 2) + (3, 4)", Tuple.of(1, 2, 3, 4))
        .testIfExactError("unsupported operand type(s) for +: 'list' and 'tuple'",
            "[1, 2] + (3, 4)")
        .testIfExactError("unsupported operand type(s) for +: 'tuple' and 'list'",
            "(1, 2) + [3, 4]");
  }

  @Test
  public void testListMultiply() throws Exception {
    newTest()
        .testStatement("[1, 2, 3] * 1", MutableList.of(env, 1, 2, 3))
        .testStatement("[1, 2] * 2", MutableList.of(env, 1, 2, 1, 2))
        .testStatement("[1, 2] * 3", MutableList.of(env, 1, 2, 1, 2, 1, 2))
        .testStatement("[1, 2] * 4", MutableList.of(env, 1, 2, 1, 2, 1, 2, 1, 2))
        .testStatement("[8] * 5", MutableList.of(env, 8, 8, 8, 8, 8))
        .testStatement("[    ] * 10", MutableList.empty())
        .testStatement("[1, 2] * 0", MutableList.empty())
        .testStatement("[1, 2] * -4", MutableList.empty())
        .testStatement("2 * [1, 2]", MutableList.of(env, 1, 2, 1, 2))
        .testStatement("10 * []", MutableList.empty())
        .testStatement("0 * [1, 2]", MutableList.empty())
        .testStatement("-4 * [1, 2]", MutableList.empty());
  }

  @Test
  public void testTupleMultiply() throws Exception {
    newTest()
        .testStatement("(1, 2, 3) * 1", Tuple.of(1, 2, 3))
        .testStatement("(1, 2) * 2", Tuple.of(1, 2, 1, 2))
        .testStatement("(1, 2) * 3", Tuple.of(1, 2, 1, 2, 1, 2))
        .testStatement("(1, 2) * 4", Tuple.of(1, 2, 1, 2, 1, 2, 1, 2))
        .testStatement("(8,) * 5", Tuple.of(8, 8, 8, 8, 8))
        .testStatement("(    ) * 10", Tuple.empty())
        .testStatement("(1, 2) * 0", Tuple.empty())
        .testStatement("(1, 2) * -4", Tuple.empty())
        .testStatement("2 * (1, 2)", Tuple.of(1, 2, 1, 2))
        .testStatement("10 * ()", Tuple.empty())
        .testStatement("0 * (1, 2)", Tuple.empty())
        .testStatement("-4 * (1, 2)", Tuple.empty());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testSelectorListConcatenation() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    SelectorList x = (SelectorList) eval("select({'foo': ['FOO'], 'bar': ['BAR']}) + []");
    List<Object> elements = x.getElements();
    assertThat(elements).hasSize(2);
    assertThat(elements.get(0)).isInstanceOf(SelectorValue.class);
    assertThat((Iterable<Object>) elements.get(1)).isEmpty();
  }

  @Test
  public void testAddSelectIncompatibleType() throws Exception {
    newTest()
        .testIfErrorContains(
            "'+' operator applied to incompatible types (select of list, int)",
            "select({'foo': ['FOO'], 'bar': ['BAR']}) + 1");
  }

  @Test
  public void testAddSelectIncompatibleType2() throws Exception {
    newTest()
        .testIfErrorContains(
            "'+' operator applied to incompatible types (select of list, select of int)",
            "select({'foo': ['FOO']}) + select({'bar': 2})");
  }

  @Test
  public void testListComprehensionFailsOnNonSequence() throws Exception {
    newTest().testIfErrorContains("type 'int' is not iterable", "[x + 1 for x in 123]");
  }

  @Test
  public void testListComprehensionOnString() throws Exception {
    newTest("--incompatible_string_is_not_iterable=false")
        .testExactOrder("[x for x in 'abc']", "a", "b", "c");
  }

  @Test
  public void testListComprehensionOnStringIsForbidden() throws Exception {
    newTest("--incompatible_string_is_not_iterable=true")
        .testIfErrorContains("type 'string' is not iterable", "[x for x in 'abc']");
  }

  @Test
  public void testInvalidAssignment() throws Exception {
    newTest().testIfErrorContains(
        "cannot assign to 'x + 1'", "x + 1 = 2");
  }

  @Test
  public void testListComprehensionOnDictionary() throws Exception {
    newTest().testExactOrder("val = ['var_' + n for n in {'a':1,'b':2}] ; val", "var_a", "var_b");
  }

  @Test
  public void testListComprehensionOnDictionaryCompositeExpression() throws Exception {
    new BuildTest()
        .setUp("d = {1:'a',2:'b'}", "l = [d[x] for x in d]")
        .testLookup("l", MutableList.of(env, "a", "b"));
  }

  @Test
  public void testListComprehensionUpdate() throws Exception {
    new BuildTest()
        .setUp("xs = [1, 2, 3]")
        .testIfErrorContains("trying to mutate a locked object",
            "[xs.append(4) for x in xs]");
  }

  @Test
  public void testNestedListComprehensionUpdate() throws Exception {
    new BuildTest()
        .setUp("xs = [1, 2, 3]")
        .testIfErrorContains("trying to mutate a locked object",
            "[xs.append(4) for x in xs for y in xs]");
  }

  @Test
  public void testListComprehensionUpdateInClause() throws Exception {
    new BuildTest()
        .setUp("xs = [1, 2, 3]")
        .testIfErrorContains("trying to mutate a locked object",
            // Use short-circuiting to produce valid output in the event
            // the exception is not raised.
            "[y for x in xs for y in (xs.append(4) or xs)]");
  }

  @Test
  public void testDictComprehensionUpdate() throws Exception {
    new BuildTest()
        .setUp("xs = {1:1, 2:2, 3:3}")
        .testIfErrorContains("trying to mutate a locked object",
            "[xs.popitem() for x in xs]");
  }

  @Test
  public void testInOperator() throws Exception {
    newTest()
        .testStatement("'b' in ['a', 'b']", Boolean.TRUE)
        .testStatement("'c' in ['a', 'b']", Boolean.FALSE)
        .testStatement("'b' in ('a', 'b')", Boolean.TRUE)
        .testStatement("'c' in ('a', 'b')", Boolean.FALSE)
        .testStatement("'b' in {'a' : 1, 'b' : 2}", Boolean.TRUE)
        .testStatement("'c' in {'a' : 1, 'b' : 2}", Boolean.FALSE)
        .testStatement("1 in {'a' : 1, 'b' : 2}", Boolean.FALSE)
        .testStatement("'b' in 'abc'", Boolean.TRUE)
        .testStatement("'d' in 'abc'", Boolean.FALSE);
  }

  @Test
  public void testNotInOperator() throws Exception {
    newTest()
        .testStatement("'b' not in ['a', 'b']", Boolean.FALSE)
        .testStatement("'c' not in ['a', 'b']", Boolean.TRUE)
        .testStatement("'b' not in ('a', 'b')", Boolean.FALSE)
        .testStatement("'c' not in ('a', 'b')", Boolean.TRUE)
        .testStatement("'b' not in {'a' : 1, 'b' : 2}", Boolean.FALSE)
        .testStatement("'c' not in {'a' : 1, 'b' : 2}", Boolean.TRUE)
        .testStatement("1 not in {'a' : 1, 'b' : 2}", Boolean.TRUE)
        .testStatement("'b' not in 'abc'", Boolean.FALSE)
        .testStatement("'d' not in 'abc'", Boolean.TRUE);
  }

  @Test
  public void testInFail() throws Exception {
    newTest()
        .testIfErrorContains(
            "'in <string>' requires string as left operand, not 'int'", "1 in '123'")
        .testIfErrorContains("'int' is not iterable. in operator only works on ", "'a' in 1");
  }

  @Test
  public void testInCompositeForPrecedence() throws Exception {
    newTest().testStatement("not 'a' in ['a'] or 0", 0);
  }

  private SkylarkValue createObjWithStr() {
    return new SkylarkValue() {
      @Override
      public void repr(SkylarkPrinter printer) {
        printer.append("<str marker>");
      }
    };
  }

  @Test
  public void testPercOnObject() throws Exception {
    newTest()
        .update("obj", createObjWithStr())
        .testStatement("'%s' % obj", "<str marker>");
    newTest()
        .update("unknown", new Object())
        .testStatement("'%s' % unknown", "<unknown object java.lang.Object>");
  }

  @Test
  public void testPercOnObjectList() throws Exception {
    newTest()
        .update("obj", createObjWithStr())
        .testStatement("'%s %s' % (obj, obj)", "<str marker> <str marker>");
    newTest()
        .update("unknown", new Object())
        .testStatement(
            "'%s %s' % (unknown, unknown)",
            "<unknown object java.lang.Object> <unknown object java.lang.Object>");
  }

  @Test
  public void testPercOnObjectInvalidFormat() throws Exception {
    newTest()
        .update("obj", createObjWithStr())
        .testIfExactError("invalid argument <str marker> for format pattern %d", "'%d' % obj");
  }

  @Test
  public void testDictKeys() throws Exception {
    newTest().testExactOrder("v = {'a': 1}.keys() + ['b', 'c'] ; v", "a", "b", "c");
  }

  @Test
  public void testDictKeysTooManyArgs() throws Exception {
    newTest().testIfExactError(
        "expected no more than 0 positional arguments, but got 1, "
            + "in method call keys(string) of 'dict'", "{'a': 1}.keys('abc')");
  }

  @Test
  public void testDictKeysTooManyKeyArgs() throws Exception {
    newTest().testIfExactError(
        "unexpected keyword 'arg', in method call keys(string arg) of 'dict'",
        "{'a': 1}.keys(arg='abc')");
  }

  @Test
  public void testDictKeysDuplicateKeyArgs() throws Exception {
    newTest().testIfExactError("duplicate keywords 'arg', 'k' in call to {\"a\": 1}.keys",
        "{'a': 1}.keys(arg='abc', arg='def', k=1, k=2)");
  }

  @Test
  public void testArgBothPosKey() throws Exception {
    newTest().testIfErrorContains(
        "got multiple values for keyword argument 'old', "
            + "in method call replace(string, string, int, string old) of 'string'",
        "'banana'.replace('a', 'o', 3, old='a')");
  }
}
