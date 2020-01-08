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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestMode;
import java.util.Collections;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test of evaluation behavior. (Implicitly uses lexer + parser.) */
// TODO(adonovan): separate tests of parser, resolver, Starlark core evaluator,
// and BUILD and .bzl features.
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
  public void testExecutionStopsAtFirstError() throws Exception {
    EventCollector printEvents = new EventCollector();
    StarlarkThread thread =
        createStarlarkThread(mutability, StarlarkThread.makeDebugPrintHandler(printEvents));
    ParserInput input = ParserInput.fromLines("print('hello'); x = 1//0; print('goodbye')");

    assertThrows(EvalException.class, () -> EvalUtils.exec(input, thread));

    // Only expect hello, should have been an error before goodbye.
    assertThat(printEvents).hasSize(1);
    assertThat(printEvents.iterator().next().getMessage()).isEqualTo("hello");
  }

  @Test
  public void testExecutionNotStartedOnInterrupt() throws Exception {
    EventCollector printEvents = new EventCollector();
    StarlarkThread thread =
        createStarlarkThread(mutability, StarlarkThread.makeDebugPrintHandler(printEvents));
    ParserInput input = ParserInput.fromLines("print('hello');");

    try {
      Thread.currentThread().interrupt();
      assertThrows(InterruptedException.class, () -> EvalUtils.exec(input, thread));
    } finally {
      // Reset interrupt bit in case the test failed to do so.
      Thread.interrupted();
    }

    assertThat(printEvents).isEmpty();
  }

  @Test
  public void testForLoopAbortedOnInterrupt() throws Exception {
    StarlarkThread thread = createStarlarkThread(mutability, (th, msg) -> {});
    InterruptFunction interruptFunction = new InterruptFunction();
    thread.getGlobals().put("interrupt", interruptFunction);

    ParserInput input =
        ParserInput.fromLines(
            "def foo():", // Can't declare for loops at top level, so wrap with a function.
            "  for i in range(100):",
            "    interrupt(i == 5)",
            "foo()");

    try {
      assertThrows(InterruptedException.class, () -> EvalUtils.exec(input, thread));
    } finally {
      // Reset interrupt bit in case the test failed to do so.
      Thread.interrupted();
    }

    assertThat(interruptFunction.callCount).isEqualTo(6);
  }

  @Test
  public void testForComprehensionAbortedOnInterrupt() throws Exception {
    StarlarkThread thread = createStarlarkThread(mutability, (th, msg) -> {});
    InterruptFunction interruptFunction = new InterruptFunction();
    thread.getGlobals().put("interrupt", interruptFunction);

    ParserInput input = ParserInput.fromLines("[interrupt(i == 5) for i in range(100)]");

    try {
      assertThrows(InterruptedException.class, () -> EvalUtils.exec(input, thread));
    } finally {
      // Reset interrupt bit in case the test failed to do so.
      Thread.interrupted();
    }

    assertThat(interruptFunction.callCount).isEqualTo(6);
  }

  @Test
  public void testFunctionCallsNotStartedOnInterrupt() throws Exception {
    StarlarkThread thread = createStarlarkThread(mutability, (th, msg) -> {});
    InterruptFunction interruptFunction = new InterruptFunction();
    thread.getGlobals().put("interrupt", interruptFunction);

    ParserInput input =
        ParserInput.fromLines("interrupt(False); interrupt(True); interrupt(False);");

    try {
      assertThrows(InterruptedException.class, () -> EvalUtils.exec(input, thread));
    } finally {
      // Reset interrupt bit in case the test failed to do so.
      Thread.interrupted();
    }

    // Third call shouldn't happen.
    assertThat(interruptFunction.callCount).isEqualTo(2);
  }

  private static class InterruptFunction implements StarlarkCallable {

    private int callCount = 0;

    @Override
    public String getName() {
      return "interrupt";
    }

    @Override
    public Object fastcall(
        StarlarkThread thread, Location loc, Object[] positional, Object[] named) {
      callCount++;
      if (positional.length > 0 && Starlark.truth(positional[0])) {
        Thread.currentThread().interrupt();
      }
      return Starlark.NONE;
    }
  }

  private static StarlarkThread createStarlarkThread(
      Mutability mutability, StarlarkThread.PrintHandler printHandler) {
    StarlarkThread thread =
        StarlarkThread.builder(mutability)
            .useDefaultSemantics()
            // Provide the UNIVERSE for print... this should not be necessary
            .setGlobals(Module.createForBuiltins(Starlark.UNIVERSE))
            .build();
    thread.setPrintHandler(printHandler);
    return thread;
  }

  @Test
  public void testExprs() throws Exception {
    newTest()
        .testExpression("'%sx' % 'foo' + 'bar1'", "fooxbar1")
        .testExpression("('%sx' % 'foo') + 'bar2'", "fooxbar2")
        .testExpression("'%sx' % ('foo' + 'bar3')", "foobar3x")
        .testExpression("123 + 456", 579)
        .testExpression("456 - 123", 333)
        .testExpression("8 % 3", 2)
        .testIfErrorContains("unsupported operand type(s) for %: 'int' and 'string'", "3 % 'foo'")
        .testExpression("-5", -5)
        .testIfErrorContains("unsupported unary operation: -string", "-'foo'");
  }

  @Test
  public void testListExprs() throws Exception {
    newTest().testExactOrder("[1, 2, 3]", 1, 2, 3).testExactOrder("(1, 2, 3)", 1, 2, 3);
  }

  @Test
  public void testStringFormatMultipleArgs() throws Exception {
    newTest().testExpression("'%sY%s' % ('X', 'Z')", "XYZ");
  }

  @Test
  public void testConditionalExpressions() throws Exception {
    newTest()
        .testExpression("1 if True else 2", 1)
        .testExpression("1 if False else 2", 2)
        .testExpression("1 + 2 if 3 + 4 else 5 + 6", 3);
  }

  @Test
  public void testListComparison() throws Exception {
    newTest()
        .testExpression("[] < [1]", true)
        .testExpression("[1] < [1, 1]", true)
        .testExpression("[1, 1] < [1, 2]", true)
        .testExpression("[1, 2] < [1, 2, 3]", true)
        .testExpression("[1, 2, 3] <= [1, 2, 3]", true)
        .testExpression("['a', 'b'] > ['a']", true)
        .testExpression("['a', 'b'] >= ['a']", true)
        .testExpression("['a', 'b'] < ['a']", false)
        .testExpression("['a', 'b'] <= ['a']", false)
        .testExpression("('a', 'b') > ('a', 'b')", false)
        .testExpression("('a', 'b') >= ('a', 'b')", true)
        .testExpression("('a', 'b') < ('a', 'b')", false)
        .testExpression("('a', 'b') <= ('a', 'b')", true)
        .testExpression("[[1, 1]] > [[1, 1], []]", false)
        .testExpression("[[1, 1]] < [[1, 1], []]", true);
  }

  @Test
  public void testSetComparison() throws Exception {
    newTest().testIfExactError("Cannot compare depsets", "depset([1, 2]) < depset([3, 4])");
  }

  @Test
  public void testSumFunction() throws Exception {
    StarlarkCallable sum =
        new StarlarkCallable() {
          @Override
          public String getName() {
            return "sum";
          }

          @Override
          public Object fastcall(
              StarlarkThread thread, Location loc, Object[] positional, Object[] named) {
            int sum = 0;
            for (Object arg : positional) {
              sum += (Integer) arg;
            }
            return sum;
          }
        };

    newTest()
        .update(sum.getName(), sum)
        .testExpression("sum(1, 2, 3, 4, 5, 6)", 21)
        .testExpression("sum", sum)
        .testExpression("sum(a=1, b=2)", 0);
  }

  @Test
  public void testNotCallInt() throws Exception {
    newTest()
        .setUp("sum = 123456")
        .testLookup("sum", 123456)
        .testIfExactError("'int' object is not callable", "sum(1, 2, 3, 4, 5, 6)")
        .testExpression("sum", 123456);
  }

  @Test
  public void testComplexFunctionCall() throws Exception {
    newTest().setUp("functions = [min, max]", "l = [1,2]")
        .testEval("(functions[0](l), functions[1](l))", "(1, 2)");
  }

  @Test
  public void testKeywordArgs() throws Exception {
    // This function returns the map of keyword arguments passed to it.
    StarlarkCallable kwargs =
        new StarlarkCallable() {
          @Override
          public String getName() {
            return "kwargs";
          }

          @Override
          public Object call(
              StarlarkThread thread,
              Location loc,
              Tuple<Object> args,
              Dict<String, Object> kwargs) {
            return kwargs;
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
        .testExpression("6 % 2", 0)
        .testExpression("6 % 4", 2)
        .testExpression("3 % 6", 3)
        .testExpression("7 % -4", -1)
        .testExpression("-7 % 4", 1)
        .testExpression("-7 % -4", -3)
        .testIfExactError("integer modulo by zero", "5 % 0");
  }

  @Test
  public void testMult() throws Exception {
    newTest()
        .testExpression("6 * 7", 42)
        .testExpression("3 * 'ab'", "ababab")
        .testExpression("0 * 'ab'", "")
        .testExpression("'1' + '0' * 5", "100000")
        .testExpression("'ab' * -4", "")
        .testExpression("-1 * ''", "");
  }

  @Test
  public void testSlashOperatorIsForbidden() throws Exception {
    newTest().testIfErrorContains("The `/` operator is not allowed.", "5 / 2");
  }

  @Test
  public void testFloorDivision() throws Exception {
    newTest()
        .testExpression("6 // 2", 3)
        .testExpression("6 // 4", 1)
        .testExpression("3 // 6", 0)
        .testExpression("7 // -2", -4)
        .testExpression("-7 // 2", -4)
        .testExpression("-7 // -2", 3)
        .testExpression("2147483647 // 2", 1073741823)
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
        .testExpression("2 + 3 * 4", 14)
        .testExpression("2 + 3 // 4", 2)
        .testExpression("2 * 3 + 4 // -2", 4);
  }

  @Test
  public void testConcatStrings() throws Exception {
    newTest().testExpression("'foo' + 'bar'", "foobar");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testConcatLists() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    // list
    Object x = eval("[1,2] + [3,4]");
    assertThat((Iterable<Object>) x).containsExactly(1, 2, 3, 4).inOrder();
    assertThat(x).isEqualTo(StarlarkList.of(thread.mutability(), 1, 2, 3, 4));
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
    newTest()
        .setUp("li = [[1, 2], [3, 4]]")
        .testExactOrder("[j for i in li for j in i]", 1, 2, 3, 4);
    newTest()
        .setUp("input = [['abc'], ['def', 'ghi']]\n")
        .testExactOrder(
            "['%s %s' % (b, c) for a in input for b in a for c in b.elems()]",
            "abc a", "abc b", "abc c", "def d", "def e", "def f", "ghi g", "ghi h", "ghi i");
  }

  @Test
  public void testListComprehensionsMultipleVariables() throws Exception {
    newTest().testEval("[x + y for x, y in [(1, 2), (3, 4)]]", "[3, 7]")
        .testEval("[z + t for (z, t) in [[1, 2], [3, 4]]]", "[3, 7]");
  }

  @Test
  public void testListComprehensionsMultipleVariablesFail() throws Exception {
    newTest()
        .testIfErrorContains(
            "assignment length mismatch: left-hand side has length 3, but right-hand side"
                + " evaluates to value of length 2",
            "[x + y for x, y, z in [(1, 2), (3, 4)]]")
        .testIfExactError("type 'int' is not iterable", "[x + y for x, y in (1, 2)]");
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
    new BuildTest()
        .testIfErrorContains("name 'y' is not defined", "[x for x in (1, 2) if y for y in (3, 4)]");

    new SkylarkTest()
        .testIfErrorContains(
            "local variable 'y' is referenced before assignment",
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
        .testExpression("{a : a for a in []}", Collections.emptyMap())
        .testExpression("{b : b for b in [1, 2]}", ImmutableMap.of(1, 1, 2, 2))
        .testExpression(
            "{c : 'v_' + c for c in ['a', 'b']}", ImmutableMap.of("a", "v_a", "b", "v_b"))
        .testExpression(
            "{'k_' + d : d for d in ['a', 'b']}", ImmutableMap.of("k_a", "a", "k_b", "b"))
        .testExpression(
            "{'k_' + e : 'v_' + e for e in ['a', 'b']}",
            ImmutableMap.of("k_a", "v_a", "k_b", "v_b"))
        .testExpression("{x+y : x*y for x, y in [[2, 3]]}", ImmutableMap.of(5, 6));
  }

  @Test
  public void testDictComprehensionOnNonIterable() throws Exception {
    newTest().testIfExactError("type 'int' is not iterable", "{k : k for k in 3}");
  }

  @Test
  public void testDictComprehension_ManyClauses() throws Exception {
    new SkylarkTest()
        .testExpression(
            "{x : x * y for x in range(1, 10) if x % 2 == 0 for y in range(1, 10) if y == x}",
            ImmutableMap.of(2, 4, 4, 16, 6, 36, 8, 64));
  }

  @Test
  public void testDictComprehensions_MultipleKey() throws Exception {
    newTest()
        .testExpression("{x : x for x in [1, 2, 1]}", ImmutableMap.of(1, 1, 2, 2))
        .testExpression(
            "{y : y for y in ['ab', 'c', 'a' + 'b']}", ImmutableMap.of("ab", "ab", "c", "c"));
  }

  @Test
  public void testListConcatenation() throws Exception {
    newTest()
        .testExpression("[1, 2] + [3, 4]", StarlarkList.of(thread.mutability(), 1, 2, 3, 4))
        .testExpression("(1, 2) + (3, 4)", Tuple.of(1, 2, 3, 4))
        .testIfExactError(
            "unsupported operand type(s) for +: 'list' and 'tuple'", "[1, 2] + (3, 4)")
        .testIfExactError(
            "unsupported operand type(s) for +: 'tuple' and 'list'", "(1, 2) + [3, 4]");
  }

  @Test
  public void testListMultiply() throws Exception {
    Mutability mu = thread.mutability();
    newTest()
        .testExpression("[1, 2, 3] * 1", StarlarkList.of(mu, 1, 2, 3))
        .testExpression("[1, 2] * 2", StarlarkList.of(mu, 1, 2, 1, 2))
        .testExpression("[1, 2] * 3", StarlarkList.of(mu, 1, 2, 1, 2, 1, 2))
        .testExpression("[1, 2] * 4", StarlarkList.of(mu, 1, 2, 1, 2, 1, 2, 1, 2))
        .testExpression("[8] * 5", StarlarkList.of(mu, 8, 8, 8, 8, 8))
        .testExpression("[    ] * 10", StarlarkList.empty())
        .testExpression("[1, 2] * 0", StarlarkList.empty())
        .testExpression("[1, 2] * -4", StarlarkList.empty())
        .testExpression("2 * [1, 2]", StarlarkList.of(mu, 1, 2, 1, 2))
        .testExpression("10 * []", StarlarkList.empty())
        .testExpression("0 * [1, 2]", StarlarkList.empty())
        .testExpression("-4 * [1, 2]", StarlarkList.empty());
  }

  @Test
  public void testTupleMultiply() throws Exception {
    newTest()
        .testExpression("(1, 2, 3) * 1", Tuple.of(1, 2, 3))
        .testExpression("(1, 2) * 2", Tuple.of(1, 2, 1, 2))
        .testExpression("(1, 2) * 3", Tuple.of(1, 2, 1, 2, 1, 2))
        .testExpression("(1, 2) * 4", Tuple.of(1, 2, 1, 2, 1, 2, 1, 2))
        .testExpression("(8,) * 5", Tuple.of(8, 8, 8, 8, 8))
        .testExpression("(    ) * 10", Tuple.empty())
        .testExpression("(1, 2) * 0", Tuple.empty())
        .testExpression("(1, 2) * -4", Tuple.empty())
        .testExpression("2 * (1, 2)", Tuple.of(1, 2, 1, 2))
        .testExpression("10 * ()", Tuple.empty())
        .testExpression("0 * (1, 2)", Tuple.empty())
        .testExpression("-4 * (1, 2)", Tuple.empty());
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
  public void testListComprehensionOnStringIsForbidden() throws Exception {
    newTest().testIfErrorContains("type 'string' is not iterable", "[x for x in 'abc']");
  }

  @Test
  public void testInvalidAssignment() throws Exception {
    newTest().testIfErrorContains(
        "cannot assign to 'x + 1'", "x + 1 = 2");
  }

  @Test
  public void testListComprehensionOnDictionary() throws Exception {
    newTest().testExactOrder("['var_' + n for n in {'a':1,'b':2}]", "var_a", "var_b");
  }

  @Test
  public void testListComprehensionOnDictionaryCompositeExpression() throws Exception {
    new BuildTest()
        .setUp("d = {1:'a',2:'b'}", "l = [d[x] for x in d]")
        .testLookup("l", StarlarkList.of(thread.mutability(), "a", "b"));
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
  public void testListComprehensionScope() throws Exception {
    // Test list comprehension creates a scope, so outer variables kept unchanged
    new BuildTest()
        .setUp("x = 1", "l = [x * 3 for x in [2]]", "y = x")
        .testEval("y", "1")
        .testEval("l", "[6]");
  }

  @Test
  public void testInOperator() throws Exception {
    newTest()
        .testExpression("'b' in ['a', 'b']", Boolean.TRUE)
        .testExpression("'c' in ['a', 'b']", Boolean.FALSE)
        .testExpression("'b' in ('a', 'b')", Boolean.TRUE)
        .testExpression("'c' in ('a', 'b')", Boolean.FALSE)
        .testExpression("'b' in {'a' : 1, 'b' : 2}", Boolean.TRUE)
        .testExpression("'c' in {'a' : 1, 'b' : 2}", Boolean.FALSE)
        .testExpression("1 in {'a' : 1, 'b' : 2}", Boolean.FALSE)
        .testExpression("'b' in 'abc'", Boolean.TRUE)
        .testExpression("'d' in 'abc'", Boolean.FALSE);
  }

  @Test
  public void testNotInOperator() throws Exception {
    newTest()
        .testExpression("'b' not in ['a', 'b']", Boolean.FALSE)
        .testExpression("'c' not in ['a', 'b']", Boolean.TRUE)
        .testExpression("'b' not in ('a', 'b')", Boolean.FALSE)
        .testExpression("'c' not in ('a', 'b')", Boolean.TRUE)
        .testExpression("'b' not in {'a' : 1, 'b' : 2}", Boolean.FALSE)
        .testExpression("'c' not in {'a' : 1, 'b' : 2}", Boolean.TRUE)
        .testExpression("1 not in {'a' : 1, 'b' : 2}", Boolean.TRUE)
        .testExpression("'b' not in 'abc'", Boolean.FALSE)
        .testExpression("'d' not in 'abc'", Boolean.TRUE);
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
    newTest().testExpression("not 'a' in ['a'] or 0", 0);
  }

  private StarlarkValue createObjWithStr() {
    return new StarlarkValue() {
      @Override
      public void repr(Printer printer) {
        printer.append("<str marker>");
      }
    };
  }

  private static class Dummy implements StarlarkValue {}

  @Test
  public void testPercentOnDummyValue() throws Exception {
    newTest().update("obj", createObjWithStr()).testExpression("'%s' % obj", "<str marker>");
    newTest()
        .update("unknown", new Dummy())
        .testExpression(
            "'%s' % unknown",
            "<unknown object com.google.devtools.build.lib.syntax.EvaluationTest$Dummy>");
  }

  @Test
  public void testPercentOnTupleOfDummyValues() throws Exception {
    newTest()
        .update("obj", createObjWithStr())
        .testExpression("'%s %s' % (obj, obj)", "<str marker> <str marker>");
    newTest()
        .update("unknown", new Dummy())
        .testExpression(
            "'%s %s' % (unknown, unknown)",
            "<unknown object com.google.devtools.build.lib.syntax.EvaluationTest$Dummy> <unknown"
                + " object com.google.devtools.build.lib.syntax.EvaluationTest$Dummy>");
  }

  @Test
  public void testPercOnObjectInvalidFormat() throws Exception {
    newTest()
        .update("obj", createObjWithStr())
        .testIfExactError("invalid argument <str marker> for format pattern %d", "'%d' % obj");
  }

  @Test
  public void testDictKeys() throws Exception {
    newTest().testExactOrder("{'a': 1}.keys() + ['b', 'c']", "a", "b", "c");
  }

  @Test
  public void testDictKeysTooManyArgs() throws Exception {
    newTest().testIfExactError("keys() got unexpected positional argument", "{'a': 1}.keys('abc')");
  }

  @Test
  public void testDictKeysTooManyKeyArgs() throws Exception {
    newTest()
        .testIfExactError(
            "keys() got unexpected keyword argument 'arg'", "{'a': 1}.keys(arg='abc')");
  }

  @Test
  public void testDictKeysDuplicateKeyArgs() throws Exception {
    // TODO(adonovan): when the duplication is literal, this should be caught by a static check.
    newTest()
        .testIfExactError(
            "int() got multiple values for argument 'base'", "int('1', base=10, base=16)");
    new SkylarkTest()
        .testIfExactError(
            "int() got multiple values for argument 'base'", "int('1', base=10, **dict(base=16))");
  }

  @Test
  public void testArgBothPosKey() throws Exception {
    newTest()
        .testIfErrorContains(
            "int() got multiple values for argument 'base'", "int('2', 3, base=3)");
  }

  @Test
  public void testStaticNameResolution() throws Exception {
    newTest().testIfErrorContains("name 'foo' is not defined", "[foo for x in []]");
  }

  @Test
  public void testDefInBuild() throws Exception {
    new BuildTest()
        .testIfErrorContains(
            "function definitions are not allowed in BUILD files", "def func(): pass");
  }

  @Test
  public void testForStatementForbiddenInBuild() throws Exception {
    new BuildTest().testIfErrorContains("for loops are not allowed", "for _ in []: pass");
  }

  @Test
  public void testIfStatementForbiddenInBuild() throws Exception {
    new BuildTest().testIfErrorContains("if statements are not allowed", "if False: pass");
  }

  @Test
  public void testKwargsForbiddenInBuild() throws Exception {
    new BuildTest()
        .testIfErrorContains("**kwargs arguments are not allowed in BUILD files", "print(**dict)");

    new BuildTest()
        .testIfErrorContains(
            "**kwargs arguments are not allowed in BUILD files", "len(dict(**{'a': 1}))");
  }

  @Test
  public void testArgsForbiddenInBuild() throws Exception {
    new BuildTest()
        .testIfErrorContains("*args arguments are not allowed in BUILD files", "print(*['a'])");
  }
}
