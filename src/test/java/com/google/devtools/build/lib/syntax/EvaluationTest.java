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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test of evaluation behavior. (Implicitly uses lexer + parser.) */
// TODO(adonovan): separate tests of parser, resolver, Starlark core evaluator,
// and BUILD and .bzl features.
@RunWith(JUnit4.class)
public final class EvaluationTest extends EvaluationTestCase {

  @Test
  public void testExecutionStopsAtFirstError() throws Exception {
    List<String> printEvents = new ArrayList<>();
    StarlarkThread thread =
        createStarlarkThread(/*printHandler=*/ (_thread, msg) -> printEvents.add(msg));
    ParserInput input = ParserInput.fromLines("print('hello'); x = 1//0; print('goodbye')");

    Module module = thread.getGlobals();
    assertThrows(
        EvalException.class, () -> EvalUtils.exec(input, FileOptions.DEFAULT, module, thread));

    // Only expect hello, should have been an error before goodbye.
    assertThat(printEvents.toString()).isEqualTo("[hello]");
  }

  @Test
  public void testExecutionNotStartedOnInterrupt() throws Exception {
    StarlarkThread thread =
        createStarlarkThread(
            /*printHandler=*/ (_thread, msg) -> {
              throw new AssertionError("print statement was reached");
            });
    ParserInput input = ParserInput.fromLines("print('hello');");
    Module module = thread.getGlobals();

    try {
      Thread.currentThread().interrupt();
      assertThrows(
          InterruptedException.class,
          () -> EvalUtils.exec(input, FileOptions.DEFAULT, module, thread));
    } finally {
      // Reset interrupt bit in case the test failed to do so.
      Thread.interrupted();
    }
  }

  @Test
  public void testForLoopAbortedOnInterrupt() throws Exception {
    StarlarkThread thread = createStarlarkThread((th, msg) -> {});
    InterruptFunction interruptFunction = new InterruptFunction();
    Module module = thread.getGlobals();
    module.put("interrupt", interruptFunction);

    ParserInput input =
        ParserInput.fromLines(
            "def foo():", // Can't declare for loops at top level, so wrap with a function.
            "  for i in range(100):",
            "    interrupt(i == 5)",
            "foo()");

    try {
      assertThrows(
          InterruptedException.class,
          () -> EvalUtils.exec(input, FileOptions.DEFAULT, module, thread));
    } finally {
      // Reset interrupt bit in case the test failed to do so.
      Thread.interrupted();
    }

    assertThat(interruptFunction.callCount).isEqualTo(6);
  }

  @Test
  public void testForComprehensionAbortedOnInterrupt() throws Exception {
    StarlarkThread thread = createStarlarkThread((th, msg) -> {});
    Module module = thread.getGlobals();
    InterruptFunction interruptFunction = new InterruptFunction();
    module.put("interrupt", interruptFunction);

    ParserInput input = ParserInput.fromLines("[interrupt(i == 5) for i in range(100)]");

    try {
      assertThrows(
          InterruptedException.class,
          () -> EvalUtils.exec(input, FileOptions.DEFAULT, module, thread));
    } finally {
      // Reset interrupt bit in case the test failed to do so.
      Thread.interrupted();
    }

    assertThat(interruptFunction.callCount).isEqualTo(6);
  }

  @Test
  public void testFunctionCallsNotStartedOnInterrupt() throws Exception {
    StarlarkThread thread = createStarlarkThread((th, msg) -> {});
    Module module = thread.getGlobals();
    InterruptFunction interruptFunction = new InterruptFunction();
    module.put("interrupt", interruptFunction);

    ParserInput input =
        ParserInput.fromLines("interrupt(False); interrupt(True); interrupt(False);");

    try {
      assertThrows(
          InterruptedException.class,
          () -> EvalUtils.exec(input, FileOptions.DEFAULT, module, thread));
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
    public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
      callCount++;
      if (positional.length > 0 && Starlark.truth(positional[0])) {
        Thread.currentThread().interrupt();
      }
      return Starlark.NONE;
    }
  }

  private static StarlarkThread createStarlarkThread(StarlarkThread.PrintHandler printHandler) {
    Mutability mu = Mutability.create("test");
    StarlarkThread thread =
        StarlarkThread.builder(mu)
            .useDefaultSemantics()
            // Provide the UNIVERSE for print... this should not be necessary
            .setGlobals(Module.createForBuiltins(Starlark.UNIVERSE))
            .build();
    thread.setPrintHandler(printHandler);
    return thread;
  }

  @Test
  public void testExprs() throws Exception {
    new Scenario()
        .testExpression("'%sx' % 'foo' + 'bar1'", "fooxbar1")
        .testExpression("('%sx' % 'foo') + 'bar2'", "fooxbar2")
        .testExpression("'%sx' % ('foo' + 'bar3')", "foobar3x")
        .testExpression("123 + 456", 579)
        .testExpression("456 - 123", 333)
        .testExpression("8 % 3", 2)
        .testIfErrorContains("unsupported binary operation: int % string", "3 % 'foo'")
        .testExpression("-5", -5)
        .testIfErrorContains("unsupported unary operation: -string", "-'foo'");
  }

  @Test
  public void testListExprs() throws Exception {
    new Scenario().testExactOrder("[1, 2, 3]", 1, 2, 3).testExactOrder("(1, 2, 3)", 1, 2, 3);
  }

  @Test
  public void testStringFormatMultipleArgs() throws Exception {
    new Scenario().testExpression("'%sY%s' % ('X', 'Z')", "XYZ");
  }

  @Test
  public void testConditionalExpressions() throws Exception {
    new Scenario()
        .testExpression("1 if True else 2", 1)
        .testExpression("1 if False else 2", 2)
        .testExpression("1 + 2 if 3 + 4 else 5 + 6", 3);
  }

  @Test
  public void testListComparison() throws Exception {
    new Scenario()
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
    new Scenario().testIfExactError("Cannot compare depsets", "depset([1, 2]) < depset([3, 4])");
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
          public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
            int sum = 0;
            for (Object arg : positional) {
              sum += (Integer) arg;
            }
            return sum;
          }
        };

    new Scenario()
        .update(sum.getName(), sum)
        .testExpression("sum(1, 2, 3, 4, 5, 6)", 21)
        .testExpression("sum", sum)
        .testExpression("sum(a=1, b=2)", 0);
  }

  @Test
  public void testNotCallInt() throws Exception {
    new Scenario()
        .setUp("sum = 123456")
        .testLookup("sum", 123456)
        .testIfExactError("'int' object is not callable", "sum(1, 2, 3, 4, 5, 6)")
        .testExpression("sum", 123456);
  }

  @Test
  public void testComplexFunctionCall() throws Exception {
    new Scenario()
        .setUp("functions = [min, max]", "l = [1,2]")
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
              Tuple<Object> args,
              Dict<String, Object> kwargs) {
            return kwargs;
          }
        };

    new Scenario()
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
    new Scenario()
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
    new Scenario()
        .testExpression("6 * 7", 42)
        .testExpression("3 * 'ab'", "ababab")
        .testExpression("0 * 'ab'", "")
        .testExpression("'1' + '0' * 5", "100000")
        .testExpression("'ab' * -4", "")
        .testExpression("-1 * ''", "");
  }

  @Test
  public void testSlashOperatorIsForbidden() throws Exception {
    new Scenario().testIfErrorContains("The `/` operator is not allowed.", "5 / 2");
  }

  @Test
  public void testFloorDivision() throws Exception {
    new Scenario()
        .testExpression("6 // 2", 3)
        .testExpression("6 // 4", 1)
        .testExpression("3 // 6", 0)
        .testExpression("7 // -2", -4)
        .testExpression("-7 // 2", -4)
        .testExpression("-7 // -2", 3)
        .testExpression("2147483647 // 2", 1073741823)
        .testIfErrorContains("unsupported binary operation: string // int", "'str' // 2")
        .testIfExactError("integer division by zero", "5 // 0");
  }

  @Test
  public void testCheckedArithmetic() throws Exception {
    new Scenario()
        .testIfErrorContains("integer overflow", "2000000000 + 2000000000")
        .testIfErrorContains("integer overflow", "1234567890 * 987654321")
        .testIfErrorContains("integer overflow", "- 2000000000 - 2000000000")

        // literal 2147483648 is not allowed, so we compute it
        .setUp("minint = - 2147483647 - 1")
        .testIfErrorContains("integer overflow", "-minint");
  }

  @Test
  public void testOperatorPrecedence() throws Exception {
    new Scenario()
        .testExpression("2 + 3 * 4", 14)
        .testExpression("2 + 3 // 4", 2)
        .testExpression("2 * 3 + 4 // -2", 4);
  }

  @Test
  public void testConcatStrings() throws Exception {
    new Scenario().testExpression("'foo' + 'bar'", "foobar");
  }

  @Test
  public void testConcatLists() throws Exception {
    new Scenario()
        .testExactOrder("[1,2] + [3,4]", 1, 2, 3, 4)
        .testExactOrder("(1,2)", 1, 2)
        .testExactOrder("(1,2) + (3,4)", 1, 2, 3, 4);

    // TODO(fwe): cannot be handled by current testing suite
    // list
    Object x = eval("[1,2] + [3,4]");
    assertThat((Iterable<?>) x).containsExactly(1, 2, 3, 4).inOrder();
    assertThat(x).isInstanceOf(StarlarkList.class);
    assertThat(EvalUtils.isImmutable(x)).isFalse();

    // tuple
    x = eval("(1,2) + (3,4)");
    assertThat((Iterable<?>) x).containsExactly(1, 2, 3, 4).inOrder();
    assertThat(x).isInstanceOf(Tuple.class);
    assertThat(x).isEqualTo(Tuple.of(1, 2, 3, 4));
    assertThat(EvalUtils.isImmutable(x)).isTrue();

    checkEvalError("unsupported binary operation: tuple + list", "(1,2) + [3,4]");
  }

  @Test
  public void testListComprehensions() throws Exception {
    new Scenario()
        .testExactOrder("['foo/%s.java' % x for x in []]")
        .testExactOrder(
            "['foo/%s.java' % y for y in ['bar', 'wiz', 'quux']]",
            "foo/bar.java", "foo/wiz.java", "foo/quux.java")
        .testExactOrder(
            "['%s/%s.java' % (z, t) for z in ['foo', 'bar'] " + "for t in ['baz', 'wiz', 'quux']]",
            "foo/baz.java",
            "foo/wiz.java",
            "foo/quux.java",
            "bar/baz.java",
            "bar/wiz.java",
            "bar/quux.java")
        .testExactOrder(
            "['%s/%s.java' % (b, b) for a in ['foo', 'bar'] " + "for b in ['baz', 'wiz', 'quux']]",
            "baz/baz.java",
            "wiz/wiz.java",
            "quux/quux.java",
            "baz/baz.java",
            "wiz/wiz.java",
            "quux/quux.java")
        .testExactOrder(
            "['%s/%s.%s' % (c, d, e) for c in ['foo', 'bar'] "
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
    new Scenario()
        .setUp("li = [[1, 2], [3, 4]]")
        .testExactOrder("[j for i in li for j in i]", 1, 2, 3, 4);
    new Scenario()
        .setUp("input = [['abc'], ['def', 'ghi']]\n")
        .testExactOrder(
            "['%s %s' % (b, c) for a in input for b in a for c in b.elems()]",
            "abc a", "abc b", "abc c", "def d", "def e", "def f", "ghi g", "ghi h", "ghi i");
  }

  @Test
  public void testListComprehensionsMultipleVariables() throws Exception {
    new Scenario()
        .testEval("[x + y for x, y in [(1, 2), (3, 4)]]", "[3, 7]")
        .testEval("[z + t for (z, t) in [[1, 2], [3, 4]]]", "[3, 7]");
  }

  @Test
  public void testSequenceAssignment() throws Exception {
    // assignment to empty list/tuple
    // See https://github.com/bazelbuild/starlark/issues/93 for discussion
    checkEvalError(
        "can't assign to ()", //
        "() = ()");
    checkEvalError(
        "can't assign to ()", //
        "() = 1");
    checkEvalError(
        "can't assign to []", //
        "[] = ()");

    // RHS not iterable
    checkEvalError(
        "got 'int' in sequence assignment", //
        "x, y = 1");
    checkEvalError(
        "got 'int' in sequence assignment", //
        "(x,) = 1");
    checkEvalError(
        "got 'int' in sequence assignment", //
        "[x] = 1");

    // too few
    checkEvalError(
        "too few values to unpack (got 0, want 2)", //
        "x, y = ()");
    checkEvalError(
        "too few values to unpack (got 0, want 2)", //
        "[x, y] = ()");

    // just right
    exec("x, y = 1, 2");
    exec("[x, y] = 1, 2");
    exec("(x,) = [1]");

    // too many
    checkEvalError(
        "too many values to unpack (got 3, want 2)", //
        "x, y = 1, 2, 3");
    checkEvalError(
        "too many values to unpack (got 3, want 2)", //
        "[x, y] = 1, 2, 3");
  }

  @Test
  public void testListComprehensionsMultipleVariablesFail() throws Exception {
    new Scenario()
        .testIfErrorContains(
            "too few values to unpack (got 2, want 3)", //
            "[x + y for x, y, z in [(1, 2), (3, 4)]]")
        .testIfExactError(
            "got 'int' in sequence assignment", //
            "[x + y for x, y in (1, 2)]");

    new Scenario()
        .testIfErrorContains(
            "too few values to unpack (got 2, want 3)", //
            "def foo (): return [x + y for x, y, z in [(1, 2), (3, 4)]]",
            "foo()");

    new Scenario()
        .testIfErrorContains(
            "got 'int' in sequence assignment", //
            "def bar (): return [x + y for x, y in (1, 2)]",
            "bar()");

    new Scenario()
        .testIfErrorContains(
            "too few values to unpack (got 2, want 3)", //
            "[x + y for x, y, z in [(1, 2), (3, 4)]]");

    new Scenario()
        .testIfErrorContains(
            "got 'int' in sequence assignment", //
            "[x2 + y2 for x2, y2 in (1, 2)]");

    new Scenario()
        // Behavior varies across Python2 and 3 and Starlark in {Go,Java}.
        // See https://github.com/bazelbuild/starlark/issues/93 for discussion.
        .testIfErrorContains(
            "can't assign to []", //
            "[2 for [] in [()]]");
  }

  @Test
  public void testListComprehensionsWithFiltering() throws Exception {
    new Scenario()
        .setUp("range3 = [0, 1, 2]")
        .testEval("[a for a in (4, None, 2, None, 1) if a != None]", "[4, 2, 1]")
        .testEval("[b+c for b in [0, 1, 2] for c in [0, 1, 2] if b + c > 2]", "[3, 3, 4]")
        .testEval("[d+e for d in range3 if d % 2 == 1 for e in range3]", "[1, 2, 3]")
        .testEval(
            "[[f,g] for f in [0, 1, 2, 3, 4] if f for g in [5, 6, 7, 8] if f * g % 12 == 0 ]",
            "[[2, 6], [3, 8], [4, 6]]")
        .testEval("[h for h in [4, 2, 0, 1] if h]", "[4, 2, 1]");
  }

  @Test
  public void testListComprehensionDefinitionOrder() throws Exception {
    // This exercises the .bzl file behavior. This is a dynamic error.
    // (The error message for BUILD files is slightly different (no "local")
    // because it doesn't record the scope in the syntax tree.)
    new Scenario()
        .testIfErrorContains(
            "local variable 'y' is referenced before assignment", //
            "[x for x in (1, 2) if y for y in (3, 4)]");

    // This is the corresponding test for BUILD files.
    EvalException ex =
        assertThrows(
            EvalException.class, () -> execBUILD("[x for x in (1, 2) if y for y in (3, 4)]"));
    assertThat(ex).hasMessageThat().isEqualTo("variable 'y' is referenced before assignment");
  }

  private static void execBUILD(String... lines)
      throws SyntaxError.Exception, EvalException, InterruptedException {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkThread thread =
        StarlarkThread.builder(Mutability.create("test")).useDefaultSemantics().build();
    Module module = thread.getGlobals();
    FileOptions options = FileOptions.builder().recordScope(false).build();
    EvalUtils.exec(input, options, module, thread);
  }

  @Test
  public void testTupleDestructuring() throws Exception {
    new Scenario()
        .setUp("a, b = 1, 2")
        .testLookup("a", 1)
        .testLookup("b", 2)
        .setUp("c, d = {'key1':2, 'key2':3}")
        .testLookup("c", "key1")
        .testLookup("d", "key2");
  }

  @Test
  public void testSingleTuple() throws Exception {
    new Scenario().setUp("(a,) = [1]").testLookup("a", 1);
  }

  @Test
  public void testHeterogeneousDict() throws Exception {
    new Scenario()
        .setUp("d = {'str': 1, 2: 3}", "a = d['str']", "b = d[2]")
        .testLookup("a", 1)
        .testLookup("b", 3);
  }

  @Test
  public void testAccessDictWithATupleKey() throws Exception {
    new Scenario().setUp("x = {(1, 2): 3}[1, 2]").testLookup("x", 3);
  }

  @Test
  public void testDictWithDuplicatedKey() throws Exception {
    new Scenario()
        .testIfErrorContains(
            "Duplicated key \"str\" when creating dictionary", "{'str': 1, 'x': 2, 'str': 3}");
  }

  @Test
  public void testRecursiveTupleDestructuring() throws Exception {
    new Scenario()
        .setUp("((a, b), (c, d)) = [(1, 2), (3, 4)]")
        .testLookup("a", 1)
        .testLookup("b", 2)
        .testLookup("c", 3)
        .testLookup("d", 4);
  }

  @Test
  public void testListComprehensionAtTopLevel() throws Exception {
    // It is allowed to have a loop variable with the same name as a global variable.
    new Scenario()
        .update("x", 42)
        .setUp("y = [x + 1 for x in [1,2,3]]")
        .testExactOrder("y", 2, 3, 4);
  }

  @Test
  public void testDictComprehensions() throws Exception {
    new Scenario()
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
    new Scenario().testIfExactError("type 'int' is not iterable", "{k : k for k in 3}");
  }

  @Test
  public void testDictComprehension_ManyClauses() throws Exception {
    new Scenario()
        .testExpression(
            "{x : x * y for x in range(1, 10) if x % 2 == 0 for y in range(1, 10) if y == x}",
            ImmutableMap.of(2, 4, 4, 16, 6, 36, 8, 64));
  }

  @Test
  public void testDictComprehensions_MultipleKey() throws Exception {
    new Scenario()
        .testExpression("{x : x for x in [1, 2, 1]}", ImmutableMap.of(1, 1, 2, 2))
        .testExpression(
            "{y : y for y in ['ab', 'c', 'a' + 'b']}", ImmutableMap.of("ab", "ab", "c", "c"));
  }

  @Test
  public void testListConcatenation() throws Exception {
    new Scenario()
        .testExpression("[1, 2] + [3, 4]", StarlarkList.of(null, 1, 2, 3, 4))
        .testExpression("(1, 2) + (3, 4)", Tuple.of(1, 2, 3, 4))
        .testIfExactError("unsupported binary operation: list + tuple", "[1, 2] + (3, 4)")
        .testIfExactError("unsupported binary operation: tuple + list", "(1, 2) + [3, 4]");
  }

  @Test
  public void testListMultiply() throws Exception {
    Mutability mu = Mutability.create("test");
    new Scenario()
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
    new Scenario()
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

  @Test
  public void testListComprehensionFailsOnNonSequence() throws Exception {
    new Scenario().testIfErrorContains("type 'int' is not iterable", "[x + 1 for x in 123]");
  }

  @Test
  public void testListComprehensionOnStringIsForbidden() throws Exception {
    new Scenario().testIfErrorContains("type 'string' is not iterable", "[x for x in 'abc']");
  }

  @Test
  public void testInvalidAssignment() throws Exception {
    new Scenario().testIfErrorContains("cannot assign to 'x + 1'", "x + 1 = 2");
  }

  @Test
  public void testListComprehensionOnDictionary() throws Exception {
    new Scenario().testExactOrder("['var_' + n for n in {'a':1,'b':2}]", "var_a", "var_b");
  }

  @Test
  public void testListComprehensionOnDictionaryCompositeExpression() throws Exception {
    new Scenario()
        .setUp("d = {1:'a',2:'b'}", "l = [d[x] for x in d]")
        .testLookup("l", StarlarkList.of(null, "a", "b"));
  }

  @Test
  public void testListComprehensionUpdate() throws Exception {
    new Scenario()
        .setUp("xs = [1, 2, 3]")
        .testIfErrorContains(
            "list value is temporarily immutable due to active for-loop iteration",
            "[xs.append(4) for x in xs]");
  }

  @Test
  public void testNestedListComprehensionUpdate() throws Exception {
    new Scenario()
        .setUp("xs = [1, 2, 3]")
        .testIfErrorContains(
            "list value is temporarily immutable due to active for-loop iteration",
            "[xs.append(4) for x in xs for y in xs]");
  }

  @Test
  public void testListComprehensionUpdateInClause() throws Exception {
    new Scenario()
        .setUp("xs = [1, 2, 3]")
        .testIfErrorContains(
            "list value is temporarily immutable due to active for-loop iteration",
            // Use short-circuiting to produce valid output in the event
            // the exception is not raised.
            "[y for x in xs for y in (xs.append(4) or xs)]");
  }

  @Test
  public void testDictComprehensionUpdate() throws Exception {
    new Scenario()
        .setUp("xs = {1:1, 2:2, 3:3}")
        .testIfErrorContains(
            "dict value is temporarily immutable due to active for-loop iteration",
            "[xs.popitem() for x in xs]");
  }

  @Test
  public void testListComprehensionScope() throws Exception {
    // Test list comprehension creates a scope, so outer variables kept unchanged
    new Scenario()
        .setUp("x = 1", "l = [x * 3 for x in [2]]", "y = x")
        .testEval("y", "1")
        .testEval("l", "[6]");
  }

  @Test
  public void testInOperator() throws Exception {
    new Scenario()
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
    new Scenario()
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
    new Scenario()
        .testIfErrorContains(
            "'in <string>' requires string as left operand, not 'int'", "1 in '123'")
        .testIfErrorContains("unsupported binary operation: string in int", "'a' in 1");
  }

  @Test
  public void testInCompositeForPrecedence() throws Exception {
    new Scenario().testExpression("not 'a' in ['a'] or 0", 0);
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
    new Scenario().update("obj", createObjWithStr()).testExpression("'%s' % obj", "<str marker>");
    new Scenario()
        .update("unknown", new Dummy())
        .testExpression(
            "'%s' % unknown",
            "<unknown object com.google.devtools.build.lib.syntax.EvaluationTest$Dummy>");
  }

  @Test
  public void testPercentOnTupleOfDummyValues() throws Exception {
    new Scenario()
        .update("obj", createObjWithStr())
        .testExpression("'%s %s' % (obj, obj)", "<str marker> <str marker>");
    new Scenario()
        .update("unknown", new Dummy())
        .testExpression(
            "'%s %s' % (unknown, unknown)",
            "<unknown object com.google.devtools.build.lib.syntax.EvaluationTest$Dummy> <unknown"
                + " object com.google.devtools.build.lib.syntax.EvaluationTest$Dummy>");
  }

  @Test
  public void testPercOnObjectInvalidFormat() throws Exception {
    new Scenario()
        .update("obj", createObjWithStr())
        .testIfExactError("invalid argument <str marker> for format pattern %d", "'%d' % obj");
  }

  @Test
  public void testDictKeys() throws Exception {
    new Scenario().testExactOrder("{'a': 1}.keys() + ['b', 'c']", "a", "b", "c");
  }

  @Test
  public void testDictKeysTooManyArgs() throws Exception {
    new Scenario()
        .testIfExactError("keys() got unexpected positional argument", "{'a': 1}.keys('abc')");
  }

  @Test
  public void testDictKeysTooManyKeyArgs() throws Exception {
    new Scenario()
        .testIfExactError(
            "keys() got unexpected keyword argument 'arg'", "{'a': 1}.keys(arg='abc')");
  }

  @Test
  public void testDictKeysDuplicateKeyArgs() throws Exception {
    // TODO(adonovan): when the duplication is literal, this should be caught by a static check.
    new Scenario()
        .testIfExactError(
            "int() got multiple values for argument 'base'", "int('1', base=10, base=16)");
    new Scenario()
        .testIfExactError(
            "int() got multiple values for argument 'base'", "int('1', base=10, **dict(base=16))");
  }

  @Test
  public void testArgBothPosKey() throws Exception {
    new Scenario()
        .testIfErrorContains(
            "int() got multiple values for argument 'base'", "int('2', 3, base=3)");
  }

  @Test
  public void testStaticNameResolution() throws Exception {
    new Scenario().testIfErrorContains("name 'foo' is not defined", "[foo for x in []]");
  }

  @Test
  public void testExec() throws Exception {
    StarlarkThread thread =
        StarlarkThread.builder(Mutability.create("test")).useDefaultSemantics().build();
    Module module = thread.getGlobals();
    EvalUtils.exec(
        ParserInput.fromLines(
            "# a file in the build language",
            "",
            "x = [1, 2, 'foo', 4] + [1, 2, \"%s%d\" % ('foo', 1)]"),
        FileOptions.DEFAULT,
        module,
        thread);
    assertThat(thread.getGlobals().lookup("x"))
        .isEqualTo(StarlarkList.of(/*mutability=*/ null, 1, 2, "foo", 4, 1, 2, "foo1"));
  }
}
