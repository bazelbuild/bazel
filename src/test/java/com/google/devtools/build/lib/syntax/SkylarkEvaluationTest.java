// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.testutil.TestMode;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Evaluation tests with Skylark Environment.
 */
@RunWith(JUnit4.class)
public class SkylarkEvaluationTest extends EvaluationTest {

  @Before
  public final void setup() throws Exception {
    setMode(TestMode.SKYLARK);
  }

  /**
   * Creates an instance of {@code SkylarkTest} in order to run the tests from the base class in a
   * Skylark context
   */
  @Override
  protected ModalTestCase newTest() {
    return new SkylarkTest();
  }

  @Immutable
  static class Bad {
    Bad () {
    }
  }

  @SkylarkSignature(name = "foobar", returnType = String.class, documented = false)
  static BuiltinFunction foobar = new BuiltinFunction("foobar") {
    public String invoke() throws EvalException {
      return "foobar";
    }
  };

  @SkylarkModule(name = "Mock", doc = "")
  static class Mock {
    @SkylarkCallable(doc = "")
    public static Integer valueOf(String str) {
      return Integer.valueOf(str);
    }
    @SkylarkCallable(doc = "")
    public Boolean isEmpty(String str) {
      return str.isEmpty();
    }
    public void value() {}
    @SkylarkCallable(doc = "")
    public Bad returnBad() {
      return new Bad();
    }
    @SkylarkCallable(name = "struct_field", doc = "", structField = true)
    public String structField() {
      return "a";
    }
    @SkylarkCallable(name = "struct_field_callable", doc = "", structField = true)
    public BuiltinFunction structFieldCallable() {
      return foobar;
    }
    @SkylarkCallable(name = "function", doc = "", structField = false)
    public String function() {
      return "a";
    }
    @SuppressWarnings("unused")
    @SkylarkCallable(name = "nullfunc_failing", doc = "", allowReturnNones = false)
    public SkylarkValue nullfuncFailing(String p1, Integer p2) {
      return null;
    }
    @SkylarkCallable(name = "nullfunc_working", doc = "", allowReturnNones = true)
    public SkylarkValue nullfuncWorking() {
      return null;
    }
    @SkylarkCallable(name = "voidfunc", doc = "")
    public void voidfunc() {}
    @SkylarkCallable(name = "string_list", doc = "")
    public ImmutableList<String> stringList() {
      return ImmutableList.<String>of("a", "b");
    }
    @SkylarkCallable(name = "string", doc = "")
    public String string() {
      return "a";
    }

    @SkylarkCallable(
      name = "with_params",
      doc = "",
      mandatoryPositionals = 1,
      parameters = {
        @Param(name = "pos2", defaultValue = "False", type = Boolean.class),
        @Param(
          name = "posOrNamed",
          defaultValue = "False",
          type = Boolean.class,
          positional = true,
          named = true
        ),
        @Param(name = "named", type = Boolean.class, positional = false, named = true),
        @Param(
          name = "optionalNamed",
          type = Boolean.class,
          defaultValue = "False",
          positional = false,
          named = true
        ),
        @Param(
          name = "nonNoneable",
          type = Object.class,
          defaultValue = "\"a\"",
          positional = false,
          named = true
        ),
        @Param(
          name = "noneable",
          type = Integer.class,
          defaultValue = "None",
          noneable = true,
          positional = false,
          named = true
        ),
      }
    )
    public String withParams(
        Integer pos1,
        boolean pos2,
        boolean posOrNamed,
        boolean named,
        boolean optionalNamed,
        Object nonNoneable,
        Object noneable) {
      return "with_params("
          + pos1
          + ", "
          + pos2
          + ", "
          + posOrNamed
          + ", "
          + named
          + ", "
          + optionalNamed
          + ", "
          + nonNoneable.toString()
          + ")";
    }

    @Override
    public String toString() {
      return "<mock>";
    }
  }

  @SkylarkModule(name = "MockInterface", doc = "")
  static interface MockInterface {
    @SkylarkCallable(doc = "")
    public Boolean isEmptyInterface(String str);
  }

  static final class MockSubClass extends Mock implements MockInterface {
    @Override
    public Boolean isEmpty(String str) {
      return str.isEmpty();
    }
    @Override
    public Boolean isEmptyInterface(String str) {
      return str.isEmpty();
    }
    @SkylarkCallable(doc = "")
    public Boolean isEmptyClassNotAnnotated(String str) {
      return str.isEmpty();
    }
  }

  @SkylarkModule(name = "MockClassObject", doc = "", documented = false)
  static final class MockClassObject implements ClassObject {
    @Override
    public Object getValue(String name) {
      switch (name) {
        case "field": return "a";
        case "nset": return NestedSetBuilder.stableOrder().build();
      }
      return null;
    }

    @Override
    public ImmutableCollection<String> getKeys() {
      return ImmutableList.of("field", "nset");
    }

    @Override
    public String errorMessage(String name) {
      return null;
    }
  }

  @SkylarkModule(name = "MockMultipleMethodClass", doc = "")
  static final class MockMultipleMethodClass {
    @SuppressWarnings("unused")
    @SkylarkCallable(doc = "")
    public void method(Object o) {}
    @SuppressWarnings("unused")
    @SkylarkCallable(doc = "")
    public void method(String i) {}
  }

  @Test
  public void testSimpleIf() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  a = 0",
        "  x = 0",
        "  if x: a = 5",
        "  return a",
        "a = foo()").testLookup("a", 0);
  }

  @Test
  public void testIfPass() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  a = 1",
        "  x = True",
        "  if x: pass",
        "  return a",
        "a = foo()").testLookup("a", 1);
  }

  @Test
  public void testNestedIf() throws Exception {
    executeNestedIf(0, 0, 0);
    executeNestedIf(1, 0, 3);
    executeNestedIf(1, 1, 5);
  }

  private void executeNestedIf(int x, int y, int expected) throws Exception {
    String fun = String.format("foo%s%s", x, y);
    new SkylarkTest().setUp("def " + fun + "():",
        "  x = " + x,
        "  y = " + y,
        "  a = 0",
        "  b = 0",
        "  if x:",
        "    if y:",
        "      a = 2",
        "    b = 3",
        "  return a + b",
        "x = " + fun + "()").testLookup("x", expected);
  }

  @Test
  public void testIfElse() throws Exception {
    executeIfElse("foo", "something", 2);
    executeIfElse("bar", "", 3);
  }

  private void executeIfElse(String fun, String y, int expected) throws Exception {
    new SkylarkTest().setUp("def " + fun + "():",
        "  y = '" + y + "'",
        "  x = 5",
        "  if x:",
        "    if y: a = 2",
        "    else: a = 3",
        "  return a",
        "z = " + fun + "()").testLookup("z", expected);
  }

  @Test
  public void testIfElifElse_IfExecutes() throws Exception {
    execIfElifElse(1, 0, 1);
  }

  @Test
  public void testIfElifElse_ElifExecutes() throws Exception {
    execIfElifElse(0, 1, 2);
  }

  @Test
  public void testIfElifElse_ElseExecutes() throws Exception {
    execIfElifElse(0, 0, 3);
  }

  private void execIfElifElse(int x, int y, int v) throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  x = " + x + "",
        "  y = " + y + "",
        "  if x:",
        "    return 1",
        "  elif y:",
        "    return 2",
        "  else:",
        "    return 3",
        "v = foo()").testLookup("v", v);
  }

  @Test
  public void testForOnList() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  s = ''",
        "  for i in ['hello', ' ', 'world']:",
        "    s = s + i",
        "  return s",
        "s = foo()").testLookup("s", "hello world");
  }

  @Test
  public void testForOnString() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  s = []",
        "  for i in 'abc':",
        "    s = s + [i]",
        "  return s",
        "s = foo()").testExactOrder("s", "a", "b", "c");
  }

  @Test
  public void testForAssignmentList() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  d = ['a', 'b', 'c']",
        "  s = ''",
        "  for i in d:",
        "    s = s + i",
        "    d = ['d', 'e', 'f']", // check that we use the old list
        "  return s",
        "s = foo()").testLookup("s", "abc");
  }

  @Test
  public void testForAssignmentDict() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  d = {'a' : 1, 'b' : 2, 'c' : 3}",
        "  s = ''",
        "  for i in d:",
        "    s = s + i",
        "    d = {'d' : 1, 'e' : 2, 'f' : 3}",
        "  return s",
        "s = func()").testLookup("s", "abc");
  }

  @Test
  public void testForUpdateList() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  xs = [1, 2, 3]",
        "  for x in xs:",
        "    if x == 1:",
        "      xs.append(10)"
        ).testIfErrorContains("trying to mutate a locked object", "foo()");
  }

  @Test
  public void testForUpdateDict() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  d = {'a': 1, 'b': 2, 'c': 3}",
        "  for k in d:",
        "    d[k] *= 2"
        ).testIfErrorContains("trying to mutate a locked object", "foo()");
  }

  @Test
  public void testForUnlockedAfterBreak() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  xs = [1, 2]",
        "  for x in xs:",
        "    break",
        "  xs.append(3)",
        "  return xs"
        ).testEval("foo()", "[1, 2, 3]");
  }

  @Test
  public void testForNestedOnSameListStillLocked() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  xs = [1, 2]",
        "  ys = []",
        "  for x1 in xs:",
        "    for x2 in xs:",
        "      ys.append(x1 * x2)",
        "    xs.append(4)",
        "  return ys"
        ).testIfErrorContains("trying to mutate a locked object", "foo()");
  }

  @Test
  public void testForNestedOnSameListErrorMessage() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  xs = [1, 2]",
        "  ys = []",
        "  for x1 in xs:",
        "    for x2 in xs:",
        "      ys.append(x1 * x2)",
        "      xs.append(4)",
        "  return ys"
        // No file name in message, due to how test is set up.
        ).testIfErrorContains("Object locked at the following location(s): :4:3, :5:5", "foo()");
  }

  @Test
  public void testForNestedOnSameListUnlockedAtEnd() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  xs = [1, 2]",
        "  ys = []",
        "  for x1 in xs:",
        "    for x2 in xs:",
        "      ys.append(x1 * x2)",
        "  xs.append(4)",
        "  return ys"
        ).testEval("foo()", "[1, 2, 2, 4]");
  }

  @Test
  public void testForNestedWithListCompGood() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  xs = [1, 2]",
        "  ys = []",
        "  for x in xs:",
        "    zs = [None for x in xs for y in (ys.append(x) or ys)]",
        "  return ys"
        ).testEval("foo()", "[1, 2, 1, 2]");
  }
  @Test
  public void testForNestedWithListCompBad() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  xs = [1, 2, 3]",
        "  ys = []",
        "  for x in xs:",
        "    zs = [None for x in xs for y in (xs.append(x) or ys)]",
        "  return ys"
        ).testIfErrorContains("trying to mutate a locked object", "foo()");
  }

  @Test
  public void testForDeepUpdate() throws Exception {
    // Check that indirectly reachable values can still be manipulated as normal.
    new SkylarkTest().setUp("def foo():",
        "  xs = [['a'], ['b'], ['c']]",
        "  ys = []",
        "  for x in xs:",
        "    for y in x:",
        "      ys.append(y)",
        "    xs[2].append(x[0])",
        "  return ys",
        "ys = foo()").testLookup("ys", MutableList.of(null, "a", "b", "c", "a", "b"));
  }

  @Test
  public void testForNotIterable() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfErrorContains(
            "type 'int' is not iterable",
            "def func():",
            "  for i in mock.value_of('1'): a = i",
            "func()\n");
  }

  @Test
  public void testForOnDictionary() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  d = {1: 'a', 2: 'b', 3: 'c'}",
        "  s = ''",
        "  for i in d: s = s + d[i]",
        "  return s",
        "s = foo()").testLookup("s", "abc");
  }

  @Test
  public void testBadDictKey() throws Exception {
    new SkylarkTest().testIfErrorContains(
        "unhashable type: 'list'",
        "{ [1, 2]: [3, 4] }");
  }

  @Test
  public void testForLoopReuseVariable() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  s = ''",
        "  for i in ['a', 'b']:",
        "    for i in ['c', 'd']: s = s + i",
        "  return s",
        "s = foo()").testLookup("s", "cdcd");
  }

  @Test
  public void testForLoopMultipleVariables() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  s = ''",
        "  for [i, j] in [[1, 2], [3, 4]]:",
        "    s = s + str(i) + str(j) + '.'",
        "  return s",
        "s = foo()").testLookup("s", "12.34.");
  }

  @Test
  public void testForLoopBreak() throws Exception {
    simpleFlowTest("break", 1);
  }

  @Test
  public void testForLoopContinue() throws Exception {
    simpleFlowTest("continue", 10);
  }

  @SuppressWarnings("unchecked")
  private void simpleFlowTest(String statement, int expected) throws Exception {
    eval("def foo():",
        "  s = 0",
        "  hit = 0",
        "  for i in range(0, 10):",
        "    s = s + 1",
        "    " + statement + "",
        "    hit = 1",
        "  return [s, hit]",
        "x = foo()");
    assertThat((Iterable<Object>) lookup("x")).containsExactly(expected, 0).inOrder();
  }

  @Test
  public void testForLoopBreakFromDeeperBlock() throws Exception {
    flowFromDeeperBlock("break", 1);
    flowFromNestedBlocks("break", 29);
  }

  @Test
  public void testForLoopContinueFromDeeperBlock() throws Exception {
    flowFromDeeperBlock("continue", 5);
    flowFromNestedBlocks("continue", 39);
  }

  private void flowFromDeeperBlock(String statement, int expected) throws Exception {
    eval("def foo():",
        "   s = 0",
        "   for i in range(0, 10):",
        "       if i % 2 != 0:",
        "           " + statement + "",
        "       s = s + 1",
        "   return s",
        "x = foo()");
    assertThat(lookup("x")).isEqualTo(expected);
  }

  private void flowFromNestedBlocks(String statement, int expected) throws Exception {
    eval("def foo2():",
        "   s = 0",
        "   for i in range(1, 41):",
        "       if i % 2 == 0:",
        "           if i % 3 == 0:",
        "               if i % 5 == 0:",
        "                   " + statement + "",
        "       s = s + 1",
        "   return s",
        "y = foo2()");
    assertThat(lookup("y")).isEqualTo(expected);
  }

  @Test
  public void testNestedForLoopsMultipleBreaks() throws Exception {
    nestedLoopsTest("break", 2, 6, 6);
  }

  @Test
  public void testNestedForLoopsMultipleContinues() throws Exception {
    nestedLoopsTest("continue", 4, 20, 20);
  }

  @SuppressWarnings("unchecked")
  private void nestedLoopsTest(String statement, Integer outerExpected, int firstExpected,
      int secondExpected) throws Exception {
    eval("def foo():",
        "   outer = 0",
        "   first = 0",
        "   second = 0",
        "   for i in range(0, 5):",
        "       for j in range(0, 5):",
        "           if j == 2:",
        "               " + statement + "",
        "           first = first + 1",
        "       for k in range(0, 5):",
        "           if k == 2:",
        "               " + statement + "",
        "           second = second + 1",
        "       if i == 2:",
        "           " + statement + "",
        "       outer = outer + 1",
        "   return [outer, first, second]",
        "x = foo()");
    assertThat((Iterable<Object>) lookup("x"))
        .containsExactly(outerExpected, firstExpected, secondExpected).inOrder();
  }

  @Test
  public void testForLoopBreakError() throws Exception {
    flowStatementInsideFunction("break");
    flowStatementAfterLoop("break");
  }

  @Test
  public void testForLoopContinueError() throws Exception {
    flowStatementInsideFunction("continue");
    flowStatementAfterLoop("continue");
  }

  private void flowStatementInsideFunction(String statement) throws Exception {
    checkEvalErrorContains(statement + " statement must be inside a for loop",
        "def foo():",
        "  " + statement + "",
        "x = foo()");
  }

  private void flowStatementAfterLoop(String statement) throws Exception  {
    checkEvalErrorContains(statement + " statement must be inside a for loop",
        "def foo2():",
        "   for i in range(0, 3):",
        "      pass",
        "   " + statement + "",
        "y = foo2()");
  }

  @Test
  public void testNoneAssignment() throws Exception {
    new SkylarkTest()
        .setUp("def foo(x=None):", "  x = 1", "  x = None", "  return 2", "s = foo()")
        .testLookup("s", 2);
  }

  @Test
  public void testReassignment() throws Exception {
    eval("def foo(x=None):",
        "  x = 1",
        "  x = [1, 2]",
        "  x = 'str'",
        "  return x",
        "s = foo()");
    assertThat(lookup("s")).isEqualTo("str");
  }

  @Test
  public void testJavaCalls() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("b = mock.is_empty('a')")
        .testLookup("b", Boolean.FALSE);
  }

  @Test
  public void testJavaCallsOnSubClass() throws Exception {
    new SkylarkTest()
        .update("mock", new MockSubClass())
        .setUp("b = mock.is_empty('a')")
        .testLookup("b", Boolean.FALSE);
  }

  @Test
  public void testJavaCallsOnInterface() throws Exception {
    new SkylarkTest()
        .update("mock", new MockSubClass())
        .setUp("b = mock.is_empty_interface('a')")
        .testLookup("b", Boolean.FALSE);
  }

  @Test
  public void testJavaCallsNotSkylarkCallable() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfExactError("type 'Mock' has no method value()", "mock.value()");
  }

  @Test
  public void testNoOperatorIndex() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfExactError("type 'Mock' has no operator [](int)", "mock[2]");
  }

  @Test
  public void testJavaCallsNoMethod() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfExactError("type 'Mock' has no method bad()", "mock.bad()");
  }

  @Test
  public void testJavaCallsNoMethodErrorMsg() throws Exception {
    new SkylarkTest()
        .testIfExactError(
            "type 'int' has no method bad(string, string, string)", "s = 3.bad('a', 'b', 'c')");
  }

  @Test
  public void testJavaCallsMultipleMethod() throws Exception {
    new SkylarkTest()
        .update("mock", new MockMultipleMethodClass())
        .testIfExactError(
            "type 'MockMultipleMethodClass' has multiple matches for function method(string)",
            "s = mock.method('string')");
  }

  @Test
  public void testJavaCallWithKwargs() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfExactError(
            "type 'Mock' has no method isEmpty(string str)", "mock.isEmpty(str='abc')");
  }


  @Test
  public void testJavaCallWithPositionalAndKwargs() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("b = mock.with_params(1, True, named=True)")
        .testLookup("b", "with_params(1, true, false, true, false, a)");
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("")
        .testIfExactError(
            "parameter 'named' has no default value, in method with_params(int, bool) of 'Mock'",
            "mock.with_params(1, True)");
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("")
        .testIfExactError(
            "parameter 'named' has no default value, in method with_params(int, bool, bool) "
                + "of 'Mock'",
            "mock.with_params(1, True, True)");
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("b = mock.with_params(1, True, True, named=True)")
        .testLookup("b", "with_params(1, true, true, true, false, a)");
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("b = mock.with_params(1, True, named=True, posOrNamed=True)")
        .testLookup("b", "with_params(1, true, true, true, false, a)");
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("b = mock.with_params(1, True, named=True, posOrNamed=True, optionalNamed=True)")
        .testLookup("b", "with_params(1, true, true, true, true, a)");
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("")
        .testIfExactError(
            "too many arguments, in method with_params(int, bool, bool named, "
                + "bool posOrNamed, int n) of 'Mock'",
            "mock.with_params(1, True, named=True, posOrNamed=True, n=2)");
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("")
        .testIfExactError(
            "parameter 'nonNoneable' cannot be None, in method with_params(int, bool, bool, "
                + "bool named, bool optionalNamed, NoneType nonNoneable) of 'Mock'",
            "mock.with_params(1, True, True, named=True, optionalNamed=False, nonNoneable=None)");
  }

  @Test
  public void testNoJavaCallsWithoutSkylark() throws Exception {
    new SkylarkTest().testIfExactError("type 'int' has no method to_string()", "s = 3.to_string()");
  }

  @Test
  public void testNoJavaCallsIfClassNotAnnotated() throws Exception {
    new SkylarkTest()
        .update("mock", new MockSubClass())
        .testIfExactError(
            "type 'Mock' has no method is_empty_class_not_annotated(string)",
            "b = mock.is_empty_class_not_annotated('a')");
  }

  @Test
  public void testStructAccess() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("v = mock.struct_field")
        .testLookup("v", "a");
  }

  @Test
  public void testStructAccessAsFuncallNonCallable() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfExactError("'string' object is not callable", "v = mock.struct_field()");
  }

  @Test
  public void testStructAccessAsFuncall() throws Exception {
    foobar.configure(getClass().getDeclaredField("foobar").getAnnotation(SkylarkSignature.class));
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("v = mock.struct_field_callable()")
        .testLookup("v", "foobar");
  }

  @Test
  public void testStructAccessOfMethod() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfExactError("object of type 'Mock' has no field 'function'", "v = mock.function");
  }

  @Test
  public void testStructAccessTypo() throws Exception {
    new SkylarkTest()
        .update("mock", new MockClassObject())
        .testIfExactError(
            "object of type 'MockClassObject' has no field 'fild' (did you mean 'field'?)",
            "mock.fild");
  }

  @Test
  public void testJavaFunctionReturnsMutableObject() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfExactError(
            "method 'return_bad' returns an object of invalid type Bad", "mock.return_bad()");
  }

  @Test
  public void testJavaFunctionReturnsNullFails() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testIfErrorContains(
            "method invocation returned None,"
                + " please file a bug report: nullfunc_failing(\"abc\", 1)",
            "mock.nullfunc_failing('abc', 1)");
  }

  @Test
  public void testClassObjectAccess() throws Exception {
    new SkylarkTest()
        .update("mock", new MockClassObject())
        .setUp("v = mock.field")
        .testLookup("v", "a");
  }

  @Test
  public void testInSet() throws Exception {
    new SkylarkTest()
        .testStatement("'b' in depset(['a', 'b'])", Boolean.TRUE)
        .testStatement("'c' in depset(['a', 'b'])", Boolean.FALSE)
        .testStatement("1 in depset(['a', 'b'])", Boolean.FALSE);
  }

  @Test
  public void testUnionSet() throws Exception {
    new SkylarkTest()
        .testStatement("str(depset([1, 3]) | depset([1, 2]))", "depset([1, 2, 3])")
        .testStatement("str(depset([1, 2]) | [1, 3])", "depset([1, 2, 3])")
        .testIfExactError("unsupported operand type(s) for |: 'int' and 'int'", "2 | 4");
  }

  @Test
  public void testClassObjectCannotAccessNestedSet() throws Exception {
    new SkylarkTest()
        .update("mock", new MockClassObject())
        .testIfErrorContains("internal error: type 'NestedSet' is not allowed", "v = mock.nset");
  }

  @Test
  public void testJavaFunctionReturnsNone() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("v = mock.nullfunc_working()")
        .testLookup("v", Runtime.NONE);
  }

  @Test
  public void testVoidJavaFunctionReturnsNone() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("v = mock.voidfunc()")
        .testLookup("v", Runtime.NONE);
  }

  @Test
  public void testAugmentedAssignment() throws Exception {
    new SkylarkTest().setUp("def f1(x):",
        "  x += 1",
        "  return x",
        "",
        "foo = f1(41)").testLookup("foo", 42);
  }

  @Test
  public void testAugmentedAssignmentHasNoSideEffects() throws Exception {
    new SkylarkTest().setUp(
        "counter = [0]",
        "value = [1, 2]",
        "",
        "def f():",
        "  counter[0] = counter[0] + 1",
        "  return value",
        "",
        "f()[1] += 1")  // `f()` should be called only once here
        .testLookup("counter", MutableList.of(env, 1));
  }

  @Test
  public void testAugmentedAssignmentNotAllowedForListLiterals() throws Exception {
    new SkylarkTest().testIfErrorContains("Cannot perform augment assignment on a list literal",
        "def f(a, b):",
        "  [a, b] += []",
        "f(1, 2)");
  }

  @Test
  public void testAssignmentEvaluationOrder() throws Exception {
    new SkylarkTest().setUp(
        "ordinary = []",
        "augmented = []",
        "value = [1, 2]",
        "",
        "def f(record):",
        "  record.append('f')",
        "  return value",
        "",
        "def g(record):",
        "  record.append('g')",
        "  return value",
        "",
        "f(ordinary)[0] = g(ordinary)[1]",
        "f(augmented)[0] += g(augmented)[1]")
        .testLookup("ordinary", MutableList.of(env, "g", "f"))    // This order is consistent
        .testLookup("augmented", MutableList.of(env, "f", "g"));  // with Python
  }

  @Test
  public void testStaticDirectJavaCall() throws Exception {
    new SkylarkTest().update("Mock", Mock.class).setUp("val = Mock.value_of('8')")
        .testLookup("val", 8);
  }

  @Test
  public void testStaticDirectJavaCallMethodIsNonStatic() throws Exception {
    new SkylarkTest()
        .update("Mock", Mock.class)
        .testIfExactError("method 'is_empty' is not static", "val = Mock.is_empty('a')");
  }

  @Test
  public void testDictComprehensions_IterationOrder() throws Exception {
    new SkylarkTest().setUp("def foo():",
        "  d = {x : x for x in ['c', 'a', 'b']}",
        "  s = ''",
        "  for a in d:",
        "    s += a",
        "  return s",
        "s = foo()").testLookup("s", "cab");
  }

  @Test
  public void testDotExpressionOnNonStructObject() throws Exception {
    new SkylarkTest()
        .testIfExactError("object of type 'string' has no field 'field'", "x = 'a'.field");
  }

  @Test
  public void testPlusEqualsOnListCopying() throws Exception {
    new SkylarkTest("--incompatible_list_plus_equals=false")
        .setUp(
            "def func():",
            "  l1 = [1, 2]",
            "  l2 = l1",
            "  l2 += [3, 4]",
            "  return l1, l2",
            "lists = str(func())")
        .testLookup("lists", "([1, 2], [1, 2, 3, 4])");
  }

  @Test
  public void testPlusEqualsOnListMutating() throws Exception {
    new SkylarkTest("--incompatible_list_plus_equals=true")
        .setUp(
            "def func():",
            "  l1 = [1, 2]",
            "  l2 = l1",
            "  l2 += [3, 4]",
            "  return l1, l2",
            "lists = str(func())")
        .testLookup("lists", "([1, 2, 3, 4], [1, 2, 3, 4])");

    // The same but with += after an IndexExpression
    new SkylarkTest("--incompatible_list_plus_equals=true")
        .setUp(
            "def func():",
            "  l = [1, 2]",
            "  d = {0: l}",
            "  d[0] += [3, 4]",
            "  return l, d[0]",
            "lists = str(func())")
        .testLookup("lists", "([1, 2, 3, 4], [1, 2, 3, 4])");
  }

  @Test
  public void testPlusEqualsOnTuple() throws Exception {
    new SkylarkTest("--incompatible_list_plus_equals=false")
        .setUp(
            "def func():",
            "  t1 = (1, 2)",
            "  t2 = t1",
            "  t2 += (3, 4)",
            "  return t1, t2",
            "tuples = func()")
        .testLookup("tuples", SkylarkList.Tuple.of(
            SkylarkList.Tuple.of(1, 2),
            SkylarkList.Tuple.of(1, 2, 3, 4)
        ));

    // This behavior should remain the same regardless of the incompatible_list_plus_equals flag
    new SkylarkTest("--incompatible_list_plus_equals=true")
        .setUp(
            "def func():",
            "  t1 = (1, 2)",
            "  t2 = t1",
            "  t2 += (3, 4)",
            "  return t1, t2",
            "tuples = func()")
        .testLookup("tuples", SkylarkList.Tuple.of(
            SkylarkList.Tuple.of(1, 2),
            SkylarkList.Tuple.of(1, 2, 3, 4)
        ));
  }

  @Test
  public void testPlusEqualsOnDict() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  d = {'a' : 1}",
        "  d += {'b' : 2}",
        "  return d",
        "d = func()")
        .testLookup("d", ImmutableMap.of("a", 1, "b", 2));
  }

  @Test
  public void testDictAssignmentAsLValue() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  d = {'a' : 1}",
        "  d['b'] = 2",
        "  return d",
        "d = func()").testLookup("d", ImmutableMap.of("a", 1, "b", 2));
  }

  @Test
  public void testNestedDictAssignmentAsLValue() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  d = {'a' : 1}",
        "  e = {'d': d}",
        "  e['d']['b'] = 2",
        "  return e",
        "e = func()").testLookup("e", ImmutableMap.of("d", ImmutableMap.of("a", 1, "b", 2)));
  }

  @Test
  public void testListAssignmentAsLValue() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  a = [1, 2]",
        "  a[1] = 3",
        "  a[-2] = 4",
        "  return a",
        "a = str(func())").testLookup("a", "[4, 3]");
  }

  @Test
  public void testNestedListAssignmentAsLValue() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  d = [1, 2]",
        "  e = [3, d]",
        "  e[1][1] = 4",
        "  return e",
        "e = str(func())").testLookup("e", "[3, [1, 4]]");
  }
  
  @Test
  public void testDictTupleAssignmentAsLValue() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  d = {'a' : 1}",
        "  d['b'], d['c'] = 2, 3",
        "  return d",
        "d = func()").testLookup("d", ImmutableMap.of("a", 1, "b", 2, "c", 3));
  }

  @Test
  public void testDictItemPlusEqual() throws Exception {
    new SkylarkTest().setUp("def func():",
        "  d = {'a' : 2}",
        "  d['a'] += 3",
        "  return d",
        "d = func()").testLookup("d", ImmutableMap.of("a", 5));
  }

  @Test
  public void testDictAssignmentAsLValueSideEffects() throws Exception {
    new SkylarkTest().setUp("def func(d):",
        "  d['b'] = 2",
        "d = {'a' : 1}",
        "func(d)").testLookup("d", SkylarkDict.of(null, "a", 1, "b", 2));
  }

  @Test
  public void testAssignmentToListInDictSideEffect() throws Exception {
    new SkylarkTest().setUp(
        "l = [1, 2]",
        "d = {0: l}",
        "d[0].append(3)").testLookup("l", MutableList.of(null, 1, 2, 3));
  }

  @Test
  public void testTopLevelDict() throws Exception {
    new SkylarkTest().setUp("if 1:",
      "  v = 'a'",
      "else:",
      "  v = 'b'").testLookup("v", "a");
  }

  @Test
  public void testUserFunctionKeywordArgs() throws Exception {
    new SkylarkTest().setUp("def foo(a, b, c):",
        "  return a + b + c", "s = foo(1, c=2, b=3)")
        .testLookup("s", 6);
  }

  @Test
  public void testFunctionCallOrdering() throws Exception {
    new SkylarkTest().setUp("def func(): return foo() * 2",
         "def foo(): return 2",
         "x = func()")
         .testLookup("x", 4);
  }

  @Test
  public void testFunctionCallBadOrdering() throws Exception {
    new SkylarkTest().testIfErrorContains("name 'foo' is not defined",
         "def func(): return foo() * 2",
         "x = func()",
         "def foo(): return 2");
  }

  @Test
  public void testTypo() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "name 'my_variable' is not defined (did you mean 'myVariable'?)",
            "myVariable = 2",
            "x = my_variable + 1");
  }

  @Test
  public void testNoneTrueFalseInSkylark() throws Exception {
    new SkylarkTest().setUp("a = None",
      "b = True",
      "c = False")
      .testLookup("a", Runtime.NONE)
      .testLookup("b", Boolean.TRUE)
      .testLookup("c", Boolean.FALSE);
  }

  @Test
  public void testHasattrMethods() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .setUp("a = hasattr(mock, 'struct_field')", "b = hasattr(mock, 'function')",
            "c = hasattr(mock, 'is_empty')", "d = hasattr('str', 'replace')",
            "e = hasattr(mock, 'other')\n")
        .testLookup("a", Boolean.TRUE)
        .testLookup("b", Boolean.TRUE)
        .testLookup("c", Boolean.TRUE)
        .testLookup("d", Boolean.TRUE)
        .testLookup("e", Boolean.FALSE);
  }

  @Test
  public void testListAnTupleConcatenationDoesNotWorkInSkylark() throws Exception {
    new SkylarkTest().testIfExactError("unsupported operand type(s) for +: 'list' and 'tuple'",
        "[1, 2] + (3, 4)");
  }

  @Test
  public void testCannotCreateMixedListInSkylark() throws Exception {
    new SkylarkTest().testExactOrder("['a', 'b', 1, 2]", "a", "b", 1, 2);
  }

  @Test
  public void testCannotConcatListInSkylarkWithDifferentGenericTypes() throws Exception {
    new SkylarkTest().testExactOrder("[1, 2] + ['a', 'b']", 1, 2, "a", "b");
  }

  @Test
  public void testConcatEmptyListWithNonEmptyWorks() throws Exception {
    new SkylarkTest().testExactOrder("[] + ['a', 'b']", "a", "b");
  }

  @Test
  public void testFormatStringWithTuple() throws Exception {
    new SkylarkTest().setUp("v = '%s%s' % ('a', 1)").testLookup("v", "a1");
  }

  @Test
  public void testSingletonTuple() throws Exception {
    new SkylarkTest().testExactOrder("(1,)", 1);
  }

  @Test
  public void testDirFindsClassObjectFields() throws Exception {
    new SkylarkTest().update("mock", new MockClassObject())
        .testExactOrder("dir(mock)", "field", "nset");
  }

  @Test
  public void testDirFindsJavaObjectStructFieldsAndMethods() throws Exception {
    new SkylarkTest()
        .update("mock", new Mock())
        .testExactOrder(
            "dir(mock)",
            "function",
            "is_empty",
            "nullfunc_failing",
            "nullfunc_working",
            "return_bad",
            "string",
            "string_list",
            "struct_field",
            "struct_field_callable",
            "value_of",
            "voidfunc",
            "with_params");
  }

  @Test
  public void testPrint() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    setFailFast(false);
    eval("print('hello')");
    assertContainsWarning("hello");
    eval("print('a', 'b')");
    assertContainsWarning("a b");
    eval("print('a', 'b', sep='x')");
    assertContainsWarning("axb");
  }

  @Test
  public void testPrintBadKwargs() throws Exception {
    new SkylarkTest().testIfExactError(
        "unexpected keywords 'end', 'other' in call to print(*args, sep: string = \" \")",
        "print(end='x', other='y')");
  }

  @Test
  public void testSkylarkTypes() {
    assertEquals(TransitiveInfoCollection.class,
        EvalUtils.getSkylarkType(FileConfiguredTarget.class));
    assertEquals(TransitiveInfoCollection.class,
        EvalUtils.getSkylarkType(RuleConfiguredTarget.class));
    assertEquals(Artifact.class, EvalUtils.getSkylarkType(SpecialArtifact.class));
  }

  // Override tests in EvaluationTest incompatible with Skylark

  @SuppressWarnings("unchecked")
  @Override
  @Test
  public void testConcatLists() throws Exception {
    new SkylarkTest().testExactOrder("[1,2] + [3,4]", 1, 2, 3, 4).testExactOrder("(1,2)", 1, 2)
        .testExactOrder("(1,2) + (3,4)", 1, 2, 3, 4);

    // TODO(fwe): cannot be handled by current testing suite
    // list
    Object x = eval("[1,2] + [3,4]");
    assertThat((Iterable<Object>) x).containsExactly(1, 2, 3, 4).inOrder();

    // tuple
    x = eval("(1,2)");
    assertThat((Iterable<Object>) x).containsExactly(1, 2).inOrder();
    assertTrue(((SkylarkList) x).isTuple());

    x = eval("(1,2) + (3,4)");
    assertThat((Iterable<Object>) x).containsExactly(1, 2, 3, 4).inOrder();
    assertTrue(((SkylarkList) x).isTuple());
  }

  @Override
  @Test
  public void testListConcatenation() throws Exception {}

  @Override
  @Test
  public void testListComprehensionsMultipleVariablesFail() throws Exception {
    new SkylarkTest()
        .testIfErrorContains(
            "lvalue has length 3, but rvalue has has length 2",
            "def foo (): return [x + y for x, y, z in [(1, 2), (3, 4)]]",
            "foo()");

    new SkylarkTest()
        .testIfErrorContains(
            "type 'int' is not a collection",
            "def bar (): return [x + y for x, y in (1, 2)]",
            "bar()");

    new SkylarkTest()
        .testIfErrorContains(
            "lvalue has length 3, but rvalue has has length 2",
            "[x + y for x, y, z in [(1, 2), (3, 4)]]");

    // can't reuse the same local variable twice(!)
    new SkylarkTest()
        .testIfErrorContains(
            "Variable x is read only", "[x + y for x, y in (1, 2)]", "[x + y for x, y in (1, 2)]");

    new SkylarkTest()
        .testIfErrorContains("type 'int' is not a collection", "[x2 + y2 for x2, y2 in (1, 2)]");
  }

  @Override
  @Test
  public void testNotCallInt() throws Exception {
    new SkylarkTest().setUp("sum = 123456").testLookup("sum", 123456)
        .testIfExactError("'int' object is not callable", "sum(1, 2, 3, 4, 5, 6)")
        .testStatement("sum", 123456);
  }

  @Test
  public void testConditionalExpressionAtToplevel() throws Exception {
    new SkylarkTest().setUp("x = 1 if 2 else 3").testLookup("x", 1);
  }

  @Test
  public void testConditionalExpressionInFunction() throws Exception {
    new SkylarkTest().setUp("def foo(a, b, c): return a+b if c else a-b\n").testStatement(
        "foo(23, 5, 0)", 18);
  }

  @SkylarkModule(name = "SkylarkClassObjectWithSkylarkCallables", doc = "")
  static final class SkylarkClassObjectWithSkylarkCallables extends SkylarkClassObject {
    private static final NativeClassObjectConstructor CONSTRUCTOR =
        new NativeClassObjectConstructor("struct_with_skylark_callables") {};

    SkylarkClassObjectWithSkylarkCallables() {
      super(
          CONSTRUCTOR,
          ImmutableMap.of(
              "values_only_field",
              "fromValues",
              "values_only_method",
              new BuiltinFunction("values_only_method", FunctionSignature.of()) {
                public String invoke() {
                  return "fromValues";
                }
              },
              "collision_field",
              "fromValues",
              "collision_method",
              new BuiltinFunction("collision_method", FunctionSignature.of()) {
                public String invoke() {
                  return "fromValues";
                }
              }));
    }

    @SkylarkCallable(name = "callable_only_field", doc = "", structField = true)
    public String getCallableOnlyField() {
      return "fromSkylarkCallable";
    }

    @SkylarkCallable(name = "callable_only_method", doc = "", structField = false)
    public String getCallableOnlyMethod() {
      return "fromSkylarkCallable";
    }

    @SkylarkCallable(name = "collision_field", doc = "", structField = true)
    public String getCollisionField() {
      return "fromSkylarkCallable";
    }

    @SkylarkCallable(name = "collision_method", doc = "", structField = false)
    public String getCollisionMethod() {
      return "fromSkylarkCallable";
    }
  }

  @Test
  public void testStructFieldDefinedOnlyInValues() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .setUp("v = val.values_only_field")
        .testLookup("v", "fromValues");
  }

  @Test
  public void testStructMethodDefinedOnlyInValues() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .setUp("v = val.values_only_method()")
        .testLookup("v", "fromValues");
  }

  @Test
  public void testStructFieldDefinedOnlyInSkylarkCallable() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .setUp("v = val.callable_only_field")
        .testLookup("v", "fromSkylarkCallable");
  }

  @Test
  public void testStructMethodDefinedOnlyInSkylarkCallable() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .setUp("v = val.callable_only_method()")
        .testLookup("v", "fromSkylarkCallable");
  }

  @Test
  public void testStructFieldDefinedInValuesAndSkylarkCallable() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .setUp("v = val.collision_field")
        .testLookup("v", "fromValues");
  }

  @Test
  public void testStructMethodDefinedInValuesAndSkylarkCallable() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .setUp("v = val.collision_method()")
        .testLookup("v", "fromValues");
  }

  @Test
  public void testStructFieldNotDefined() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .testIfExactError(
            // TODO(bazel-team): This should probably list callable_only_field/method as well.
            "'struct_with_skylark_callables' object has no attribute 'nonexistent_field'\n"
                + "Available attributes: collision_field, collision_method, values_only_field, "
                + "values_only_method",
            "v = val.nonexistent_field");
  }

  @Test
  public void testStructMethodNotDefined() throws Exception {
    new SkylarkTest()
        .update("val", new SkylarkClassObjectWithSkylarkCallables())
        .testIfExactError(
            // TODO(bazel-team): This should probably match the error above better.
            "struct has no method 'nonexistent_method'", "v = val.nonexistent_method()");
  }
}
