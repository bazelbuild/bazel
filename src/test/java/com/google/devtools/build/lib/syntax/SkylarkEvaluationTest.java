// Copyright 2015 Google Inc. All rights reserved.
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Evaluation tests with Skylark Environment.
 */
@RunWith(JUnit4.class)
public class SkylarkEvaluationTest extends EvaluationTest {

  // Restoring defaults overridden by EvaluationTest
  @Override
  public EvaluationContext newEvaluationContext() {
    return SkylarkModules.newEvaluationContext(getEventHandler());
  }

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
    public Mock returnMutable() {
      return new Mock();
    }
    @SkylarkCallable(name = "struct_field", doc = "", structField = true)
    public String structField() {
      return "a";
    }
    @SkylarkCallable(name = "function", doc = "", structField = false)
    public String function() {
      return "a";
    }
    @SuppressWarnings("unused")
    @SkylarkCallable(name = "nullfunc_failing", doc = "", allowReturnNones = false)
    public Object nullfuncFailing(String p1, Integer p2) {
      return null;
    }
    @SkylarkCallable(name = "nullfunc_working", doc = "", allowReturnNones = true)
    public Object nullfuncWorking() {
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

  static final class MockClassObject implements ClassObject {
    @Override
    public Object getValue(String name) {
      switch (name) {
        case "field": return "a";
        case "nset": return NestedSetBuilder.stableOrder().build();
      }
      throw new IllegalStateException();
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
    eval("def foo():",
        "  a = 0",
        "  x = 0",
        "  if x: a = 5",
        "  return a",
        "a = foo()");
    assertEquals(0, lookup("a"));
  }

  @Test
  public void testIfPass() throws Exception {
    eval("def foo():",
        "  a = 1",
        "  x = True",
        "  if x: pass",
        "  return a",
        "a = foo()");
    assertEquals(1, lookup("a"));
  }

  @Test
  public void testNestedIf() throws Exception {
    executeNestedIf(0, 0);
    assertEquals(0, lookup("x"));

    setUp();
    executeNestedIf(1, 0);
    assertEquals(3, lookup("x"));

    setUp();
    executeNestedIf(1, 1);
    assertEquals(5, lookup("x"));
  }

  private void executeNestedIf(int x, int y) throws Exception {
    String fun = String.format("foo%s%s", x, y);
    eval("def " + fun + "():",
        "  x = " + x,
        "  y = " + y,
        "  a = 0",
        "  b = 0",
        "  if x:",
        "    if y:",
        "      a = 2",
        "    b = 3",
        "  return a + b",
        "x = " + fun + "()");
  }

  @Test
  public void testIfElse() throws Exception {
    executeIfElse("foo", "something", 2);
    executeIfElse("bar", "", 3);
  }

  private void executeIfElse(String fun, String y, int expected) throws Exception {
    eval("def " + fun + "():",
        "  y = '" + y + "'",
        "  x = 5",
        "  if x:",
        "    if y: a = 2",
        "    else: a = 3",
        "  return a\n");
    assertEquals(expected, eval(fun + "()"));
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
    eval("def foo():",
        "  x = " + x + "",
        "  y = " + y + "",
        "  if x:",
        "    return 1",
        "  elif y:",
        "    return 2",
        "  else:",
        "    return 3",
        "v = foo()");
    assertEquals(v, lookup("v"));
  }

  @Test
  public void testForOnList() throws Exception {
    eval("def foo():",
        "  s = ''",
        "  for i in ['hello', ' ', 'world']:",
        "    s = s + i",
        "  return s",
        "s = foo()");
    assertEquals("hello world", lookup("s"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testForOnString() throws Exception {
    eval("def foo():",
        "  s = []",
        "  for i in 'abc':",
        "    s = s + [i]",
        "  return s",
        "s = foo()");
    assertThat((Iterable<Object>) lookup("s")).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void testForAssignmentList() throws Exception {
    eval("def foo():",
        "  d = ['a', 'b', 'c']",
        "  s = ''",
        "  for i in d:",
        "    s = s + i",
        "    d = ['d', 'e', 'f']",  // check that we use the old list
        "  return s",
        "s = foo()");
    assertEquals("abc", lookup("s"));
  }

  @Test
  public void testForAssignmentDict() throws Exception {
    eval("def func():",
        "  d = {'a' : 1, 'b' : 2, 'c' : 3}",
        "  s = ''",
        "  for i in d:",
        "    s = s + i",
        "    d = {'d' : 1, 'e' : 2, 'f' : 3}",
        "  return s",
        "s = func()");
    assertEquals("abc", lookup("s"));
  }

  @Test
  public void testForNotIterable() throws Exception {
    update("mock", new Mock());
    checkEvalError("type 'int' is not iterable",
        "def func():",
        "  for i in mock.value_of('1'): a = i",
        "func()");

  }

  @Test
  public void testForOnDictionary() throws Exception {
    eval("def foo():",
        "  d = {1: 'a', 2: 'b', 3: 'c'}",
        "  s = ''",
        "  for i in d: s = s + d[i]",
        "  return s",
        "s = foo()");
    assertEquals("abc", lookup("s"));
  }

  @Test
  public void testForLoopReuseVariable() throws Exception {
    eval("def foo():",
        "  s = ''",
        "  for i in ['a', 'b']:",
        "    for i in ['c', 'd']: s = s + i",
        "  return s",
        "s = foo()");
    assertEquals("cdcd", lookup("s"));
  }

  @Test
  public void testForLoopMultipleVariables() throws Exception {
    eval("def foo():",
        "  s = ''",
        "  for [i, j] in [[1, 2], [3, 4]]:",
        "    s = s + str(i) + str(j) + '.'",
        "  return s",
        "s = foo()");
    assertEquals("12.34.", lookup("s"));
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
    eval("def foo(x=None):",
        "  x = 1",
        "  x = None",
        "  return 2",
        "s = foo()");
    assertEquals(2, lookup("s"));
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
    update("mock", new Mock());
    eval("b = mock.is_empty('a')");
    assertEquals(Boolean.FALSE, lookup("b"));
  }

  @Test
  public void testJavaCallsOnSubClass() throws Exception {
    update("mock", new MockSubClass());
    eval("b = mock.is_empty('a')");
    assertEquals(Boolean.FALSE, lookup("b"));
  }

  @Test
  public void testJavaCallsOnInterface() throws Exception {
    update("mock", new MockSubClass());
    eval("b = mock.is_empty_interface('a')");
    assertEquals(Boolean.FALSE, lookup("b"));
  }

  @Test
  public void testJavaCallsNotSkylarkCallable() throws Exception {
    update("mock", new Mock());
    checkEvalError("No matching method found for value() in Mock", "mock.value()");
  }

  @Test
  public void testJavaCallsNoMethod() throws Exception {
    checkEvalError("No matching method found for bad() in int", "s = 3.bad()");
  }

  @Test
  public void testJavaCallsNoMethodErrorMsg() throws Exception {
    checkEvalError("No matching method found for bad(string, string, string) in int",
        "s = 3.bad('a', 'b', 'c')");
  }

  @Test
  public void testJavaCallsMultipleMethod() throws Exception {
    update("mock", new MockMultipleMethodClass());
    checkEvalError("Multiple matching methods for method(string) in MockMultipleMethodClass",
        "s = mock.method('string')");
  }

  @Test
  public void testJavaCallWithKwargs() throws Exception {
    checkEvalError("Keyword arguments are not allowed when calling a java method"
        + "\nwhile calling method 'compare_to' on object 3 of type int",
        "comp = 3.compare_to(x = 4)");
  }

  @Test
  public void testNoJavaCallsWithoutSkylark() throws Exception {
    checkEvalError("No matching method found for to_string() in int", "s = 3.to_string()\n");
  }

  @Test
  public void testNoJavaCallsIfClassNotAnnotated() throws Exception {
    update("mock", new MockSubClass());
    checkEvalError(
        "No matching method found for is_empty_class_not_annotated(string) in MockSubClass",
        "b = mock.is_empty_class_not_annotated('a')");
  }

  @Test
  public void testStructAccess() throws Exception {
    update("mock", new Mock());
    eval("v = mock.struct_field");
    assertEquals("a", lookup("v"));
  }

  @Test
  public void testStructAccessAsFuncall() throws Exception {
    update("mock", new Mock());
    checkEvalError("No matching method found for struct_field() in Mock",
        "v = mock.struct_field()");
  }

  @Test
  public void testStructAccessOfMethod() throws Exception {
    update("mock", new Mock());
    checkEvalError("Object of type 'Mock' has no field \"function\"",
        "v = mock.function");
  }

  @Test
  public void testConditionalStructConcatenation() throws Exception {
    eval("def func():",
        "  x = struct(a = 1, b = 2)",
        "  if True:",
        "    x += struct(c = 1, d = 2)",
        "  return x",
        "x = func()\n");
    SkylarkClassObject x = (SkylarkClassObject) lookup("x");
    assertEquals(1, x.getValue("a"));
    assertEquals(2, x.getValue("b"));
    assertEquals(1, x.getValue("c"));
    assertEquals(2, x.getValue("d"));
  }

  @Test
  public void testJavaFunctionReturnsMutableObject() throws Exception {
    update("mock", new Mock());
    checkEvalError("Method 'return_mutable' returns a mutable object (type of Mock)",
        "mock.return_mutable()");
  }

  @Test
  public void testJavaFunctionReturnsNullFails() throws Exception {
    update("mock", new Mock());
    checkEvalError("Method invocation returned None,"
        + " please contact Skylark developers: nullfunc_failing(\"abc\", 1)",
        "mock.nullfunc_failing('abc', 1)");
  }

  @Test
  public void testClassObjectAccess() throws Exception {
    update("mock", new MockClassObject());
    eval("v = mock.field");
    assertEquals("a", lookup("v"));
  }

  @Test
  public void testInSet() throws Exception {
    assertEquals(Boolean.TRUE, eval("'b' in set(['a', 'b'])"));
    assertEquals(Boolean.FALSE, eval("'c' in set(['a', 'b'])"));
    assertEquals(Boolean.FALSE, eval("1 in set(['a', 'b'])"));
  }

  @Test
  public void testClassObjectCannotAccessNestedSet() throws Exception {
    update("mock", new MockClassObject());
    checkEvalError("Type is not allowed in Skylark: EmptyNestedSet", "v = mock.nset");
  }

  @Test
  public void testJavaFunctionReturnsNone() throws Exception {
    update("mock", new Mock());
    eval("v = mock.nullfunc_working()");
    assertSame(Environment.NONE, lookup("v"));
  }

  @Test
  public void testVoidJavaFunctionReturnsNone() throws Exception {
    update("mock", new Mock());
    eval("v = mock.voidfunc()");
    assertSame(Environment.NONE, lookup("v"));
  }

  @Test
  public void testAugmentedAssignment() throws Exception {
    eval("def f1(x):",
        "  x += 1",
        "  return x",
        "",
        "foo = f1(41)\n");
    assertEquals(42, lookup("foo"));
  }

  @Test
  public void testStaticDirectJavaCall() throws Exception {
    update("Mock", Mock.class);
    eval("val = Mock.value_of('8')");
    assertEquals(8, lookup("val"));
  }

  @Test
  public void testStaticDirectJavaCallMethodIsNonStatic() throws Exception {
    update("Mock", Mock.class);
    checkEvalError("Method 'is_empty' is not static", "val = Mock.is_empty('a')");
  }

  @Test
  public void testDictComprehensions_IterationOrder() throws Exception {
    eval("def foo():",
        "  d = {x : x for x in ['c', 'a', 'b']}",
        "  s = ''",
        "  for a in d:",
        "    s += a",
        "  return s",
        "s = foo()");
    assertEquals("abc", lookup("s"));
  }

  @Test
  public void testStructCreation() throws Exception {
    eval("x = struct(a = 1, b = 2)");
    assertThat(lookup("x")).isInstanceOf(ClassObject.class);
  }

  @Test
  public void testStructFields() throws Exception {
    eval("x = struct(a = 1, b = 2)");
    ClassObject x = (ClassObject) lookup("x");
    assertEquals(1, x.getValue("a"));
    assertEquals(2, x.getValue("b"));
  }

  @Test
  public void testStructAccessingFieldsFromSkylark() throws Exception {
    eval("x = struct(a = 1, b = 2)",
        "x1 = x.a",
        "x2 = x.b\n");
    assertEquals(1, lookup("x1"));
    assertEquals(2, lookup("x2"));
  }

  @Test
  public void testStructAccessingUnknownField() throws Exception {
    checkEvalError("Object of type 'struct' has no field \"c\"",
        "x = struct(a = 1, b = 2)",
        "y = x.c\n");
  }

  @Test
  public void testStructAccessingFieldsWithArgs() throws Exception {
    checkEvalError("No matching method found for a(int) in struct",
        "x = struct(a = 1, b = 2)",
        "x1 = x.a(1)\n");
  }

  @Test
  public void testStructPosArgs() throws Exception {
    checkEvalError("struct(**kwargs) does not accept positional arguments, but got 1",
        "x = struct(1, b = 2)\n");
  }

  @Test
  public void testStructConcatenationFieldNames() throws Exception {
    eval("x = struct(a = 1, b = 2)",
        "y = struct(c = 1, d = 2)",
        "z = x + y\n");
    SkylarkClassObject z = (SkylarkClassObject) lookup("z");
    assertEquals(ImmutableSet.of("a", "b", "c", "d"), z.getKeys());
  }

  @Test
  public void testStructConcatenationFieldValues() throws Exception {
    eval("x = struct(a = 1, b = 2)",
        "y = struct(c = 1, d = 2)",
        "z = x + y\n");
    SkylarkClassObject z = (SkylarkClassObject) lookup("z");
    assertEquals(1, z.getValue("a"));
    assertEquals(2, z.getValue("b"));
    assertEquals(1, z.getValue("c"));
    assertEquals(2, z.getValue("d"));
  }

  @Test
  public void testStructConcatenationCommonFields() throws Exception {
    checkEvalError("Cannot concat structs with common field(s): a",
        "x = struct(a = 1, b = 2)",
        "y = struct(c = 1, a = 2)",
        "z = x + y\n");
  }

  @Test
  public void testDotExpressionOnNonStructObject() throws Exception {
    checkEvalError("Object of type 'string' has no field \"field\"", "x = 'a'.field");
  }

  @Test
  public void testPlusEqualsOnDict() throws Exception {
    eval("def func():",
        "  d = {'a' : 1}",
        "  d += {'b' : 2}",
        "  return d",
        "d = func()");
    assertEquals(ImmutableMap.of("a", 1, "b", 2), lookup("d"));
  }

  @Test
  public void testDictAssignmentAsLValue() throws Exception {
    eval("def func():",
        "  d = {'a' : 1}",
        "  d['b'] = 2",
        "  return d",
        "d = func()");
    assertEquals(ImmutableMap.of("a", 1, "b", 2), lookup("d"));
  }

  @Test
  public void testDictAssignmentAsLValueNoSideEffects() throws Exception {
    eval("def func(d):",
        "  d['b'] = 2",
        "d = {'a' : 1}",
        "func(d)");
    assertEquals(ImmutableMap.of("a", 1), lookup("d"));
  }

  @Test
  public void testListIndexAsLValueAsLValue() throws Exception {
    checkEvalError("unsupported operand type(s) for +: 'list' and 'dict'",
        "def id(l):",
        "  return l",
        "def func():",
        "  l = id([1])",
        "  l[0] = 2",
        "  return l",
        "l = func()");
  }

  @Test
  public void testTopLevelDict() throws Exception {
    eval("if 1:",
      "  v = 'a'",
      "else:",
      "  v = 'b'");
    assertEquals("a", lookup("v"));
  }

  @Test
  public void testUserFunctionKeywordArgs() throws Exception {
    eval("def foo(a, b, c):",
      "  return a + b + c",
      "s = foo(1, c=2, b=3)");
    assertEquals(6, lookup("s"));
  }

  @Test
  public void testFunctionCallOrdering() throws Exception {
    eval("def func(): return foo() * 2",
         "def foo(): return 2",
         "x = func()");
    assertThat(lookup("x")).isEqualTo(4);
  }

  @Test
  public void testFunctionCallBadOrdering() throws Exception {
    checkEvalError("name 'foo' is not defined",
         "def func(): return foo() * 2",
         "x = func()",
         "def foo(): return 2");
  }

  @Test
  public void testNoneTrueFalseInSkylark() throws Exception {
    eval("a = None",
      "b = True",
      "c = False");
    assertSame(Environment.NONE, lookup("a"));
    assertTrue((Boolean) lookup("b"));
    assertFalse((Boolean) lookup("c"));
  }

  @Test
  public void testHasattr() throws Exception {
    eval("s = struct(a=1)",
      "x = hasattr(s, 'a')",
      "y = hasattr(s, 'b')\n");
    assertTrue((Boolean) lookup("x"));
    assertFalse((Boolean) lookup("y"));
  }

  @Test
  public void testHasattrMethods() throws Exception {
    update("mock", new Mock());
    eval("a = hasattr(mock, 'struct_field')",
        "b = hasattr(mock, 'function')",
        "c = hasattr(mock, 'is_empty')",
        "d = hasattr('str', 'replace')",
        "e = hasattr(mock, 'other')\n");
    assertTrue((Boolean) lookup("a"));
    assertTrue((Boolean) lookup("b"));
    assertTrue((Boolean) lookup("c"));
    assertTrue((Boolean) lookup("d"));
    assertFalse((Boolean) lookup("e"));
  }

  @Test
  public void testGetattr() throws Exception {
    eval("s = struct(a='val')",
      "x = getattr(s, 'a')",
      "y = getattr(s, 'b', 'def')",
      "z = getattr(s, 'b', default = 'def')",
      "w = getattr(s, 'a', default='ignored')");
    assertEquals("val", lookup("x"));
    assertEquals("def", lookup("y"));
    assertEquals("def", lookup("z"));
    assertEquals("val", lookup("w"));
  }

  @Test
  public void testGetattrNoAttr() throws Exception {
    checkEvalError("Object of type 'struct' has no field \"b\"",
        "s = struct(a='val')",
        "getattr(s, 'b')");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListAnTupleConcatenationDoesNotWorkInSkylark() throws Exception {
    checkEvalError("cannot concatenate lists and tuples", "[1, 2] + (3, 4)");
  }

  @Test
  public void testCannotCreateMixedListInSkylark() throws Exception {
    update("mock", new Mock());
    checkEvalError(
        "Incompatible types in list: found a int but the previous elements were strings",
        "[mock.string(), 1, 2]");
  }

  @Test
  public void testCannotConcatListInSkylarkWithDifferentGenericTypes() throws Exception {
    update("mock", new Mock());
    checkEvalError("cannot concatenate list of string with list of int",
        "mock.string_list() + [1, 2]");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testConcatEmptyListWithNonEmptyWorks() throws Exception {
    eval("l = [] + ['a', 'b']");
    assertThat((Iterable<Object>) lookup("l")).containsExactly("a", "b").inOrder();
  }

  @Test
  public void testFormatStringWithTuple() throws Exception {
    eval("v = '%s%s' % ('a', 1)");
    assertEquals("a1", lookup("v"));
  }

  @Test
  public void testSingletonTuple() throws Exception {
    eval("v = (1,)");
    assertEquals("(1,)", lookup("v").toString());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testDirFindsClassObjectFields() throws Exception {
    update("mock", new MockClassObject());
    eval("v = dir(mock)");
    assertThat((Iterable<String>) lookup("v")).containsExactly("field", "nset").inOrder();
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testDirFindsJavaObjectStructFieldsAndMethods() throws Exception {
    update("mock", new Mock());
    eval("v = dir(mock)");
    assertThat((Iterable<String>) lookup("v")).containsExactly("function", "is_empty",
        "nullfunc_failing", "nullfunc_working", "return_mutable", "string", "string_list",
        "struct_field", "value_of", "voidfunc").inOrder();
  }

  @Test
  public void testPrint() throws Exception {
    setFailFast(false);
    eval("print('hello')");
    assertContainsEvent("hello");
    eval("print('a', 'b')");
    assertContainsEvent("a b");
    eval("print('a', 'b', sep='x')");
    assertContainsEvent("axb");
  }

  @Test
  public void testPrintBadKwargs() throws Exception {
    checkEvalError(
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
  public void testInFail() throws Exception {
    checkEvalError("in operator only works on strings if the left operand is also a string",
        "1 in '123'");
    checkEvalError("in operator only works on lists, tuples, sets, dicts and strings",
        "'a' in 1");
  }

  @Override
  @Test
  public void testCompareStringInt() throws Exception {
    checkEvalError("Cannot compare string with int", "'a' >= 1");
  }

  @Override
  @Test
  public void testListComprehensionsMultipleVariablesFail() throws Exception {
    checkEvalError("lvalue has length 3, but rvalue has has length 2",
        "def foo (): return [x + y for x, y, z in [(1, 2), (3, 4)]]",
        "foo()");

    checkEvalError("type 'int' is not a collection",
        "def bar (): return [x + y for x, y in (1, 2)]",
        "bar()");

    checkEvalError("lvalue has length 3, but rvalue has has length 2",
        "[x + y for x, y, z in [(1, 2), (3, 4)]]");

    // can't reuse the same local variable twice(!)
    checkEvalError("ERROR 1:1: Variable x is read only", "[x + y for x, y in (1, 2)]");

    checkEvalError("type 'int' is not a collection", "[x2 + y2 for x2, y2 in (1, 2)]");
  }

  @Override
  @Test
  public void testNotCallInt() throws Exception {
    eval("sum = 123456");
    assertEquals(123456, lookup("sum"));
    checkEvalError("'int' object is not callable", "sum(1, 2, 3, 4, 5, 6)");
    assertEquals(123456, eval("sum"));
  }

  @Test
  public void testConditionalExpressionAtToplevel() throws Exception {
    eval("x = 1 if 2 else 3");
    assertEquals(1, lookup("x"));
  }

  @Test
  public void testConditionalExpressionInFunction() throws Exception {
    eval("def foo(a, b, c): return a+b if c else a-b\n");
    assertEquals(18, eval("foo(23, 5, 0)"));
  }
}
