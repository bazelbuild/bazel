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
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 * Evaluation tests with Skylark Environment.
 */
@RunWith(JUnit4.class)
public class SkylarkEvaluationTest extends EvaluationTest {

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

  private static final ImmutableMap<String, SkylarkType> MOCK_TYPES = ImmutableMap
      .<String, SkylarkType>of("mock", SkylarkType.UNKNOWN, "Mock", SkylarkType.UNKNOWN);

  @Override
  public void setUp() throws Exception {
    super.setUp();
    syntaxEvents = new EventCollectionApparatus(EventKind.ALL_EVENTS);
    env = new SkylarkEnvironment(syntaxEvents.collector());
    MethodLibrary.setupMethodEnvironment(env);
  }

  @Override
  public Environment singletonEnv(String id, Object value) {
    SkylarkEnvironment env = new SkylarkEnvironment(syntaxEvents.collector());
    env.update(id, value);
    return env;
  }

  @Test
  public void testSimpleIf() throws Exception {
    exec(parseFileForSkylark(
        "def foo():\n"
        + "  a = 0\n"
        + "  x = 0\n"
        + "  if x: a = 5\n"
        + "  return a\n"
        + "a = foo()"), env);
    assertEquals(0, env.lookup("a"));
  }

  @Test
  public void testIfPass() throws Exception {
    exec(parseFileForSkylark(
        "def foo():\n"
        + "  a = 1\n"
        + "  x = True\n"
        + "  if x: pass\n"
        + "  return a\n"
        + "a = foo()"), env);
    assertEquals(1, env.lookup("a"));
  }

  @Test
  public void testNestedIf() throws Exception {
    executeNestedIf(0, 0, env);
    assertEquals(0, env.lookup("x"));

    executeNestedIf(1, 0, env);
    assertEquals(3, env.lookup("x"));

    executeNestedIf(1, 1, env);
    assertEquals(5, env.lookup("x"));
  }

  private void executeNestedIf(int x, int y, Environment env) throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  x = " + x + "\n"
        + "  y = " + y + "\n"
        + "  a = 0\n"
        + "  b = 0\n"
        + "  if x:\n"
        + "    if y:\n"
        + "      a = 2\n"
        + "    b = 3\n"
        + "  return a + b\n"
        + "x = foo()");
    exec(input, env);
  }

  @Test
  public void testIfElse() throws Exception {
    executeIfElse("something", 2);
    executeIfElse("", 3);
  }

  private void executeIfElse(String y, int expectedA) throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  y = '" + y + "'\n"
        + "  x = 5\n"
        + "  if x:\n"
        + "    if y: a = 2\n"
        + "    else: a = 3\n"
        + "  return a\n"
        + "a = foo()");

    exec(input, env);
    assertEquals(expectedA, env.lookup("a"));
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
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  x = " + x + "\n"
        + "  y = " + y + "\n"
        + "  if x:\n"
        + "    return 1\n"
        + "  elif y:\n"
        + "    return 2\n"
        + "  else:\n"
        + "    return 3\n"
        + "v = foo()");
    exec(input, env);
    assertEquals(v, env.lookup("v"));
  }

  @Test
  public void testForOnList() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  s = ''\n"
        + "  for i in ['hello', ' ', 'world']:\n"
        + "    s = s + i\n"
        + "  return s\n"
        + "s = foo()\n");

    exec(input, env);
    assertEquals("hello world", env.lookup("s"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testForOnString() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  s = []\n"
        + "  for i in 'abc':\n"
        + "    s = s + [i]\n"
        + "  return s\n"
        + "s = foo()\n");

    exec(input, env);
    assertThat((Iterable<Object>) env.lookup("s")).containsExactly("a", "b", "c").inOrder();
  }

  @Test
  public void testForAssignmentList() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  d = ['a', 'b', 'c']\n"
        + "  s = ''\n"
        + "  for i in d:\n"
        + "    s = s + i\n"
        + "    d = ['d', 'e', 'f']\n"  // check that we use the old list
        + "  return s\n"
        + "s = foo()\n");

    exec(input, env);
    assertEquals("abc", env.lookup("s"));
  }

  @Test
  public void testForAssignmentDict() throws Exception {
    List<Statement> input = parseFileForSkylark(
          "def func():\n"
        + "  d = {'a' : 1, 'b' : 2, 'c' : 3}\n"
        + "  s = ''\n"
        + "  for i in d:\n"
        + "    s = s + i\n"
        + "    d = {'d' : 1, 'e' : 2, 'f' : 3}\n"
        + "  return s\n"
        + "s = func()");

    exec(input, env);
    assertEquals("abc", env.lookup("s"));
  }

  @Test
  public void testForNotIterable() throws Exception {
    env.update("mock", new Mock());
    List<Statement> input = parseFileForSkylark(
          "def func():\n"
        + "  for i in mock.value_of('1'): a = i\n"
        + "func()\n", MOCK_TYPES);
    checkEvalError(input, env, "type 'int' is not an iterable");
  }

  @Test
  public void testForOnDictionary() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  d = {1: 'a', 2: 'b', 3: 'c'}\n"
        + "  s = ''\n"
        + "  for i in d: s = s + d[i]\n"
        + "  return s\n"
        + "s = foo()");

    exec(input, env);
    assertEquals("abc", env.lookup("s"));
  }

  @Test
  public void testForLoopReuseVariable() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  s = ''\n"
        + "  for i in ['a', 'b']:\n"
        + "    for i in ['c', 'd']: s = s + i\n"
        + "  return s\n"
        + "s = foo()");

    exec(input, env);
    assertEquals("cdcd", env.lookup("s"));
  }

  @Test
  public void testForLoopMultipleVariables() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  s = ''\n"
        + "  for [i, j] in [[1, 2], [3, 4]]:\n"
        + "    s = s + str(i) + str(j) + '.'\n"
        + "  return s\n"
        + "s = foo()");

    exec(input, env);
    assertEquals("12.34.", env.lookup("s"));
  }

  @Test
  public void testNoneAssignment() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo(x=None):\n"
        + "  x = 1\n"
        + "  x = None\n"
        + "  return 2\n"
        + "s = foo()");

    exec(input, env);
    assertEquals(2, env.lookup("s"));
  }

  @Test
  public void testJavaCalls() throws Exception {
    env.update("mock", new Mock());
    List<Statement> input = parseFileForSkylark(
        "b = mock.is_empty('a')", MOCK_TYPES);
    exec(input, env);
    assertEquals(Boolean.FALSE, env.lookup("b"));
  }

  @Test
  public void testJavaCallsOnSubClass() throws Exception {
    env.update("mock", new MockSubClass());
    List<Statement> input = parseFileForSkylark(
        "b = mock.is_empty('a')", MOCK_TYPES);
    exec(input, env);
    assertEquals(Boolean.FALSE, env.lookup("b"));
  }

  @Test
  public void testJavaCallsOnInterface() throws Exception {
    env.update("mock", new MockSubClass());
    List<Statement> input = parseFileForSkylark(
        "b = mock.is_empty_interface('a')", MOCK_TYPES);
    exec(input, env);
    assertEquals(Boolean.FALSE, env.lookup("b"));
  }

  @Test
  public void testJavaCallsNotSkylarkCallable() throws Exception {
    env.update("mock", new Mock());
    List<Statement> input = parseFileForSkylark("mock.value()", MOCK_TYPES);
    checkEvalError(input, env, "No matching method found for value() in Mock");
  }

  @Test
  public void testJavaCallsNoMethod() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "s = 3.bad()");
    checkEvalError(input, env, "No matching method found for bad() in int");
  }

  @Test
  public void testJavaCallsNoMethodErrorMsg() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "s = 3.bad('a', 'b', 'c')");
    checkEvalError(input, env,
        "No matching method found for bad(string, string, string) in int");
  }

  @Test
  public void testJavaCallsMultipleMethod() throws Exception {
    env.update("mock", new MockMultipleMethodClass());
    List<Statement> input = parseFileForSkylark(
        "s = mock.method('string')", MOCK_TYPES);
    checkEvalError(input, env,
        "Multiple matching methods for method(string) in MockMultipleMethodClass");
  }

  @Test
  public void testJavaCallWithKwargs() throws Exception {
    List<Statement> input = parseFileForSkylark("comp = 3.compare_to(x = 4)");
    checkEvalError(input, env, "Keyword arguments are not allowed when calling a java method"
        + "\nwhile calling method 'compare_to' on object 3 of type int");
  }

  @Test
  public void testNoJavaCallsWithoutSkylark() throws Exception {
    List<Statement> input = parseFileForSkylark("s = 3.to_string()\n");
    checkEvalError(input, env, "No matching method found for to_string() in int");
  }

  @Test
  public void testNoJavaCallsIfClassNotAnnotated() throws Exception {
    env.update("mock", new MockSubClass());
    List<Statement> input = parseFileForSkylark(
        "b = mock.is_empty_class_not_annotated('a')", MOCK_TYPES);
    checkEvalError(input, env,
        "No matching method found for is_empty_class_not_annotated(string) in MockSubClass");
  }

  @Test
  public void testStructAccess() throws Exception {
    env.update("mock", new Mock());
    List<Statement> input = parseFileForSkylark(
        "v = mock.struct_field", MOCK_TYPES);
    exec(input, env);
    assertEquals("a", env.lookup("v"));
  }

  @Test
  public void testStructAccessAsFuncall() throws Exception {
    env.update("mock", new Mock());
    checkEvalError(parseFileForSkylark("v = mock.struct_field()", MOCK_TYPES), env,
        "No matching method found for struct_field() in Mock");
  }

  @Test
  public void testStructAccessOfMethod() throws Exception {
    env.update("mock", new Mock());
    checkEvalError(parseFileForSkylark(
        "v = mock.function", MOCK_TYPES), env, "Object of type 'Mock' has no field 'function'");
  }

  @Test
  public void testConditionalStructConcatenation() throws Exception {
    MethodLibrary.setupMethodEnvironment(env);
    exec(parseFileForSkylark(
          "def func():\n"
        + "  x = struct(a = 1, b = 2)\n"
        + "  if True:\n"
        + "    x += struct(c = 1, d = 2)\n"
        + "  return x\n"
        + "x = func()\n"), env);
    SkylarkClassObject x = (SkylarkClassObject) env.lookup("x");
    assertEquals(1, x.getValue("a"));
    assertEquals(2, x.getValue("b"));
    assertEquals(1, x.getValue("c"));
    assertEquals(2, x.getValue("d"));
  }

  @Test
  public void testJavaFunctionReturnsMutableObject() throws Exception {
    env.update("mock", new Mock());
    List<Statement> input = parseFileForSkylark("mock.return_mutable()", MOCK_TYPES);
    checkEvalError(input, env, "Method 'return_mutable' returns a mutable object (type of Mock)");
  }

  @Test
  public void testJavaFunctionReturnsNullFails() throws Exception {
    env.update("mock", new Mock());
    List<Statement> input = parseFileForSkylark("mock.nullfunc_failing('abc', 1)", MOCK_TYPES);
    checkEvalError(input, env, "Method invocation returned None,"
        + " please contact Skylark developers: nullfunc_failing(\"abc\", 1)");
  }

  @Test
  public void testClassObjectAccess() throws Exception {
    env.update("mock", new MockClassObject());
    exec(parseFileForSkylark("v = mock.field", MOCK_TYPES), env);
    assertEquals("a", env.lookup("v"));
  }

  @Test
  public void testClassObjectCannotAccessNestedSet() throws Exception {
    env.update("mock", new MockClassObject());
    checkEvalError(parseFileForSkylark("v = mock.nset", MOCK_TYPES), env,
        "Type is not allowed in Skylark: EmptyNestedSet");
  }

  @Test
  public void testJavaFunctionReturnsNone() throws Exception {
    env.update("mock", new Mock());
    exec(parseFileForSkylark("v = mock.nullfunc_working()", MOCK_TYPES), env);
    assertSame(Environment.NONE, env.lookup("v"));
  }

  @Test
  public void testVoidJavaFunctionReturnsNone() throws Exception {
    env.update("mock", new Mock());
    exec(parseFileForSkylark("v = mock.voidfunc()", MOCK_TYPES), env);
    assertSame(Environment.NONE, env.lookup("v"));
  }

  @Test
  public void testAugmentedAssignment() throws Exception {
    exec(parseFileForSkylark(
        "def f1(x):\n"
        + "  x += 1\n"
        + "  return x\n"
        + "\n"
        + "foo = f1(41)\n"), env);
    assertEquals(42, env.lookup("foo"));
  }

  @Test
  public void testStaticDirectJavaCall() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "val = Mock.value_of('8')", MOCK_TYPES);

    env.update("Mock", Mock.class);
    exec(input, env);
    assertEquals(8, env.lookup("val"));
  }

  @Test
  public void testStaticDirectJavaCallMethodIsNonStatic() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "val = Mock.is_empty('a')", MOCK_TYPES);

    env.update("Mock", Mock.class);
    checkEvalError(input, env, "Method 'is_empty' is not static");
  }

  @Test
  public void testDictComprehensions_IterationOrder() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo():\n"
        + "  d = {x : x for x in ['c', 'a', 'b']}\n"
        + "  s = ''\n"
        + "  for a in d:\n"
        + "    s += a\n"
        + "  return s\n"
        + "s = foo()");
    exec(input, env);
    assertEquals("abc", env.lookup("s"));
  }

  @Test
  public void testStructCreation() throws Exception {
    exec(parseFileForSkylark("x = struct(a = 1, b = 2)"), env);
    assertThat(env.lookup("x")).isInstanceOf(ClassObject.class);
  }

  @Test
  public void testStructFields() throws Exception {
    exec(parseFileForSkylark("x = struct(a = 1, b = 2)"), env);
    ClassObject x = (ClassObject) env.lookup("x");
    assertEquals(1, x.getValue("a"));
    assertEquals(2, x.getValue("b"));
  }

  @Test
  public void testStructAccessingFieldsFromSkylark() throws Exception {
    exec(parseFileForSkylark(
          "x = struct(a = 1, b = 2)\n"
        + "x1 = x.a\n"
        + "x2 = x.b\n"), env);
    assertEquals(1, env.lookup("x1"));
    assertEquals(2, env.lookup("x2"));
  }

  @Test
  public void testStructAccessingUnknownField() throws Exception {
    checkEvalError(parseFileForSkylark(
          "x = struct(a = 1, b = 2)\n"
        + "y = x.c\n"), env, "Object of type 'struct' has no field 'c'");
  }

  @Test
  public void testStructAccessingFieldsWithArgs() throws Exception {
    checkEvalError(parseFileForSkylark(
          "x = struct(a = 1, b = 2)\n"
        + "x1 = x.a(1)\n"),
        env, "No matching method found for a(int) in struct");
  }

  @Test
  public void testStructPosArgs() throws Exception {
    checkEvalError(parseFileForSkylark(
          "x = struct(1, b = 2)\n"),
        env, "struct only supports keyword arguments");
  }

  @Test
  public void testStructConcatenationFieldNames() throws Exception {
    exec(parseFileForSkylark(
          "x = struct(a = 1, b = 2)\n"
        + "y = struct(c = 1, d = 2)\n"
        + "z = x + y\n"), env);
    SkylarkClassObject z = (SkylarkClassObject) env.lookup("z");
    assertEquals(ImmutableSet.of("a", "b", "c", "d"), z.getKeys());
  }

  @Test
  public void testStructConcatenationFieldValues() throws Exception {
    exec(parseFileForSkylark(
          "x = struct(a = 1, b = 2)\n"
        + "y = struct(c = 1, d = 2)\n"
        + "z = x + y\n"), env);
    SkylarkClassObject z = (SkylarkClassObject) env.lookup("z");
    assertEquals(1, z.getValue("a"));
    assertEquals(2, z.getValue("b"));
    assertEquals(1, z.getValue("c"));
    assertEquals(2, z.getValue("d"));
  }

  @Test
  public void testStructConcatenationCommonFields() throws Exception {
    checkEvalError(parseFileForSkylark(
          "x = struct(a = 1, b = 2)\n"
        + "y = struct(c = 1, a = 2)\n"
        + "z = x + y\n"), env, "Cannot concat structs with common field(s): a");
  }

  @Test
  public void testDotExpressionOnNonStructObject() throws Exception {
    checkEvalError(parseFileForSkylark(
          "x = 'a'.field"), env, "Object of type 'string' has no field 'field'");
  }

  @Test
  public void testPlusEqualsOnDict() throws Exception {
    MethodLibrary.setupMethodEnvironment(env);
    exec(parseFileForSkylark(
          "def func():\n"
        + "  d = {'a' : 1}\n"
        + "  d += {'b' : 2}\n"
        + "  return d\n"
        + "d = func()"), env);
    assertEquals(ImmutableMap.of("a", 1, "b", 2), env.lookup("d"));
  }

  @Test
  public void testDictAssignmentAsLValue() throws Exception {
    exec(parseFileForSkylark(
          "def func():\n"
        + "  d = {'a' : 1}\n"
        + "  d['b'] = 2\n"
        + "  return d\n"
        + "d = func()"), env);
    assertEquals(ImmutableMap.of("a", 1, "b", 2), env.lookup("d"));
  }

  @Test
  public void testDictAssignmentAsLValueNoSideEffects() throws Exception {
    MethodLibrary.setupMethodEnvironment(env);
    exec(parseFileForSkylark(
          "def func(d):\n"
        + "  d['b'] = 2\n"
        + "d = {'a' : 1}\n"
        + "func(d)"), env);
    assertEquals(ImmutableMap.of("a", 1), env.lookup("d"));
  }

  @Test
  public void testListIndexAsLValueAsLValue() throws Exception {
    checkEvalError(parseFileForSkylark(
          "def id(l):\n"
        + "  return l\n"
        + "def func():\n"
        + "  l = id([1])\n"
        + "  l[0] = 2\n"
        + "  return l\n"
        + "l = func()"), env, "unsupported operand type(s) for +: 'list' and 'dict'");
  }

  @Test
  public void testTopLevelDict() throws Exception {
    exec(parseFileForSkylark(
        "if 1:\n"
      + "  v = 'a'\n"
      + "else:\n"
      + "  v = 'b'"), env);
    assertEquals("a", env.lookup("v"));
  }

  @Test
  public void testUserFunctionKeywordArgs() throws Exception {
    exec(parseFileForSkylark(
        "def foo(a, b, c):\n"
      + "  return a + b + c\n"
      + "s = foo(1, c=2, b=3)"), env);
    assertEquals(6, env.lookup("s"));
  }

  @Test
  public void testNoneTrueFalseInSkylark() throws Exception {
    exec(parseFileForSkylark(
        "a = None\n"
      + "b = True\n"
      + "c = False"), env);
    assertSame(Environment.NONE, env.lookup("a"));
    assertTrue((Boolean) env.lookup("b"));
    assertFalse((Boolean) env.lookup("c"));
  }

  @Test
  public void testHasattr() throws Exception {
    exec(parseFileForSkylark(
        "s = struct(a=1)\n"
      + "x = hasattr(s, 'a')\n"
      + "y = hasattr(s, 'b')\n"), env);
    assertTrue((Boolean) env.lookup("x"));
    assertFalse((Boolean) env.lookup("y"));
  }

  @Test
  public void testHasattrMethods() throws Exception {
    env.update("mock", new Mock());
    ValidationEnvironment validEnv = SkylarkModules.getValidationEnvironment();
    validEnv.update("mock", SkylarkType.of(Mock.class), null);
    exec(Parser.parseFileForSkylark(createLexer(
          "a = hasattr(mock, 'struct_field')\n"
        + "b = hasattr(mock, 'function')\n"
        + "c = hasattr(mock, 'is_empty')\n"
        + "d = hasattr('str', 'replace')\n"
        + "e = hasattr(mock, 'other')\n"),
            syntaxEvents.reporter(), null, validEnv).statements, env);
    assertTrue((Boolean) env.lookup("a"));
    assertTrue((Boolean) env.lookup("b"));
    assertTrue((Boolean) env.lookup("c"));
    assertTrue((Boolean) env.lookup("d"));
    assertFalse((Boolean) env.lookup("e"));
  }

  @Test
  public void testGetattr() throws Exception {
    exec(parseFileForSkylark(
        "s = struct(a='val')\n"
      + "x = getattr(s, 'a')\n"
      + "y = getattr(s, 'b', 'def')\n"
      + "z = getattr(s, 'b', default = 'def')\n"
      + "w = getattr(s, 'a', default='ignored')"), env);
    assertEquals("val", env.lookup("x"));
    assertEquals("def", env.lookup("y"));
    assertEquals("def", env.lookup("z"));
    assertEquals("val", env.lookup("w"));
  }

  @Test
  public void testGetattrNoAttr() throws Exception {
    checkEvalError(parseFileForSkylark(
          "s = struct(a='val')\n"
        + "getattr(s, 'b')"),
        env, "Object of type 'struct' has no field 'b'");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testListAnTupleConcatenationDoesNotWorkInSkylark() throws Exception {
    checkEvalError(parseFileForSkylark("[1, 2] + (3, 4)"), env,
        "cannot concatenate lists and tuples");
  }

  @Test
  public void testCannotCreateMixedListInSkylark() throws Exception {
    env.update("mock", new Mock());
    checkEvalError(parseFileForSkylark("[mock.string(), 1, 2]", MOCK_TYPES), env,
        "Incompatible types in list: found a int but the previous elements were strings");
  }

  @Test
  public void testCannotConcatListInSkylarkWithDifferentGenericTypes() throws Exception {
    env.update("mock", new Mock());
    checkEvalError(parseFileForSkylark("mock.string_list() + [1, 2]", MOCK_TYPES), env,
        "cannot concatenate list of string with list of int");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testConcatEmptyListWithNonEmptyWorks() throws Exception {
    exec(parseFileForSkylark("l = [] + ['a', 'b']", MOCK_TYPES), env);
    assertThat((Iterable<Object>) env.lookup("l")).containsExactly("a", "b").inOrder();
  }

  @Test
  public void testFormatStringWithTuple() throws Exception {
    exec(parseFileForSkylark("v = '%s%s' % ('a', 1)"), env);
    assertEquals("a1", env.lookup("v"));
  }

  @Test
  public void testSingletonTuple() throws Exception {
    exec(parseFileForSkylark("v = (1,)"), env);
    assertEquals("(1,)", env.lookup("v").toString());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testDirFindsClassObjectFields() throws Exception {
    env.update("mock", new MockClassObject());
    exec(parseFileForSkylark("v = dir(mock)", MOCK_TYPES), env);
    assertThat((Iterable<String>) env.lookup("v")).containsExactly("field", "nset").inOrder();
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testDirFindsJavaObjectStructFieldsAndMethods() throws Exception {
    env.update("mock", new Mock());
    exec(parseFileForSkylark("v = dir(mock)", MOCK_TYPES), env);
    assertThat((Iterable<String>) env.lookup("v")).containsExactly("function", "is_empty",
        "nullfunc_failing", "nullfunc_working", "return_mutable", "string", "string_list",
        "struct_field", "value_of", "voidfunc").inOrder();
  }

  @Test
  public void testPrint() throws Exception {
    exec(parseFileForSkylark("print('hello')"), env);
    syntaxEvents.assertContainsEvent("hello");
    exec(parseFileForSkylark("print('a', 'b')"), env);
    syntaxEvents.assertContainsEvent("a b");
    exec(parseFileForSkylark("print('a', 'b', sep='x')"), env);
    syntaxEvents.assertContainsEvent("axb");
  }

  @Test
  public void testPrintBadKwargs() throws Exception {
    checkEvalError("print(end='x', other='y')", "unexpected keywords: '[end, other]'");
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
    assertFalse(((SkylarkList) x).isTuple());

    // tuple
    x = eval("(1,2)");
    assertThat((Iterable<Object>) x).containsExactly(1, 2).inOrder();
    assertTrue(((SkylarkList) x).isTuple());

    x = eval("(1,2) + (3,4)");
    assertThat((Iterable<Object>) x).containsExactly(1, 2, 3, 4).inOrder();
    assertTrue(((SkylarkList) x).isTuple());
  }

  @SuppressWarnings("unchecked")
  @Override
  @Test
  public void testListExprs() throws Exception {
    assertThat((Iterable<Object>) eval("[1, 2, 3]")).containsExactly(1, 2, 3).inOrder();
    assertThat((Iterable<Object>) eval("(1, 2, 3)")).containsExactly(1, 2, 3).inOrder();
  }

  @Override
  @Test
  public void testListConcatenation() throws Exception {}

  @Override
  @Test
  public void testKeywordArgs() {}

  @Test
  public void testConditionalExpressionAtToplevel() throws Exception {
    exec(parseFileForSkylark("x = 1 if 2 else 3"), env);
    assertEquals(1, env.lookup("x"));
  }

  @Test
  public void testConditionalExpressionInFunction() throws Exception {
    exec(parseFileForSkylark(
        "def foo(a, b, c):\n"
        + "  return a+b if c else a-b\n"
        + "x = foo(23, 5, 0)"), env);
    assertEquals(18, env.lookup("x"));
  }

  @Test
  public void testBadConditionalExpressionInFunction() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("def foo(a): return [] if a else 0\n");
    syntaxEvents.assertContainsEvent(
        "bad else case: int is incompatible with list at /some/file.txt:1:33");
    syntaxEvents.collector().clear();
  }
}
