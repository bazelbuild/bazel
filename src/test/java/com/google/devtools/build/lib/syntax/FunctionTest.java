// Copyright 2014 Google Inc. All rights reserved.
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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.syntax.SkylarkType.SkylarkFunctionType;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A test class for functions and scoping.
 */
@RunWith(JUnit4.class)
public class FunctionTest extends AbstractEvaluationTestCase {

  private Environment env;

  private static final ImmutableMap<String, SkylarkType> OUTER_FUNC_TYPES =
      ImmutableMap.<String, SkylarkType>of(
          "outer_func", SkylarkFunctionType.of("outer_func", SkylarkType.NONE));

  @Before
  public void setUp() throws Exception {

    env = new SkylarkEnvironment(syntaxEvents.collector());
  }

  @Test
  public void testFunctionDef() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def func(a,b,c):\n"
        + "  a = 1\n"
        + "  b = a\n");

    exec(input, env);
    UserDefinedFunction stmt = (UserDefinedFunction) env.lookup("func");
    assertNotNull(stmt);
    assertEquals("func", stmt.getName());
    assertEquals(3, stmt.getFunctionSignature().getSignature().getShape().getMandatoryPositionals());
    assertThat(stmt.getStatements()).hasSize(2);
  }

  @Test
  public void testFunctionDefDuplicateArguments() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark(
        "def func(a,b,a):\n"
        + "  a = 1\n");
    syntaxEvents.assertContainsEvent("duplicate parameter name in function definition");
  }

  @Test
  public void testFunctionDefCallOuterFunc() throws Exception {
    final List<Object> params = new ArrayList<>();
    List<Statement> input = parseFileForSkylark(
        "def func(a):\n"
        + "  outer_func(a)\n"
        + "func(1)\n"
        + "func(2)",
        OUTER_FUNC_TYPES);
    createOuterFunction(env, params);
    exec(input, env);
    assertThat(params).containsExactly(1, 2).inOrder();
  }

  private void createOuterFunction(Environment env, final List<Object> params) {
    BaseFunction outerFunc = new BaseFunction("outer_func") {

      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        params.addAll(args);
        return Environment.NONE;
      }
    };
    env.update("outer_func", outerFunc);
  }

  @Test
  public void testFunctionDefNoEffectOutsideScope() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def func():\n"
        + "  a = 2\n"
        + "func()\n");
    env.update("a", 1);
    exec(input, env);
    assertEquals(1, env.lookup("a"));
  }

  @Test
  public void testFunctionDefGlobalVaribleReadInFunction() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "a = 1\n"
        + "def func():\n"
        + "  b = a\n"
        + "  return b\n"
        + "c = func()\n");
    exec(input, env);
    assertEquals(1, env.lookup("c"));
  }

  @Test
  public void testFunctionDefLocalGlobalScope() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "a = 1\n"
        + "def func():\n"
        + "  a = 2\n"
        + "  b = a\n"
        + "  return b\n"
        + "c = func()\n");
    exec(input, env);
    assertEquals(2, env.lookup("c"));
  }

  @Test
  public void testFunctionDefLocalVariableReferencedBeforeAssignment() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "a = 1\n"
        + "def func():\n"
        + "  b = a\n"
        + "  a = 2\n"
        + "  return b\n"
        + "c = func()\n");
    try {
      exec(input, env);
      fail();
    } catch (EvalException e) {
      assertThat(e.getMessage()).contains("Variable 'a' is referenced before assignment.");
    }
  }

  @Test
  public void testFunctionDefLocalVariableReferencedAfterAssignment() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "a = 1\n"
        + "def func():\n"
        + "  a = 2\n"
        + "  b = a\n"
        + "  a = 3\n"
        + "  return b\n"
        + "c = func()\n");
    exec(input, env);
    assertEquals(2, env.lookup("c"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testSkylarkGlobalComprehensionIsAllowed() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "a = [i for i in [1, 2, 3]]\n");
    exec(input, env);
    assertThat((Iterable<Object>) env.lookup("a")).containsExactly(1, 2, 3).inOrder();
  }

  @Test
  public void testFunctionReturn() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def func():\n"
        + "  return 2\n"
        + "b = func()\n");
    exec(input, env);
    assertEquals(2, env.lookup("b"));
  }

  @Test
  public void testFunctionReturnFromALoop() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def func():\n"
        + "  for i in [1, 2, 3, 4, 5]:\n"
        + "    return i\n"
        + "b = func()\n");
    exec(input, env);
    assertEquals(1, env.lookup("b"));
  }

  @Test
  public void testFunctionExecutesProperly() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def func(a):\n"
        + "  b = 1\n"
        + "  if a:\n"
        + "    b = 2\n"
        + "  return b\n"
        + "c = func(0)\n"
        + "d = func(1)\n");
    exec(input, env);
    assertEquals(1, env.lookup("c"));
    assertEquals(2, env.lookup("d"));
  }

  @Test
  public void testFunctionCallFromFunction() throws Exception {
    final List<Object> params = new ArrayList<>();
    List<Statement> input = parseFileForSkylark(
        "def func2(a):\n"
        + "  outer_func(a)\n"
        + "def func1(b):\n"
        + "  func2(b)\n"
        + "func1(1)\n"
        + "func1(2)\n",
        OUTER_FUNC_TYPES);
    createOuterFunction(env, params);
    exec(input, env);
    assertThat(params).containsExactly(1, 2).inOrder();
  }

  @Test
  public void testFunctionCallFromFunctionReadGlobalVar() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "a = 1\n"
        + "def func2():\n"
        + "  return a\n"
        + "def func1():\n"
        + "  return func2()\n"
        + "b = func1()\n");
    exec(input, env);
    assertEquals(1, env.lookup("b"));
  }

  @Test
  public void testSingleLineFunction() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def func(): return 'a'\n"
        + "s = func()\n");
    exec(input, env);
    assertEquals("a", env.lookup("s"));
  }

  @Test
  public void testFunctionReturnsDictionary() throws Exception {
    MethodLibrary.setupMethodEnvironment(env);
    List<Statement> input = parseFileForSkylark(
        "def func(): return {'a' : 1}\n"
        + "d = func()\n"
        + "a = d['a']\n");
    exec(input, env);
    assertEquals(1, env.lookup("a"));
  }

  @Test
  public void testFunctionReturnsList() throws Exception {
    MethodLibrary.setupMethodEnvironment(env);
    List<Statement> input = parseFileForSkylark(
        "def func(): return [1, 2, 3]\n"
        + "d = func()\n"
        + "a = d[1]\n");
    exec(input, env);
    assertEquals(2, env.lookup("a"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testFunctionListArgumentsAreImmutable() throws Exception {
    MethodLibrary.setupMethodEnvironment(env);
    List<Statement> input = parseFileForSkylark(
          "l = [1]\n"
        + "def func(l):\n"
        + "  l += [2]\n"
        + "func(l)");
    exec(input, env);
    assertThat((Iterable<Object>) env.lookup("l")).containsExactly(1);
  }

  @Test
  public void testFunctionDictArgumentsAreImmutable() throws Exception {
    MethodLibrary.setupMethodEnvironment(env);
    List<Statement> input = parseFileForSkylark(
          "d = {'a' : 1}\n"
        + "def func(d):\n"
        + "  d += {'a' : 2}\n"
        + "func(d)");
    exec(input, env);
    assertEquals(ImmutableMap.of("a", 1), env.lookup("d"));
  }

  @Test
  public void testFunctionNameAliasing() throws Exception {
    List<Statement> input = parseFileForSkylark(
          "def func(a):\n"
        + "  return a + 1\n"
        + "alias = func\n"
        + "r = alias(1)");
    exec(input, env);
    assertEquals(2, env.lookup("r"));
  }

  @Test
  public void testCallingFunctionsWithMixedModeArgs() throws Exception {
    List<Statement> input = parseFileForSkylark(
          "def func(a, b, c):\n"
        + "  return a + b + c\n"
        + "v = func(1, c = 2, b = 3)");
    exec(input, env);
    assertEquals(6, env.lookup("v"));
  }

  private String functionWithOptionalArgs() {
    return "def func(a, b = None, c = None):\n"
         + "  r = a + 'a'\n"
         + "  if b:\n"
         + "    r += 'b'\n"
         + "  if c:\n"
         + "    r += 'c'\n"
         + "  return r\n";
  }

  @Test
  public void testWhichOptionalArgsAreDefinedForFunctions() throws Exception {
    List<Statement> input = parseFileForSkylark(
        functionWithOptionalArgs()
        + "v1 = func('1', 1, 1)\n"
        + "v2 = func(b = 2, a = '2', c = 2)\n"
        + "v3 = func('3')\n"
        + "v4 = func('4', c = 1)\n");
    exec(input, env);
    assertEquals("1abc", env.lookup("v1"));
    assertEquals("2abc", env.lookup("v2"));
    assertEquals("3a", env.lookup("v3"));
    assertEquals("4ac", env.lookup("v4"));
  }

  @Test
  public void testDefaultArguments() throws Exception {
    List<Statement> input = parseFileForSkylark(
          "def func(a, b = 'b', c = 'c'):\n"
        + "  return a + b + c\n"
        + "v1 = func('a', 'x', 'y')\n"
        + "v2 = func(b = 'x', a = 'a', c = 'y')\n"
        + "v3 = func('a')\n"
        + "v4 = func('a', c = 'y')\n");
    exec(input, env);
    assertEquals("axy", env.lookup("v1"));
    assertEquals("axy", env.lookup("v2"));
    assertEquals("abc", env.lookup("v3"));
    assertEquals("aby", env.lookup("v4"));
  }

  @Test
  public void testDefaultArgumentsInsufficientArgNum() throws Exception {
    checkError("insufficient arguments received by func(a, b = \"b\", c = \"c\") "
        + "(got 0, expected at least 1)",
        "def func(a, b = 'b', c = 'c'):",
        "  return a + b + c",
        "func()");
  }

  @Test
  public void testKwargs() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def foo(a, b = 'b', *, c, d = 'd'):\n"
      + "  return a + b + c + d\n"
      + "args = {'a': 'x', 'c': 'z'}\n"
      + "v1 = foo(**args)\n"
      + "v2 = foo('x', c = 'c', d = 'e', **{'b': 'y'})\n"
      + "v3 = foo(c = 'z', a = 'x', **{'b': 'y', 'd': 'f'})");
    exec(input, env);
    assertEquals("xbzd", env.lookup("v1"));
    assertEquals("xyce", env.lookup("v2"));
    assertEquals("xyzf", env.lookup("v3"));
    UserDefinedFunction foo = (UserDefinedFunction) env.lookup("foo");
    assertEquals("foo(a, b = \"b\", *, c, d = \"d\")", foo.toString());
  }

  @Test
  public void testKwargsBadKey() throws Exception {
    checkError("Keywords must be strings, not int",
        "def func(a, b):",
        "  return a + b",
        "func('a', **{3: 1})");
  }

  @Test
  public void testKwargsIsNotDict() throws Exception {
    checkError("Argument after ** must be a dictionary, not int",
        "def func(a, b):",
        "  return a + b",
        "func('a', **42)");
  }

  @Test
  public void testKwargsCollision() throws Exception {
    checkError("argument 'b' passed both by position and by name in call to func(a, b)",
        "def func(a, b):",
        "  return a + b",
        "func('a', 'b', **{'b': 'foo'})");
  }

  @Test
  public void testKwargsCollisionWithNamed() throws Exception {
    checkError("duplicate keyword 'b' in call to func",
        "def func(a, b):",
        "  return a + b",
        "func('a', b = 'b', **{'b': 'foo'})");
  }

  @Test
  public void testDefaultArguments2() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "a = 2\n"
        + "def foo(x=a): return x\n"
        + "def bar():\n"
        + "  a = 3\n"
        + "  return foo()\n"
        + "v = bar()\n");
    exec(input, env);
    assertEquals(2, env.lookup("v"));
  }

  @Test
  public void testMixingPositionalOptional() throws Exception {
    List<Statement> input = parseFileForSkylark(
                "def f(name, value = '', optional = ''): return value\n"
                + "v = f('name', 'value')\n");
    exec(input, env);
    assertEquals("value", env.lookup("v"));
  }

  @Test
  public void testStarArg() throws Exception {
    List<Statement> input = parseFileForSkylark(
                "def f(name, value = '1', optional = '2'): return name + value + optional\n"
                        + "v1 = f(*['name', 'value'])\n"
                        + "v2 = f('0', *['name', 'value'])\n"
                        + "v3 = f('0', *['b'], optional = '3')\n"
                + "v4 = f(*[],name='a')\n");
    exec(input, env);
    assertEquals("namevalue2", env.lookup("v1"));
    assertEquals("0namevalue", env.lookup("v2"));
    assertEquals("0b3", env.lookup("v3"));
    assertEquals("a12", env.lookup("v4"));
  }

  @Test
  public void testStarParam() throws Exception {
    List<Statement> input = parseFileForSkylark(
        "def f(name, value = '1', *rest, mandatory, optional = '2'):\n"
        + "  r = name + value + mandatory + optional + '|'\n"
        + "  for x in rest: r += x\n"
        + "  return r\n"
        + "v1 = f('a', 'b', mandatory = 'z')\n"
        + "v2 = f('a', 'b', 'c', 'd', mandatory = 'z')\n"
        + "v3 = f('a', *['b', 'c', 'd'], mandatory = 'y', optional = 'z')\n"
        + "v4 = f(*['a'], **{'value': 'b', 'mandatory': 'c'})\n"
        + "v5 = f('a', 'b', 'c', *['d', 'e'], mandatory = 'f', **{'optional': 'g'})\n");
    exec(input, env);
    assertEquals("abz2|", env.lookup("v1"));
    assertEquals("abz2|cd", env.lookup("v2"));
    assertEquals("abyz|cd", env.lookup("v3"));
    assertEquals("abc2|", env.lookup("v4"));
    assertEquals("abfg|cde", env.lookup("v5"));
  }

  private void checkError(String msg, String... lines)
      throws Exception {
    try {
      List<Statement> input = parseFileForSkylark(Joiner.on("\n").join(lines));
      exec(input, env);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage(msg);
    }
  }
}
