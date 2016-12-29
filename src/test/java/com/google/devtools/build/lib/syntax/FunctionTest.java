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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test class for functions and scoping.
 */
@RunWith(JUnit4.class)
public class FunctionTest extends EvaluationTestCase {

  @Test
  public void testFunctionDef() throws Exception {
    eval("def func(a,b,c):",
        "  a = 1",
        "  b = a\n");
    UserDefinedFunction stmt = (UserDefinedFunction) lookup("func");
    assertNotNull(stmt);
    assertThat(stmt.getName()).isEqualTo("func");
    assertThat(stmt.getFunctionSignature().getSignature().getShape().getMandatoryPositionals())
        .isEqualTo(3);
    assertThat(stmt.getStatements()).hasSize(2);
  }

  @Test
  public void testFunctionDefDuplicateArguments() throws Exception {
    setFailFast(false);
    parseFile("def func(a,b,a):",
        "  a = 1\n");
    assertContainsError("duplicate parameter name in function definition");
  }

  @Test
  public void testFunctionDefCallOuterFunc() throws Exception {
    List<Object> params = new ArrayList<>();
    createOuterFunction(params);
    eval("def func(a):",
        "  outer_func(a)",
        "func(1)",
        "func(2)");
    assertThat(params).containsExactly(1, 2).inOrder();
  }

  private void createOuterFunction(final List<Object> params) throws Exception {
    BaseFunction outerFunc = new BaseFunction("outer_func") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        params.addAll(args);
        return Runtime.NONE;
      }
    };
    update("outer_func", outerFunc);
  }

  @Test
  public void testFunctionDefNoEffectOutsideScope() throws Exception {
    update("a", 1);
    eval("def func():",
        "  a = 2",
        "func()\n");
    assertEquals(1, lookup("a"));
  }

  @Test
  public void testFunctionDefGlobalVaribleReadInFunction() throws Exception {
    eval("a = 1",
        "def func():",
        "  b = a",
        "  return b",
        "c = func()\n");
    assertEquals(1, lookup("c"));
  }

  @Test
  public void testFunctionDefLocalGlobalScope() throws Exception {
    eval("a = 1",
        "def func():",
        "  a = 2",
        "  b = a",
        "  return b",
        "c = func()\n");
    assertEquals(2, lookup("c"));
  }

  @Test
  public void testFunctionDefLocalVariableReferencedBeforeAssignment() throws Exception {
    checkEvalErrorContains("Variable 'a' is referenced before assignment.",
        "a = 1",
        "def func():",
        "  b = a",
        "  a = 2",
        "  return b",
        "c = func()\n");
  }

  @Test
  public void testFunctionDefLocalVariableReferencedInCallBeforeAssignment() throws Exception {
    checkEvalErrorContains("Variable 'a' is referenced before assignment.",
        "def dummy(x):",
        "  pass",
        "a = 1",
        "def func():",
        "  dummy(a)",
        "  a = 2",
        "func()\n");
  }

  @Test
  public void testFunctionDefLocalVariableReferencedAfterAssignment() throws Exception {
    eval("a = 1",
        "def func():",
        "  a = 2",
        "  b = a",
        "  a = 3",
        "  return b",
        "c = func()\n");
    assertEquals(2, lookup("c"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testSkylarkGlobalComprehensionIsAllowed() throws Exception {
    eval("a = [i for i in [1, 2, 3]]\n");
    assertThat((Iterable<Object>) lookup("a")).containsExactly(1, 2, 3).inOrder();
  }

  @Test
  public void testFunctionReturn() throws Exception {
    eval("def func():",
        "  return 2",
        "b = func()\n");
    assertEquals(2, lookup("b"));
  }

  @Test
  public void testFunctionReturnFromALoop() throws Exception {
    eval("def func():",
        "  for i in [1, 2, 3, 4, 5]:",
        "    return i",
        "b = func()\n");
    assertEquals(1, lookup("b"));
  }

  @Test
  public void testFunctionExecutesProperly() throws Exception {
    eval("def func(a):",
        "  b = 1",
        "  if a:",
        "    b = 2",
        "  return b",
        "c = func(0)",
        "d = func(1)\n");
    assertEquals(1, lookup("c"));
    assertEquals(2, lookup("d"));
  }

  @Test
  public void testFunctionCallFromFunction() throws Exception {
    final List<Object> params = new ArrayList<>();
    createOuterFunction(params);
    eval("def func2(a):",
        "  outer_func(a)",
        "def func1(b):",
        "  func2(b)",
        "func1(1)",
        "func1(2)\n");
    assertThat(params).containsExactly(1, 2).inOrder();
  }

  @Test
  public void testFunctionCallFromFunctionReadGlobalVar() throws Exception {
    eval("a = 1",
        "def func2():",
        "  return a",
        "def func1():",
        "  return func2()",
        "b = func1()\n");
    assertEquals(1, lookup("b"));
  }

  @Test
  public void testSingleLineFunction() throws Exception {
    eval("def func(): return 'a'",
        "s = func()\n");
    assertEquals("a", lookup("s"));
  }

  @Test
  public void testFunctionReturnsDictionary() throws Exception {
    eval("def func(): return {'a' : 1}",
        "d = func()",
        "a = d['a']\n");
    assertEquals(1, lookup("a"));
  }

  @Test
  public void testFunctionReturnsList() throws Exception {
    eval("def func(): return [1, 2, 3]",
        "d = func()",
        "a = d[1]\n");
    assertEquals(2, lookup("a"));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testFunctionListArgumentsAreImmutable() throws Exception {
    eval("l = [1]",
        "def func(l):",
        "  l += [2]",
        "func(l)");
    assertThat((Iterable<Object>) lookup("l")).containsExactly(1);
  }

  @Test
  public void testFunctionDictArgumentsAreImmutable() throws Exception {
    eval("d = {'a' : 1}",
        "def func(d):",
        "  d += {'a' : 2}",
        "func(d)");
    assertEquals(ImmutableMap.of("a", 1), lookup("d"));
  }

  @Test
  public void testFunctionNameAliasing() throws Exception {
    eval("def func(a):",
        "  return a + 1",
        "alias = func",
        "r = alias(1)");
    assertEquals(2, lookup("r"));
  }

  @Test
  public void testCallingFunctionsWithMixedModeArgs() throws Exception {
    eval("def func(a, b, c):",
        "  return a + b + c",
        "v = func(1, c = 2, b = 3)");
    assertEquals(6, lookup("v"));
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
    eval(functionWithOptionalArgs(),
        "v1 = func('1', 1, 1)",
        "v2 = func(b = 2, a = '2', c = 2)",
        "v3 = func('3')",
        "v4 = func('4', c = 1)\n");
    assertEquals("1abc", lookup("v1"));
    assertEquals("2abc", lookup("v2"));
    assertEquals("3a", lookup("v3"));
    assertEquals("4ac", lookup("v4"));
  }

  @Test
  public void testDefaultArguments() throws Exception {
    eval("def func(a, b = 'b', c = 'c'):",
        "  return a + b + c",
        "v1 = func('a', 'x', 'y')",
        "v2 = func(b = 'x', a = 'a', c = 'y')",
        "v3 = func('a')",
        "v4 = func('a', c = 'y')\n");
    assertEquals("axy", lookup("v1"));
    assertEquals("axy", lookup("v2"));
    assertEquals("abc", lookup("v3"));
    assertEquals("aby", lookup("v4"));
  }

  @Test
  public void testDefaultArgumentsInsufficientArgNum() throws Exception {
    checkEvalError("insufficient arguments received by func(a, b = \"b\", c = \"c\") "
        + "(got 0, expected at least 1)",
        "def func(a, b = 'b', c = 'c'):",
        "  return a + b + c",
        "func()");
  }

  @Test
  public void testKwargs() throws Exception {
    eval("def foo(a, b = 'b', *, c, d = 'd'):",
      "  return a + b + c + d",
      "args = {'a': 'x', 'c': 'z'}",
      "v1 = foo(**args)",
      "v2 = foo('x', c = 'c', d = 'e', **{'b': 'y'})",
      "v3 = foo(c = 'z', a = 'x', **{'b': 'y', 'd': 'f'})");
    assertEquals("xbzd", lookup("v1"));
    assertEquals("xyce", lookup("v2"));
    assertEquals("xyzf", lookup("v3"));
    UserDefinedFunction foo = (UserDefinedFunction) lookup("foo");
    assertEquals("foo(a, b = \"b\", *, c, d = \"d\")", foo.toString());
  }

  @Test
  public void testKwargsBadKey() throws Exception {
    checkEvalError(
        "keywords must be strings, not int", "def func(a, b): return a + b", "func('a', **{3: 1})");
  }

  @Test
  public void testKwargsIsNotDict() throws Exception {
    checkEvalError(
        "argument after ** must be a dictionary, not int",
        "def func(a, b): return a + b",
        "func('a', **42)");
  }

  @Test
  public void testKwargsCollision() throws Exception {
    checkEvalError("argument 'b' passed both by position and by name in call to func(a, b)",
        "def func(a, b): return a + b",
        "func('a', 'b', **{'b': 'foo'})");
  }

  @Test
  public void testKwargsCollisionWithNamed() throws Exception {
    checkEvalError("duplicate keyword 'b' in call to func",
        "def func(a, b): return a + b",
        "func('a', b = 'b', **{'b': 'foo'})");
  }

  @Test
  public void testDefaultArguments2() throws Exception {
    eval("a = 2",
        "def foo(x=a): return x",
        "def bar():",
        "  a = 3",
        "  return foo()",
        "v = bar()\n");
    assertEquals(2, lookup("v"));
  }

  @Test
  public void testMixingPositionalOptional() throws Exception {
    eval("def f(name, value = '', optional = ''): return value",
        "v = f('name', 'value')\n");
    assertEquals("value", lookup("v"));
  }

  @Test
  public void testStarArg() throws Exception {
    eval("def f(name, value = '1', optional = '2'): return name + value + optional",
        "v1 = f(*['name', 'value'])",
        "v2 = f('0', *['name', 'value'])",
        "v3 = f('0', *['b'], optional = '3')",
        "v4 = f(*[],name='a')\n");
    assertEquals("namevalue2", lookup("v1"));
    assertEquals("0namevalue", lookup("v2"));
    assertEquals("0b3", lookup("v3"));
    assertEquals("a12", lookup("v4"));
  }

  @Test
  public void testStarParam() throws Exception {
    eval("def f(name, value = '1', *rest, mandatory, optional = '2'):",
        "  r = name + value + mandatory + optional + '|'",
        "  for x in rest: r += x",
        "  return r",
        "v1 = f('a', 'b', mandatory = 'z')",
        "v2 = f('a', 'b', 'c', 'd', mandatory = 'z')",
        "v3 = f('a', *['b', 'c', 'd'], mandatory = 'y', optional = 'z')",
        "v4 = f(*['a'], **{'value': 'b', 'mandatory': 'c'})",
        "v5 = f('a', 'b', 'c', *['d', 'e'], mandatory = 'f', **{'optional': 'g'})\n");
    assertEquals("abz2|", lookup("v1"));
    assertEquals("abz2|cd", lookup("v2"));
    assertEquals("abyz|cd", lookup("v3"));
    assertEquals("abc2|", lookup("v4"));
    assertEquals("abfg|cde", lookup("v5"));
  }
}
