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

import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.ArrayList;
import java.util.List;
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
    exec(
        "def func(a,b,c):", //
        "  a = 1",
        "  b = a\n");
    StarlarkFunction stmt = (StarlarkFunction) lookup("func");
    assertThat(stmt).isNotNull();
    assertThat(stmt.getName()).isEqualTo("func");
    assertThat(stmt.getSignature().numMandatoryPositionals()).isEqualTo(3);
    assertThat(stmt.getStatements()).hasSize(2);
  }

  @Test
  public void testFunctionDefDuplicateArguments() throws Exception {
    // TODO(adonovan): move to ParserTest.
    ParserInput input =
        ParserInput.fromLines(
            "def func(a,b,a):", //
            "  a = 1\n");
    StarlarkFile file = StarlarkFile.parse(input);
    MoreAsserts.assertContainsEvent(
        file.errors(), "duplicate parameter name in function definition");
  }

  @Test
  public void testFunctionDefCallOuterFunc() throws Exception {
    List<Object> params = new ArrayList<>();
    createOuterFunction(params);
    exec(
        "def func(a):", //
        "  outer_func(a)",
        "func(1)",
        "func(2)");
    assertThat(params).containsExactly(1, 2).inOrder();
  }

  private void createOuterFunction(final List<Object> params) throws Exception {
    StarlarkCallable outerFunc =
        new StarlarkCallable() {
          @Override
          public String getName() {
            return "outer_func";
          }

          @Override
          public NoneType call(
              StarlarkThread thread,
              FuncallExpression call,
              Tuple<Object> args,
              Dict<String, Object> kwargs)
              throws EvalException {
            params.addAll(args);
            return Starlark.NONE;
          }
        };
    update("outer_func", outerFunc);
  }

  @Test
  public void testFunctionDefNoEffectOutsideScope() throws Exception {
    update("a", 1);
    exec(
        "def func():", //
        "  a = 2",
        "func()\n");
    assertThat(lookup("a")).isEqualTo(1);
  }

  @Test
  public void testFunctionDefGlobalVaribleReadInFunction() throws Exception {
    exec(
        "a = 1", //
        "def func():",
        "  b = a",
        "  return b",
        "c = func()\n");
    assertThat(lookup("c")).isEqualTo(1);
  }

  @Test
  public void testFunctionDefLocalGlobalScope() throws Exception {
    exec(
        "a = 1", //
        "def func():",
        "  a = 2",
        "  b = a",
        "  return b",
        "c = func()\n");
    assertThat(lookup("c")).isEqualTo(2);
  }

  @Test
  public void testFunctionDefLocalVariableReferencedBeforeAssignment() throws Exception {
    checkEvalErrorContains(
        "local variable 'a' is referenced before assignment.",
        "a = 1",
        "def func():",
        "  b = a",
        "  a = 2",
        "  return b",
        "c = func()\n");
  }

  @Test
  public void testFunctionDefLocalVariableReferencedInCallBeforeAssignment() throws Exception {
    checkEvalErrorContains(
        "local variable 'a' is referenced before assignment.",
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
    exec(
        "a = 1", //
        "def func():",
        "  a = 2",
        "  b = a",
        "  a = 3",
        "  return b",
        "c = func()\n");
    assertThat(lookup("c")).isEqualTo(2);
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testSkylarkGlobalComprehensionIsAllowed() throws Exception {
    exec("a = [i for i in [1, 2, 3]]\n");
    assertThat((Iterable<Object>) lookup("a")).containsExactly(1, 2, 3).inOrder();
  }

  @Test
  public void testFunctionReturn() throws Exception {
    exec(
        "def func():", //
        "  return 2",
        "b = func()\n");
    assertThat(lookup("b")).isEqualTo(2);
  }

  @Test
  public void testFunctionReturnFromALoop() throws Exception {
    exec(
        "def func():", //
        "  for i in [1, 2, 3, 4, 5]:",
        "    return i",
        "b = func()\n");
    assertThat(lookup("b")).isEqualTo(1);
  }

  @Test
  public void testFunctionExecutesProperly() throws Exception {
    exec(
        "def func(a):",
        "  b = 1",
        "  if a:",
        "    b = 2",
        "  return b",
        "c = func(0)",
        "d = func(1)\n");
    assertThat(lookup("c")).isEqualTo(1);
    assertThat(lookup("d")).isEqualTo(2);
  }

  @Test
  public void testFunctionCallFromFunction() throws Exception {
    final List<Object> params = new ArrayList<>();
    createOuterFunction(params);
    exec(
        "def func2(a):",
        "  outer_func(a)",
        "def func1(b):",
        "  func2(b)",
        "func1(1)",
        "func1(2)\n");
    assertThat(params).containsExactly(1, 2).inOrder();
  }

  @Test
  public void testFunctionCallFromFunctionReadGlobalVar() throws Exception {
    exec(
        "a = 1", //
        "def func2():",
        "  return a",
        "def func1():",
        "  return func2()",
        "b = func1()\n");
    assertThat(lookup("b")).isEqualTo(1);
  }

  @Test
  public void testFunctionParamCanShadowGlobalVarAfterGlobalVarIsRead() throws Exception {
    exec(
        "a = 1",
        "def func2(a):",
        "  return 0",
        "def func1():",
        "  dummy = a",
        "  return func2(2)",
        "b = func1()\n");
    assertThat(lookup("b")).isEqualTo(0);
  }

  @Test
  public void testSingleLineFunction() throws Exception {
    exec(
        "def func(): return 'a'", //
        "s = func()\n");
    assertThat(lookup("s")).isEqualTo("a");
  }

  @Test
  public void testFunctionReturnsDictionary() throws Exception {
    exec(
        "def func(): return {'a' : 1}", //
        "d = func()",
        "a = d['a']\n");
    assertThat(lookup("a")).isEqualTo(1);
  }

  @Test
  public void testFunctionReturnsList() throws Exception {
    exec(
        "def func(): return [1, 2, 3]", //
        "d = func()",
        "a = d[1]\n");
    assertThat(lookup("a")).isEqualTo(2);
  }

  @Test
  public void testFunctionNameAliasing() throws Exception {
    exec(
        "def func(a):", //
        "  return a + 1",
        "alias = func",
        "r = alias(1)");
    assertThat(lookup("r")).isEqualTo(2);
  }

  @Test
  public void testCallingFunctionsWithMixedModeArgs() throws Exception {
    exec(
        "def func(a, b, c):", //
        "  return a + b + c",
        "v = func(1, c = 2, b = 3)");
    assertThat(lookup("v")).isEqualTo(6);
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
    exec(
        functionWithOptionalArgs(),
        "v1 = func('1', 1, 1)",
        "v2 = func(b = 2, a = '2', c = 2)",
        "v3 = func('3')",
        "v4 = func('4', c = 1)\n");
    assertThat(lookup("v1")).isEqualTo("1abc");
    assertThat(lookup("v2")).isEqualTo("2abc");
    assertThat(lookup("v3")).isEqualTo("3a");
    assertThat(lookup("v4")).isEqualTo("4ac");
  }

  @Test
  public void testDefaultArguments() throws Exception {
    exec(
        "def func(a, b = 'b', c = 'c'):",
        "  return a + b + c",
        "v1 = func('a', 'x', 'y')",
        "v2 = func(b = 'x', a = 'a', c = 'y')",
        "v3 = func('a')",
        "v4 = func('a', c = 'y')\n");
    assertThat(lookup("v1")).isEqualTo("axy");
    assertThat(lookup("v2")).isEqualTo("axy");
    assertThat(lookup("v3")).isEqualTo("abc");
    assertThat(lookup("v4")).isEqualTo("aby");
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
  public void testArgsIsNotIterable() throws Exception {
    checkEvalError(
        "argument after * must be an iterable, not int",
        "def func1(a, b): return a + b",
        "func1('a', *42)");

    checkEvalError(
        "argument after * must be an iterable, not string",
        "def func2(a, b): return a + b",
        "func2('a', *'str')");
  }

  @Test
  public void testKeywordOnlyIsForbidden() throws Exception {
    checkEvalErrorContains("forbidden", "def foo(a, b, *, c): return a + b + c");
  }

  @Test
  public void testParamAfterStarArgs() throws Exception {
    checkEvalErrorContains("forbidden", "def foo(a, *b, c): return a");
  }

  @Test
  public void testKwargsBadKey() throws Exception {
    checkEvalError(
        "keywords must be strings, not int", //
        "def func(a, b): return a + b",
        "func('a', **{3: 1})");
  }

  @Test
  public void testKwargsIsNotDict() throws Exception {
    checkEvalError(
        "argument after ** must be a dict, not int",
        "def func(a, b): return a + b",
        "func('a', **42)");
  }

  @Test
  public void testKwargsCollision() throws Exception {
    checkEvalError(
        "func(a, b) got multiple values for parameter 'b'",
        "def func(a, b): return a + b",
        "func('a', 'b', **{'b': 'foo'})");
  }

  @Test
  public void testKwargsCollisionWithNamed() throws Exception {
    checkEvalError(
        "func(a, b) got multiple values for parameter 'b'",
        "def func(a, b): return a + b",
        "func('a', b = 'b', **{'b': 'foo'})");
  }

  @Test
  public void testDefaultArguments2() throws Exception {
    exec(
        "a = 2",
        "def foo(x=a): return x",
        "def bar():",
        "  a = 3",
        "  return foo()",
        "v = bar()\n");
    assertThat(lookup("v")).isEqualTo(2);
  }

  @Test
  public void testMixingPositionalOptional() throws Exception {
    exec(
        "def f(name, value = '', optional = ''):", //
        "  return value",
        "v = f('name', 'value')");
    assertThat(lookup("v")).isEqualTo("value");
  }

  @Test
  public void testStarArg() throws Exception {
    exec(
        "def f(name, value = '1', optional = '2'): return name + value + optional",
        "v1 = f(*['name', 'value'])",
        "v2 = f('0', *['name', 'value'])",
        "v3 = f('0', optional = '3', *['b'])",
        "v4 = f(name='a', *[])\n");
    assertThat(lookup("v1")).isEqualTo("namevalue2");
    assertThat(lookup("v2")).isEqualTo("0namevalue");
    assertThat(lookup("v3")).isEqualTo("0b3");
    assertThat(lookup("v4")).isEqualTo("a12");
  }

  @Test
  public void testStarParam() throws Exception {
    exec(
        "def f(name, value = '1', optional = '2', *rest):",
        "  r = name + value + optional + '|'",
        "  for x in rest: r += x",
        "  return r",
        "v1 = f('a', 'b', 'c', 'd', 'e')",
        "v2 = f('a', optional='b', value='c')",
        "v3 = f('a')");
    assertThat(lookup("v1")).isEqualTo("abc|de");
    assertThat(lookup("v2")).isEqualTo("acb|");
    assertThat(lookup("v3")).isEqualTo("a12|");
  }
}
