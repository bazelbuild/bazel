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
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test class for functions and scoping. */
@RunWith(JUnit4.class)
public final class FunctionTest extends EvaluationTestCase {

  @Test
  public void testDef() throws Exception {
    exec("def f(a, b=1, *args, c, d=2, **kwargs): pass");
    StarlarkFunction f = (StarlarkFunction) lookup("f");
    assertThat(f).isNotNull();
    assertThat(f.getName()).isEqualTo("f");
    assertThat(f.getParameterNames())
        .containsExactly("a", "b", "c", "d", "args", "kwargs")
        .inOrder();
    assertThat(f.hasVarargs()).isTrue();
    assertThat(f.hasKwargs()).isTrue();
    assertThat(getDefaults(f)).containsExactly(null, 1, null, 2, null, null).inOrder();

    // same, sans varargs
    exec("def g(a, b=1, *, c, d=2, **kwargs): pass");
    StarlarkFunction g = (StarlarkFunction) lookup("g");
    assertThat(g.getParameterNames()).containsExactly("a", "b", "c", "d", "kwargs").inOrder();
    assertThat(g.hasVarargs()).isFalse();
    assertThat(g.hasKwargs()).isTrue();
    assertThat(getDefaults(g)).containsExactly(null, 1, null, 2, null).inOrder();
  }

  private static List<Object> getDefaults(StarlarkFunction fn) {
    List<Object> defaults = new ArrayList<>();
    for (int i = 0; i < fn.getParameterNames().size(); i++) {
      defaults.add(fn.getDefaultValue(i));
    }
    return defaults;
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
              StarlarkThread thread, Tuple<Object> args, Dict<String, Object> kwargs)
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
    checkEvalError(
        "func() missing 1 required positional argument: a",
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
  public void testKeywordOnly() throws Exception {
    checkEvalError(
        "func() missing 1 required keyword-only argument: b", //
        "def func(a, *, b): pass",
        "func(5)");

    checkEvalError(
        "func() accepts no more than 1 positional argument but got 2",
        "def func(a, *, b): pass",
        "func(5, 6)");

    exec("def func(a, *, b, c = 'c'): return a + b + c");
    assertThat(eval("func('a', b = 'b')")).isEqualTo("abc");
    assertThat(eval("func('a', b = 'b', c = 'd')")).isEqualTo("abd");
  }

  @Test
  public void testStarArgsAndKeywordOnly() throws Exception {
    checkEvalError(
        "func() missing 1 required keyword-only argument: b",
        "def func(a, *args, b): pass",
        "func(5)");

    checkEvalError(
        "func() missing 1 required keyword-only argument: b",
        "def func(a, *args, b): pass",
        "func(5, 6)");

    exec("def func(a, *args, b, c = 'c'): return a + str(args) + b + c");
    assertThat(eval("func('a', b = 'b')")).isEqualTo("a()bc");
    assertThat(eval("func('a', b = 'b', c = 'd')")).isEqualTo("a()bd");
    assertThat(eval("func('a', 1, 2, b = 'b')")).isEqualTo("a(1, 2)bc");
    assertThat(eval("func('a', 1, 2, b = 'b', c = 'd')")).isEqualTo("a(1, 2)bd");
  }

  @Test
  public void testKeywordOnlyAfterStarArg() throws Exception {
    checkEvalError(
        "func() missing 1 required keyword-only argument: c",
        "def func(a, *b, c): pass",
        "func(5)");

    checkEvalError(
        "func() missing 1 required keyword-only argument: c",
        "def func(a, *b, c): pass",
        "func(5, 6, 7)");

    exec("def func(a, *b, c): return a + str(b) + c");
    assertThat(eval("func('a', c = 'c')")).isEqualTo("a()c");
    assertThat(eval("func('a', 1, c = 'c')")).isEqualTo("a(1,)c");
    assertThat(eval("func('a', 1, 2, c = 'c')")).isEqualTo("a(1, 2)c");
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
        "func() got multiple values for parameter 'b'",
        "def func(a, b): return a + b",
        "func('a', 'b', **{'b': 'foo'})");
  }

  @Test
  public void testKwargsCollisionWithNamed() throws Exception {
    checkEvalError(
        "func() got multiple values for parameter 'b'",
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

  @Test
  public void testKwParam() throws Exception {
    exec(
        "def foo(a, b, c=3, d=4, g=7, h=8, *args, **kwargs):\n"
            + "  return (a, b, c, d, g, h, args, kwargs)\n"
            + "v1 = foo(1, 2)\n"
            + "v2 = foo(1, h=9, i=0, *['x', 'y', 'z', 't'])\n"
            + "v3 = foo(1, i=0, *[2, 3, 4, 5, 6, 7, 8])\n"
            + "def bar(**kwargs):\n"
            + "  return kwargs\n"
            + "b1 = bar(name='foo', type='jpg', version=42).items()\n"
            + "b2 = bar()\n");

    assertThat(Starlark.repr(lookup("v1"))).isEqualTo("(1, 2, 3, 4, 7, 8, (), {})");
    assertThat(Starlark.repr(lookup("v2")))
        .isEqualTo("(1, \"x\", \"y\", \"z\", \"t\", 9, (), {\"i\": 0})");
    assertThat(Starlark.repr(lookup("v3"))).isEqualTo("(1, 2, 3, 4, 5, 6, (7, 8), {\"i\": 0})");
    assertThat(Starlark.repr(lookup("b1")))
        .isEqualTo("[(\"name\", \"foo\"), (\"type\", \"jpg\"), (\"version\", 42)]");
    assertThat(Starlark.repr(lookup("b2"))).isEqualTo("{}");
  }

  @Test
  public void testTrailingCommas() throws Exception {
    // Test that trailing commas are allowed in function definitions and calls
    // even after last *args or **kwargs expressions, like python3
    exec(
        "def f(*args, **kwargs): pass\n"
            + "v1 = f(1,)\n"
            + "v2 = f(*(1,2),)\n"
            + "v3 = f(a=1,)\n"
            + "v4 = f(**{\"a\": 1},)\n");

    assertThat(Starlark.repr(lookup("v1"))).isEqualTo("None");
    assertThat(Starlark.repr(lookup("v2"))).isEqualTo("None");
    assertThat(Starlark.repr(lookup("v3"))).isEqualTo("None");
    assertThat(Starlark.repr(lookup("v4"))).isEqualTo("None");
  }

  @Test
  public void testCalls() throws Exception {
    exec("def f(a, b = None): return a, b");

    assertThat(Starlark.repr(eval("f(1)"))).isEqualTo("(1, None)");
    assertThat(Starlark.repr(eval("f(1, 2)"))).isEqualTo("(1, 2)");
    assertThat(Starlark.repr(eval("f(a=1)"))).isEqualTo("(1, None)");
    assertThat(Starlark.repr(eval("f(a=1, b=2)"))).isEqualTo("(1, 2)");
    assertThat(Starlark.repr(eval("f(b=2, a=1)"))).isEqualTo("(1, 2)");

    checkEvalError(
        "f() missing 1 required positional argument: a", //
        "f()");
    checkEvalError(
        "f() accepts no more than 2 positional arguments but got 3", //
        "f(1, 2, 3)");
    checkEvalError(
        "f() got unexpected keyword arguments: c, d", //
        "f(1, 2, c=3, d=4)");
    checkEvalError(
        "f() missing 1 required positional argument: a", //
        "f(b=2)");
    checkEvalError(
        "f() missing 1 required positional argument: a", //
        "f(b=2)");
    checkEvalError(
        "f() got multiple values for parameter 'a'", //
        "f(2, a=1)");
    checkEvalError(
        "f() got unexpected keyword argument: c", //
        "f(b=2, a=1, c=3)");

    exec("def g(*, one, two, three): pass");
    checkEvalError(
        "g() got unexpected keyword argument: tree (did you mean 'three'?)", //
        "g(tree=3)");
    checkEvalError(
        "g() does not accept positional arguments, but got 3", //
        "g(1, 2 ,3)");
  }
}
