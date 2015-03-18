// Copyright 2006-2015 Google Inc. All Rights Reserved.
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
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;

/**
 * Tests for {@link BaseFunction}.
 * This tests the argument processing by BaseFunction
 * between the outer call(posargs, kwargs, ast, env) and the inner call(args, ast, env).
 */
@RunWith(JUnit4.class)
public class BaseFunctionTest extends AbstractEvaluationTestCase {

  private Environment singletonEnv(String id, Object value) {
    Environment env = new Environment();
    env.update(id, value);
    return env;
  }

  /**
   * Handy implementation of {@link BaseFunction} that returns all its args as a list.
   * (We'd use SkylarkList.tuple, but it can't handle null.)
   */
  private static class TestingBaseFunction extends BaseFunction {
    TestingBaseFunction(FunctionSignature signature) {
      super("mixed", signature);
    }

    @Override
    public Object call(Object[] arguments, FuncallExpression ast, Environment env) {
      return Arrays.asList(arguments);
    }
  }

  private void checkBaseFunction(BaseFunction func, String callExpression, String expectedOutput)
      throws Exception {
    Environment env = singletonEnv(func.getName(), func);

    if (expectedOutput.charAt(0) == '[') { // a tuple => expected to pass
      assertEquals("Wrong output for " + callExpression,
          expectedOutput, eval(callExpression, env).toString());

    } else { // expected to fail with an exception
      try {
        eval(callExpression, env);
        fail();
      } catch (EvalException e) {
        assertEquals("Wrong exception for " + callExpression,
            expectedOutput, e.getMessage());
      }
    }
  }

  private static final String[] BASE_FUNCTION_EXPRESSIONS = {
    "mixed()",
    "mixed(1)",
    "mixed(1, 2)",
    "mixed(1, 2, 3)",
    "mixed(1, 2, wiz=3, quux=4)",
    "mixed(foo=1)",
    "mixed(foo=1, bar=2)",
    "mixed(bar=2, foo=1)",
    "mixed(2, foo=1)",
    "mixed(bar=2, foo=1, wiz=3)",
  };

  public void checkBaseFunctions(boolean onlyNamedArguments,
      String expectedSignature, String... expectedResults) throws Exception {
    BaseFunction func = new TestingBaseFunction(
        onlyNamedArguments
        ? FunctionSignature.namedOnly(1, "foo", "bar")
        : FunctionSignature.of(1, "foo", "bar"));

    assertThat(func.toString()).isEqualTo(expectedSignature);

    for (int i = 0; i < BASE_FUNCTION_EXPRESSIONS.length; i++) {
      String expr = BASE_FUNCTION_EXPRESSIONS[i];
      String expected = expectedResults[i];
      checkBaseFunction(func, expr, expected);
    }
  }

  @Test
  public void testNoSurplusArguments() throws Exception {
    checkBaseFunctions(false, "mixed(foo, bar = ?)",
        "insufficient arguments received by mixed(foo, bar = ?) (got 0, expected at least 1)",
        "[1, null]",
        "[1, 2]",
        "too many (3) positional arguments in call to mixed(foo, bar = ?)",
        "unexpected keywords 'quux', 'wiz' in call to mixed(foo, bar = ?)",
        "[1, null]",
        "[1, 2]",
        "[1, 2]",
        "argument 'foo' passed both by position and by name in call to mixed(foo, bar = ?)",
        "unexpected keyword 'wiz' in call to mixed(foo, bar = ?)");
  }

  @Test
  public void testOnlyNamedArguments() throws Exception {
    checkBaseFunctions(true, "mixed(*, foo = ?, bar)",
        "missing mandatory keyword arguments in call to mixed(*, foo = ?, bar)",
        "mixed(*, foo = ?, bar) does not accept positional arguments, but got 1",
        "mixed(*, foo = ?, bar) does not accept positional arguments, but got 2",
        "mixed(*, foo = ?, bar) does not accept positional arguments, but got 3",
        "mixed(*, foo = ?, bar) does not accept positional arguments, but got 2",
        "missing mandatory named-only argument 'bar' while calling mixed(*, foo = ?, bar)",
        "[1, 2]",
        "[1, 2]",
        "mixed(*, foo = ?, bar) does not accept positional arguments, but got 1",
        "unexpected keyword 'wiz' in call to mixed(*, foo = ?, bar)");
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testKwParam() throws Exception {
    Environment env = new SkylarkEnvironment(syntaxEvents.collector());
    exec(parseFileForSkylark(
        "def foo(a, b, c=3, d=4, *args, e, f, g=7, h=8, **kwargs):\n"
        + "  return (a, b, c, d, e, f, g, h, args, kwargs)\n"
        + "v1 = foo(1, 2, e=5, f=6)\n"
        + "v2 = foo(1, *['x', 'y', 'z', 't'], h=9, e=5, f=6, i=0)\n"
        + "def bar(**kwargs):\n"
        + "  return kwargs\n"
        + "b1 = bar(name='foo', type='jpg', version=42)\n"
        + "b2 = bar()\n"), env);

    assertThat(EvalUtils.prettyPrintValue(env.lookup("v1")))
        .isEqualTo("(1, 2, 3, 4, 5, 6, 7, 8, (), {})");
    assertThat(EvalUtils.prettyPrintValue(env.lookup("v2")))
        .isEqualTo("(1, \"x\", \"y\", \"z\", 5, 6, 7, 9, (\"t\",), {\"i\": 0})");

    // NB: the conversion to a TreeMap below ensures the keys are sorted.
    assertThat(EvalUtils.prettyPrintValue(
        new TreeMap<String, Object>((Map<String, Object>) env.lookup("b1"))))
        .isEqualTo("{\"name\": \"foo\", \"type\": \"jpg\", \"version\": 42}");
    assertThat(EvalUtils.prettyPrintValue(env.lookup("b2"))).isEqualTo("{}");
  }
}
