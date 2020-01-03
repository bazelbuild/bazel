// Copyright 2006 The Bazel Authors. All Rights Reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of the argument processing of {@code Starlark.matchSignature}. */
// TODO(adonovan): rename.
@RunWith(JUnit4.class)
public class BaseFunctionTest extends EvaluationTestCase {

  private void checkFunction(StarlarkCallable fn, String callExpression, String expectedOutput)
      throws Exception {
    initialize();
    update(fn.getName(), fn);

    if (expectedOutput.charAt(0) == '[') { // a tuple => expected to pass
      assertWithMessage("Wrong output for " + callExpression)
          .that(eval(callExpression).toString())
          .isEqualTo(expectedOutput);

    } else { // expected to fail with an exception
      EvalException e = assertThrows(EvalException.class, () -> eval(callExpression));
      assertWithMessage("Wrong exception for " + callExpression)
          .that(e.getMessage())
          .isEqualTo(expectedOutput);
    }
  }

  // TODO(adonovan): redesign this test so that inputs and expected outputs are adjacent.
  private static final String[] BASE_FUNCTION_EXPRESSIONS = {
    "mixed()",
    "mixed(1)",
    "mixed(1, 2)",
    "mixed(1, 2, 3)",
    "mixed(1, 2, wiz=3, quux=4)",
    "mixed(foo=1)",
    "mixed(bar=2)",
    "mixed(foo=1, bar=2)",
    "mixed(bar=2, foo=1)",
    "mixed(2, foo=1)",
    "mixed(bar=2, foo=1, wiz=3)",
  };

  public void checkFunctions(
      boolean onlyNamedArguments, String expectedSignature, String... expectedResults)
      throws Exception {
    FunctionSignature sig =
        onlyNamedArguments
            ? FunctionSignature.namedOnly(1, "foo", "bar")
            : FunctionSignature.of(1, "foo", "bar");
    // This test uses BaseFunction only for its 'repr' implementation.
    // The meat of this test exercises only StarlarkCallable.
    // TODO(adonovan): make it easier to get repr correct and eliminate BaseFunction here.
    BaseFunction func =
        new BaseFunction() {
          @Override
          public String getName() {
            return "mixed";
          }

          @Override
          public FunctionSignature getSignature() {
            return sig;
          }

          @Override
          public Object fastcall(
              StarlarkThread thread, Location loc, Object[] positional, Object[] named)
              throws EvalException {
            Object[] arguments =
                Starlark.matchSignature(
                    sig, this, /*defaults=*/ null, thread.mutability(), positional, named);
            return Arrays.asList(arguments);
          }
        };

    assertThat(func.toString()).isEqualTo(expectedSignature);

    for (int i = 0; i < BASE_FUNCTION_EXPRESSIONS.length; i++) {
      String expr = BASE_FUNCTION_EXPRESSIONS[i];
      String expected = expectedResults[i];
      checkFunction(func, expr, expected);
    }
  }

  @Test
  public void testNoSurplusArguments() throws Exception {
    checkFunctions(
        false,
        "mixed(foo, bar = ?)",
        "insufficient arguments received by mixed(foo, bar = ?) (got 0, expected at least 1)",
        "[1, null]",
        "[1, 2]",
        "too many (3) positional arguments in call to mixed(foo, bar = ?)",
        "unexpected keywords 'quux', 'wiz' in call to mixed(foo, bar = ?)",
        "[1, null]",
        "missing mandatory positional argument 'foo' while calling mixed(foo, bar = ?)",
        "[1, 2]",
        "[1, 2]",
        "mixed(foo, bar = ?) got multiple values for parameter 'foo'",
        "unexpected keyword 'wiz' in call to mixed(foo, bar = ?)");
  }

  @Test
  public void testOnlyNamedArguments() throws Exception {
    checkFunctions(
        true,
        "mixed(*, foo, bar = ?)",
        "missing mandatory keyword arguments in call to mixed(*, foo, bar = ?)",
        "mixed(*, foo, bar = ?) does not accept positional arguments, but got 1",
        "mixed(*, foo, bar = ?) does not accept positional arguments, but got 2",
        "mixed(*, foo, bar = ?) does not accept positional arguments, but got 3",
        "mixed(*, foo, bar = ?) does not accept positional arguments, but got 2",
        "[1, null, null]",
        "missing mandatory named-only argument 'foo' while calling mixed(*, foo, bar = ?)",
        "[1, 2, null]",
        "[1, 2, null]",
        "mixed(*, foo, bar = ?) does not accept positional arguments, but got 1",
        "unexpected keyword 'wiz' in call to mixed(*, foo, bar = ?)");
  }

  @SuppressWarnings("unchecked")
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
            + "b1 = bar(name='foo', type='jpg', version=42)\n"
            + "b2 = bar()\n");

    assertThat(Starlark.repr(lookup("v1"))).isEqualTo("(1, 2, 3, 4, 7, 8, (), {})");
    assertThat(Starlark.repr(lookup("v2")))
        .isEqualTo("(1, \"x\", \"y\", \"z\", \"t\", 9, (), {\"i\": 0})");
    assertThat(Starlark.repr(lookup("v3"))).isEqualTo("(1, 2, 3, 4, 5, 6, (7, 8), {\"i\": 0})");

    // NB: the conversion to a TreeMap below ensures the keys are sorted.
    assertThat(Starlark.repr(new TreeMap<String, Object>((Map<String, Object>) lookup("b1"))))
        .isEqualTo("{\"name\": \"foo\", \"type\": \"jpg\", \"version\": 42}");
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
}
