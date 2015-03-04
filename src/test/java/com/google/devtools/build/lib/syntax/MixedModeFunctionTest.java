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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;

/**
 * Tests for {@link MixedModeFunction}.
 */
@RunWith(JUnit4.class)
public class MixedModeFunctionTest extends AbstractEvaluationTestCase {

  private Environment singletonEnv(String id, Object value) {
    Environment env = new Environment();
    env.update(id, value);
    return env;
  }

  /**
   * Handy implementation of {@link MixedModeFunction} that just tuples up its args and returns
   * them.
   */
  private static class TestingMixedModeFunction extends MixedModeFunction {
    TestingMixedModeFunction(Iterable<String> parameters,
                             int numMandatoryParameters,
                             boolean onlyNamedArguments) {
      super("mixed", parameters, numMandatoryParameters, onlyNamedArguments);
    }
    @Override
    public Object call(Object[] namedParameters, FuncallExpression ast) {
      return Arrays.asList(namedParameters);
    }
  }

  private void checkMixedMode(Function func,
                              String callExpression,
                              String expectedOutput) throws Exception {
    Environment env = singletonEnv(func.getName(), func);

    if (expectedOutput.charAt(0) == '[') { // a tuple => expected to pass
      assertEquals(expectedOutput,
                   eval(callExpression, env).toString());
    } else { // expected to fail with an exception
      try {
        eval(callExpression, env);
        fail();
      } catch (EvalException e) {
        assertEquals(expectedOutput, e.getMessage());
      }
    }
  }

  private static final String[] mixedModeExpressions = {
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

  public void checkMixedModeFunctions(boolean onlyNamedArguments,
                                      String expectedSignature,
                                      String[] expectedResults)
      throws Exception {
    MixedModeFunction func =
        new TestingMixedModeFunction(ImmutableList.of("foo", "bar"), 1, onlyNamedArguments);

    assertEquals(expectedSignature, func.getSignature());

    for (int ii = 0; ii < mixedModeExpressions.length; ++ii) {
      String expr = mixedModeExpressions[ii];
      String expected = expectedResults[ii];
      checkMixedMode(func, expr, expected);
    }
  }

  @Test
  public void testNoSurplusArguments() throws Exception {
    checkMixedModeFunctions(false,
                            "mixed(foo, bar = null)",
                            new String[]
      {
        "mixed(foo, bar = null) received insufficient arguments",
        "[1, null]",
        "[1, 2]",
        "too many positional arguments in call to mixed(foo, bar = null)",
        "unexpected keywords 'quux', 'wiz' in call to mixed(foo, bar = null)",
        "[1, null]",
        "[1, 2]",
        "[1, 2]",
        "mixed(foo, bar = null) got multiple values for keyword"
        + " argument 'foo'",
        "unexpected keyword 'wiz' in call to mixed(foo, bar = null)",
      });
  }

  @Test
  public void testOnlyNamedArguments() throws Exception {
    checkMixedModeFunctions(true,
                            "mixed(foo, bar = null)",
                            new String[]
      {
        "mixed(foo, bar = null) received insufficient arguments",
        "mixed(foo, bar = null) does not accept positional arguments",
        "mixed(foo, bar = null) does not accept positional arguments",
        "mixed(foo, bar = null) does not accept positional arguments",
        "mixed(foo, bar = null) does not accept positional arguments",
        "[1, null]",
        "[1, 2]",
        "[1, 2]",
        "mixed(foo, bar = null) does not accept positional arguments",
        "unexpected keyword 'wiz' in call to mixed(foo, bar = null)",
      });
  }
}
