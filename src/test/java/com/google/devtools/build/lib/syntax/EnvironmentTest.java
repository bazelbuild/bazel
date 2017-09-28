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
import static org.junit.Assert.fail;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests of Environment.
 */
@RunWith(JUnit4.class)
public class EnvironmentTest extends EvaluationTestCase {

  @Override
  public Environment newEnvironment() {
    return newBuildEnvironment();
  }

  // Test the API directly
  @Test
  public void testLookupAndUpdate() throws Exception {
    assertThat(lookup("foo")).isNull();
    update("foo", "bar");
    assertThat(lookup("foo")).isEqualTo("bar");
  }

  @Test
  public void testHasVariable() throws Exception {
    assertThat(getEnvironment().hasVariable("VERSION")).isFalse();
    update("VERSION", 42);
    assertThat(getEnvironment().hasVariable("VERSION")).isTrue();
  }

  @Test
  public void testDoubleUpdateSucceeds() throws Exception {
    update("VERSION", 42);
    assertThat(lookup("VERSION")).isEqualTo(42);
    update("VERSION", 43);
    assertThat(lookup("VERSION")).isEqualTo(43);
  }

  // Test assign through interpreter, lookup through API:
  @Test
  public void testAssign() throws Exception {
    assertThat(lookup("foo")).isNull();
    eval("foo = 'bar'");
    assertThat(lookup("foo")).isEqualTo("bar");
  }

  // Test update through API, reference through interpreter:
  @Test
  public void testReference() throws Exception {
    setFailFast(false);
    try {
      eval("foo");
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }
    update("foo", "bar");
    assertThat(eval("foo")).isEqualTo("bar");
  }

  // Test assign and reference through interpreter:
  @Test
  public void testAssignAndReference() throws Exception {
    setFailFast(false);
    try {
      eval("foo");
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }
    eval("foo = 'bar'");
    assertThat(eval("foo")).isEqualTo("bar");
  }

  @Test
  public void testGetVariableNames() throws Exception {
    Environment outerEnv;
    Environment innerEnv;
    try (Mutability mut = Mutability.create("outer")) {
      outerEnv =
          Environment.builder(mut)
              .setGlobals(Environment.DEFAULT_GLOBALS)
              .build()
              .update("foo", "bar")
              .update("wiz", 3);
    }
    try (Mutability mut = Mutability.create("inner")) {
      innerEnv = Environment.builder(mut)
          .setGlobals(outerEnv.getGlobals()).build()
          .update("foo", "bat")
          .update("quux", 42);
    }

    assertThat(outerEnv.getVariableNames())
        .isEqualTo(
            Sets.newHashSet(
                "foo",
                "wiz",
                "False",
                "None",
                "True",
                "all",
                "any",
                "bool",
                "dict",
                "dir",
                "enumerate",
                "fail",
                "getattr",
                "hasattr",
                "hash",
                "int",
                "len",
                "list",
                "max",
                "min",
                "print",
                "range",
                "repr",
                "reversed",
                "sorted",
                "str",
                "tuple",
                "zip"));
    assertThat(innerEnv.getVariableNames())
        .isEqualTo(
            Sets.newHashSet(
                "foo",
                "wiz",
                "quux",
                "False",
                "None",
                "True",
                "all",
                "any",
                "bool",
                "dict",
                "dir",
                "enumerate",
                "fail",
                "getattr",
                "hasattr",
                "hash",
                "int",
                "len",
                "list",
                "max",
                "min",
                "print",
                "range",
                "repr",
                "reversed",
                "sorted",
                "str",
                "tuple",
                "zip"));
  }

  @Test
  public void testToString() throws Exception {
    update("subject", new StringLiteral("Hello, 'world'."));
    update("from", new StringLiteral("Java"));
    assertThat(getEnvironment().toString()).isEqualTo("<Environment[test]>");
  }

  @Test
  public void testBindToNullThrowsException() throws Exception {
    try {
      update("some_name", null);
      fail();
    } catch (NullPointerException e) {
      assertThat(e).hasMessage("update(value == null)");
    }
  }

  @Test
  public void testFrozen() throws Exception {
    Environment env;
    try (Mutability mutability = Mutability.create("testFrozen")) {
      env =
          Environment.builder(mutability)
              .setGlobals(Environment.DEFAULT_GLOBALS)
              .setEventHandler(Environment.FAIL_FAST_HANDLER)
              .build();
      env.update("x", 1);
      assertThat(env.lookup("x")).isEqualTo(1);
      env.update("y", 2);
      assertThat(env.lookup("y")).isEqualTo(2);
      assertThat(env.lookup("x")).isEqualTo(1);
      env.update("x", 3);
      assertThat(env.lookup("x")).isEqualTo(3);
    }
    try {
      // This update to an existing variable should fail because the environment was frozen.
      env.update("x", 4);
      throw new Exception("failed to fail"); // not an AssertionError like fail()
    } catch (AssertionError e) {
      assertThat(e).hasMessage("Can't update x to 4 in frozen environment");
    }
    try {
      // This update to a new variable should also fail because the environment was frozen.
      env.update("newvar", 5);
      throw new Exception("failed to fail"); // not an AssertionError like fail()
    } catch (AssertionError e) {
      assertThat(e).hasMessage("Can't update newvar to 5 in frozen environment");
    }
  }

  @Test
  public void testReadOnly() throws Exception {
    Environment env = newSkylarkEnvironment()
        .setup("special_var", 42)
        .update("global_var", 666);

    // We don't even get a runtime exception trying to modify these,
    // because we get compile-time exceptions even before we reach runtime!
    try {
      BuildFileAST.eval(env, "special_var = 41");
      throw new AssertionError("failed to fail");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Variable special_var is read only");
    }

    try {
      BuildFileAST.eval(env, "def foo(x): x += global_var; global_var = 36; return x", "foo(1)");
      throw new AssertionError("failed to fail");
    } catch (EvalExceptionWithStackTrace e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Variable 'global_var' is referenced before assignment. "
                  + "The variable is defined in the global scope.");
    }
  }
}
