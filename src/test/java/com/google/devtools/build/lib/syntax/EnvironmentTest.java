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
import static org.junit.Assert.assertEquals;
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
    try {
      lookup("foo");
      fail();
    } catch (Environment.NoSuchVariableException e) {
      assertThat(e).hasMessage("no such variable: foo");
    }
    update("foo", "bar");
    assertEquals("bar", lookup("foo"));
  }

  @Test
  public void testLookupWithDefault() throws Exception {
    assertEquals(21, getEnvironment().lookup("VERSION", 21));
    update("VERSION", 42);
    assertEquals(42, getEnvironment().lookup("VERSION", 21));
  }

  @Test
  public void testDoubleUpdateSucceeds() throws Exception {
    update("VERSION", 42);
    assertEquals(42, lookup("VERSION"));
    update("VERSION", 43);
    assertEquals(43, lookup("VERSION"));
  }

  // Test assign through interpreter, lookup through API:
  @Test
  public void testAssign() throws Exception {
    try {
      lookup("foo");
      fail();
    } catch (Environment.NoSuchVariableException e) {
      assertThat(e).hasMessage("no such variable: foo");
    }
    eval("foo = 'bar'");
    assertEquals("bar", lookup("foo"));
  }

  // Test update through API, reference through interpreter:
  @Test
  public void testReference() throws Exception {
    try {
      eval("foo");
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }
    update("foo", "bar");
    assertEquals("bar", eval("foo"));
  }

  // Test assign and reference through interpreter:
  @Test
  public void testAssignAndReference() throws Exception {
    try {
      eval("foo");
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }
    eval("foo = 'bar'");
    assertEquals("bar", eval("foo"));
  }

  @Test
  public void testGetVariableNames() throws Exception {
    Environment outerEnv;
    Environment innerEnv;
    try (Mutability mut = Mutability.create("outer")) {
      outerEnv = Environment.builder(mut)
          .setGlobals(Environment.BUILD).build()
          .update("foo", "bar")
          .update("wiz", 3);
    }
    try (Mutability mut = Mutability.create("inner")) {
      innerEnv = Environment.builder(mut)
          .setGlobals(outerEnv.getGlobals()).build()
          .update("foo", "bat")
          .update("quux", 42);
    }

    assertEquals(Sets.newHashSet("foo", "wiz",
            "False", "None", "True",
            "-", "all", "any", "bool", "dict", "enumerate", "int", "len", "list",
            "max", "min", "range", "repr", "reversed", "select", "set", "sorted", "str", "zip"),
        outerEnv.getVariableNames());
    assertEquals(Sets.newHashSet("foo", "wiz", "quux",
            "False", "None", "True",
            "-", "all", "any", "bool", "dict", "enumerate", "int", "len", "list",
            "max", "min", "range", "repr", "reversed", "select", "set", "sorted", "str", "zip"),
        innerEnv.getVariableNames());
  }

  @Test
  public void testToString() throws Exception {
    update("subject", new StringLiteral("Hello, 'world'.", '\''));
    update("from", new StringLiteral("Java", '"'));
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
      env = Environment.builder(mutability)
          .setGlobals(Environment.BUILD).setEventHandler(Environment.FAIL_FAST_HANDLER).build();
      env.update("x", 1);
      assertEquals(env.lookup("x"), 1);
      env.update("y", 2);
      assertEquals(env.lookup("y"), 2);
      assertEquals(env.lookup("x"), 1);
      env.update("x", 3);
      assertEquals(env.lookup("x"), 3);
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
      env.eval("special_var = 41");
      throw new AssertionError("failed to fail");
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessage("ERROR 1:1: Variable special_var is read only");
    }

    try {
      env.eval("def foo(x): x += global_var; global_var = 36; return x", "foo(1)");
      throw new AssertionError("failed to fail");
    } catch (EvalExceptionWithStackTrace e) {
      assertThat(e.getMessage()).contains("Variable 'global_var' is referenced before assignment. "
          + "The variable is defined in the global scope.");
    }
  }
}
