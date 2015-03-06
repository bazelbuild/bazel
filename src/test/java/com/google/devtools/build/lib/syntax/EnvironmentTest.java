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

import com.google.common.collect.Sets;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests of Environment.
 */
@RunWith(JUnit4.class)
public class EnvironmentTest extends AbstractEvaluationTestCase {

  // Test the API directly
  @Test
  public void testLookupAndUpdate() throws Exception {
    Environment env = new Environment();

    try {
      env.lookup("foo");
      fail();
    } catch (Environment.NoSuchVariableException e) {
      assertThat(e).hasMessage("no such variable: foo");
    }

    env.update("foo", "bar");

    assertEquals("bar", env.lookup("foo"));
  }

  @Test
  public void testLookupWithDefault() throws Exception {
    Environment env = new Environment();
    assertEquals(21, env.lookup("VERSION", 21));
    env.update("VERSION", 42);
    assertEquals(42, env.lookup("VERSION", 21));
  }

  @Test
  public void testDoubleUpdateSucceeds() throws Exception {
    Environment env = new Environment();
    env.update("VERSION", 42);
    assertEquals(42, env.lookup("VERSION"));
    env.update("VERSION", 43);
    assertEquals(43, env.lookup("VERSION"));
  }

  // Test assign through interpreter, lookup through API:
  @Test
  public void testAssign() throws Exception {
    Environment env = new Environment();

    try {
      env.lookup("foo");
      fail();
    } catch (Environment.NoSuchVariableException e) {
      assertThat(e).hasMessage("no such variable: foo");
    }

    exec(parseStmt("foo = 'bar'"), env);

    assertEquals("bar", env.lookup("foo"));
  }

  // Test update through API, reference through interpreter:
  @Test
  public void testReference() throws Exception {
    Environment env = new Environment();

    try {
      eval(parseExpr("foo"), env);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }

    env.update("foo", "bar");

    assertEquals("bar", eval(parseExpr("foo"), env));
  }

  // Test assign and reference through interpreter:
  @Test
  public void testAssignAndReference() throws Exception {
    Environment env = new Environment();

    try {
      eval(parseExpr("foo"), env);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage("name 'foo' is not defined");
    }

    exec(parseStmt("foo = 'bar'"), env);

    assertEquals("bar", eval(parseExpr("foo"), env));
  }

  @Test
  public void testGetVariableNames() throws Exception {
    Environment env = new Environment();
    env.update("foo", "bar");
    env.update("wiz", 3);

    Environment nestedEnv = new Environment(env);
    nestedEnv.update("foo", "bat");
    nestedEnv.update("quux", 42);

    assertEquals(Sets.newHashSet("True", "False", "None", "foo", "wiz"), env.getVariableNames());
    assertEquals(Sets.newHashSet("True", "False", "None", "foo", "wiz", "quux"),
        nestedEnv.getVariableNames());
  }

  @Test
  public void testToString() throws Exception {
    Environment env = new Environment();
    env.update("subject", new StringLiteral("Hello, 'world'.", '\''));
    env.update("from", new StringLiteral("Java", '"'));
    assertEquals("Environment{False -> false, None -> None, True -> true, from -> \"Java\", "
        + "subject -> 'Hello, \\'world\\'.', }", env.toString());
  }

  @Test
  public void testBindToNullThrowsException() throws Exception {
    try {
      new Environment().update("some_name", null);
      fail();
    } catch (NullPointerException e) {
      assertThat(e).hasMessage("update(value == null)");
    }
  }
}
