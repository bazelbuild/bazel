// Copyright 2006 Google Inc. All Rights Reserved.
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

import com.google.common.collect.Sets;

/**
 * Tests of Environment.
 */
public class EnvironmentTest extends AbstractEvaluationTestCase {

  // Test the API directly
  public void testLookupAndUpdate() throws Exception {
    Environment env = new Environment();

    try {
      env.lookup("foo");
      fail();
    } catch (Environment.NoSuchVariableException e) {
       assertEquals("no such variable: foo", e.getMessage());
    }

    env.update("foo", "bar");

    assertEquals("bar", env.lookup("foo"));
  }

  public void testLookupWithDefault() throws Exception {
    Environment env = new Environment();
    assertEquals(21, env.lookup("VERSION", 21));
    env.update("VERSION", 42);
    assertEquals(42, env.lookup("VERSION", 21));
  }

  public void testDoubleUpdateSucceeds() throws Exception {
    Environment env = new Environment();
    env.update("VERSION", 42);
    assertEquals(42, env.lookup("VERSION"));
    env.update("VERSION", 43);
    assertEquals(43, env.lookup("VERSION"));
  }

  // Test assign through interpreter, lookup through API:
  public void testAssign() throws Exception {
    Environment env = new Environment();

    try {
      env.lookup("foo");
      fail();
    } catch (Environment.NoSuchVariableException e) {
      assertEquals("no such variable: foo", e.getMessage());
    }

    exec(parseStmt("foo = 'bar'"), env);

    assertEquals("bar", env.lookup("foo"));
  }

  // Test update through API, reference through interpreter:
  public void testReference() throws Exception {
    Environment env = new Environment();

    try {
      eval(parseExpr("foo"), env);
      fail();
    } catch (EvalException e) {
      assertEquals("name 'foo' is not defined", e.getMessage());
    }

    env.update("foo", "bar");

    assertEquals("bar", eval(parseExpr("foo"), env));
  }

  // Test assign and reference through interpreter:
  public void testAssignAndReference() throws Exception {
    Environment env = new Environment();

    try {
      eval(parseExpr("foo"), env);
      fail();
    } catch (EvalException e) {
      assertEquals("name 'foo' is not defined", e.getMessage());
    }

    exec(parseStmt("foo = 'bar'"), env);

    assertEquals("bar", eval(parseExpr("foo"), env));
  }

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

  public void testToString() throws Exception {
    Environment env = new Environment();
    env.update("subject", new StringLiteral("Hello, 'world'.", '\''));
    env.update("from", new StringLiteral("Java", '"'));
    assertEquals("Environment{False -> false, None -> None, True -> true, from -> \"Java\", "
        + "subject -> 'Hello, \\'world\\'.', }", env.toString());
  }

  public void testBindToNullThrowsException() throws Exception {
    try {
      new Environment().update("some_name", null);
      fail();
    } catch (NullPointerException e) {
      assertEquals("update(value == null)", e.getMessage());
    }
  }
}
