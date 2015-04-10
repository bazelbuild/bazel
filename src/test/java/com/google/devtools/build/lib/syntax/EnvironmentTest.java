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
public class EnvironmentTest extends EvaluationTestCase {

  @Override
  public EvaluationContext newEvaluationContext() {
    return EvaluationContext.newBuildContext(getEventHandler());
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
    update("foo", "bar");
    update("wiz", 3);

    Environment nestedEnv = new Environment(getEnvironment());
    nestedEnv.update("foo", "bat");
    nestedEnv.update("quux", 42);

    assertEquals(Sets.newHashSet("True", "False", "None", "foo", "wiz"),
        getEnvironment().getVariableNames());
    assertEquals(Sets.newHashSet("True", "False", "None", "foo", "wiz", "quux"),
        nestedEnv.getVariableNames());
  }

  @Test
  public void testToString() throws Exception {
    update("subject", new StringLiteral("Hello, 'world'.", '\''));
    update("from", new StringLiteral("Java", '"'));
    assertEquals("Environment{False -> false, None -> None, True -> true, from -> \"Java\", "
        + "subject -> 'Hello, \\'world\\'.', }", getEnvironment().toString());
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
}
