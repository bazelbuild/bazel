// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.ExitCode;

import junit.framework.TestCase;

import java.util.Objects;
import java.util.Set;

/**
 * Most of this stuff is copied from junit's {@link junit.framework.Assert}
 * class, and then customized to make the error messages a bit more informative.
 */
public abstract class ChattyAssertsTestCase extends TestCase {
  private long currentTestStartTime = -1;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    currentTestStartTime = BlazeClock.instance().currentTimeMillis();
  }

  @Override
  protected void tearDown() throws Exception {
    JunitTestUtils.nullifyInstanceFields(this);
    assertFalse("tearDown without setUp!", currentTestStartTime == -1);

    super.tearDown();
  }

  /**
   * Asserts that two objects are equal. If they are not
   * an AssertionFailedError is thrown with the given message.
   */
  public static void assertEquals(String message, Object expected,
      Object actual) {
    if (Objects.equals(expected, actual)) {
      return;
    }
    chattyFailNotEquals(message, expected, actual);
  }

  /**
   * Asserts that two objects are equal. If they are not
   * an AssertionFailedError is thrown.
   */
  public static void assertEquals(Object expected, Object actual) {
    assertEquals(null, expected, actual);
  }

  /**
   * Asserts that two Strings are equal.
   */
  public static void assertEquals(String message, String expected, String actual) {
    assertWithMessage(message).that(actual).isEqualTo(expected);
  }

  /**
   * Asserts that two Strings are equal.
   */
  public static void assertEquals(String expected, String actual) {
    assertEquals(null, expected, actual);
  }

  private static void chattyFailNotEquals(String message, Object expected,
      Object actual) {
    fail(MoreAsserts.chattyFormat(message, expected, actual));
  }

  public static void assertNonZeroExitCode(int exitCode, String stdout, String stderr) {
    MoreAsserts.assertNonZeroExitCode(exitCode, stdout, stderr);
  }

  public static void assertZeroExitCode(int exitCode, String stdout, String stderr) {
    MoreAsserts.assertExitCode(0, exitCode, stdout, stderr);
  }

  public static void assertExitCode(ExitCode expectedExitCode,
      int exitCode, String stdout, String stderr) {
    int expectedExitCodeValue = expectedExitCode.getNumericExitCode();
    if (exitCode != expectedExitCodeValue) {
      fail(String.format("expected exit code '%s' <%d> but exit code was <%d> and stdout was <%s> "
              + "and stderr was <%s>",
              expectedExitCode.name(), expectedExitCodeValue, exitCode, stdout, stderr));
    }
  }

  public static void assertExitCode(int expectedExitCode,
      int exitCode, String stdout, String stderr) {
    MoreAsserts.assertExitCode(expectedExitCode, exitCode,  stdout, stderr);
  }

  public static void assertStdoutContainsString(String expected, String stdout, String stderr) {
    MoreAsserts.assertStdoutContainsString(expected, stdout, stderr);
  }

  public static void assertStderrContainsString(String expected, String stdout, String stderr) {
    MoreAsserts.assertStderrContainsString(expected, stdout, stderr);
  }

  public static void assertStdoutContainsRegex(String expectedRegex,
      String stdout, String stderr) {
    MoreAsserts.assertStdoutContainsRegex(expectedRegex, stdout, stderr);
  }

  public static void assertStderrContainsRegex(String expectedRegex,
      String stdout, String stderr) {
    MoreAsserts.assertStderrContainsRegex(expectedRegex, stdout, stderr);
  }



  /********************************************************************
   *                                                                  *
   *       Other testing utilities (unrelated to "chattiness")        *
   *                                                                  *
   ********************************************************************/

  /**
   * Returns the elements from the given collection in a set.
   */
  protected static <T> Set<T> asSet(Iterable<T> collection) {
    return Sets.newHashSet(collection);
  }

  /**
   * Returns the arguments given as varargs as a set.
   */
  @SuppressWarnings({"unchecked", "varargs"})
  protected static <T> Set<T> asSet(T... elements) {
    return Sets.newHashSet(elements);
  }

  /**
   * An equivalence relation for Collection, based on mapping to Set.
   *
   * Oft-forgotten fact: for all x in Set, y in List, !x.equals(y) even if
   * their elements are the same.
   */
  protected static <T> void
      assertSameContents(Iterable<? extends T> expected, Iterable<? extends T> actual) {
    MoreAsserts.assertSameContents(expected, actual);
  }

  /**
   * Asserts the presence or absence of values in the collection.
   */
  protected <T> void assertPresence(Iterable<T> actual, Iterable<Presence<T>> expectedPresences) {
    for (Presence<T> expected : expectedPresences) {
      if (expected.presence) {
        assertThat(actual).contains(expected.value);
      } else {
        assertThat(actual).doesNotContain(expected.value);
      }
    }
  }

  /** Creates a presence information with expected value. */
  protected static <T> Presence<T> present(T expected) {
    return new Presence<>(expected, true);
  }

  /** Creates an absence information with expected value. */
  protected static <T> Presence<T> absent(T expected) {
    return new Presence<>(expected, false);
  }

  /**
   * Combines value with the boolean presence flag.
   *
   * @param <T> value type
   */
  protected final static class Presence <T> {
    /** wrapped value */
    public final T value;
    /** boolean presence flag */
    public final boolean presence;

    /** Creates a tuple of value and a boolean presence flag. */
    Presence(T value, boolean presence) {
      this.value = value;
      this.presence = presence;
    }
  }

}
