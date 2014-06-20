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

import com.google.common.base.Joiner;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.ExitCode;

import junit.framework.AssertionFailedError;
import junit.framework.ComparisonFailure;
import junit.framework.TestCase;

import java.util.Arrays;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Most of this stuff is copied from junit's {@link junit.framework.Assert}
 * class, and then customized to make the error messages a bit more informative.
 */
public abstract class ChattyAssertsTestCase extends TestCase {

  // TODO(kcooney): Parts of this class may become irrelevant when Google
  // comletes its move to JUnit 4.5. Revisit the value of these methods post-
  // migration.

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

  private static class ChattyComparisonFailure extends AssertionFailedError {

    private final String expected;
    private final String actual;

    public ChattyComparisonFailure(String message, String expected,
        String actual) {
      super(message);
      this.expected = expected;
      this.actual = actual;
    }

    @Override
    public String getMessage() {
      return chattyFormat(super.getMessage(), expected, actual);
    }

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

  static boolean CHATTY_FAILURES = false;
  static {
    if (System.getenv().containsKey("CHATTY_COMPARISON_FAILURES")) {
      CHATTY_FAILURES = true;
    }
  }

  /**
   * Asserts that two Strings are equal.
   */
  public static void assertEquals(String message, String expected, String actual) {
    if (Objects.equals(expected, actual)) {
      return;
    }
    comparisonFailure(message, expected, actual);
  }

  /**
   * Report two Strings as different.
   */
  public static void comparisonFailure(String message, String expected, String actual) {
    if (CHATTY_FAILURES) {
      throw new ChattyComparisonFailure(message, expected, actual);
    }
    throw new ComparisonFailure(message, expected, actual);
  }

  /**
   * Asserts that two Strings are equal.
   */
  public static void assertEquals(String expected, String actual) {
    assertEquals(null, expected, actual);
  }

  /**
   * Asserts that two Strings are equal considering the line separator to be \n
   * independently of the operating system.
   */
  public static void assertEqualsUnifyingLineEnds(String expected, String actual) {
    if (actual != null) {
      actual = actual.replaceAll(System.getProperty("line.separator"), "\n");
    }
    assertEquals(expected, actual);
  }

  private static void chattyFailNotEquals(String message, Object expected,
      Object actual) {
    fail(chattyFormat(message, expected, actual));
  }

  /**
   * Asserts that {@code e}'s exception message contains each of {@code strings}
   * <b>surrounded by single quotation marks</b>.
   */
  public static void assertMessageContainsWordsWithQuotes(Exception e,
                                                          String... strings) {
    assertContainsWordsWithQuotes(e.getMessage(), strings);
  }

  /**
   * Asserts that {@code message} contains each of {@code strings}
   * <b>surrounded by single quotation marks</b>.
   */
  public static void assertContainsWordsWithQuotes(String message,
                                                   String... strings) {
    for (String string : strings) {
      assertTrue(message + " should contain '" + string + "' (with quotes)",
          message.contains("'" + string + "'"));
    }
  }

  public static void assertNonZeroExitCode(int exitCode, String stdout, String stderr) {
    if (exitCode == 0) {
      fail("expected non-zero exit code but exit code was 0 and stdout was <"
          + stdout + "> and stderr was <" + stderr + ">");
    }
  }

  public static void assertZeroExitCode(int exitCode, String stdout, String stderr) {
    assertExitCode(0, exitCode, stdout, stderr);
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
    if (exitCode != expectedExitCode) {
      fail(String.format("expected exit code <%d> but exit code was <%d> and stdout was <%s> "
          + "and stderr was <%s>", expectedExitCode, exitCode, stdout, stderr));
    }
  }

  public static void assertStdoutContainsString(String expected, String stdout, String stderr) {
    if (!stdout.contains(expected)) {
      fail("expected stdout to contain string <" + expected + "> but stdout was <"
          + stdout + "> and stderr was <" + stderr + ">");
    }
  }

  public static void assertStderrContainsString(String expected, String stdout, String stderr) {
    if (!stderr.contains(expected)) {
      fail("expected stderr to contain string <" + expected + "> but stdout was <"
          + stdout + "> and stderr was <" + stderr + ">");
    }
  }

  public static void assertStdoutContainsRegex(String expectedRegex,
      String stdout, String stderr) {
    if (!Pattern.compile(expectedRegex).matcher(stdout).find()) {
      fail("expected stdout to contain regex <" + expectedRegex + "> but stdout was <"
          + stdout + "> and stderr was <" + stderr + ">");
    }
  }

  public static void assertStderrContainsRegex(String expectedRegex,
      String stdout, String stderr) {
    if (!Pattern.compile(expectedRegex).matcher(stderr).find()) {
      fail("expected stderr to contain regex <" + expectedRegex + "> but stdout was <"
          + stdout + "> and stderr was <" + stderr + ">");
    }
  }

  private static String getClassDescription(Object object) {
    return object == null
        ? "null"
        : ("instance of " + object.getClass().getName());
  }

  static String chattyFormat(String message, Object expected, Object actual) {
    String expectedClass = getClassDescription(expected);
    String actualClass = getClassDescription(actual);

    return Joiner.on('\n').join((message != null) ? ("\n" + message) : "",
        "  expected " + expectedClass + ": <" + expected + ">",
        "  but was " + actualClass + ": <" + actual + ">");
  }

  /********************************************************************
   *                                                                  *
   *       Other testing utilities (unrelated to "chattiness")        *
   *                                                                  *
   ********************************************************************/

  /**
   * Returns the varargs as an iterable. This is useful for writing loops like
   * <code>
   *    for (String name :in ("foo", "bar", "baz")) {
   *       System.out.println("name = " + name);
   *    }
   * </code>
   */
  @SuppressWarnings({"unchecked", "varargs"})
  protected static <T> Iterable<T> in(T... elements) {
    return Arrays.asList(elements);
  }

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
   * Returns the arguments given as varargs as a set of sorted Strings.
   */
  protected static Set<String> asStringSet(Iterable<?> collection) {
    Set<String> set = Sets.newTreeSet();
    for (Object o : collection) {
      set.add("\"" + String.valueOf(o) + "\"");
    }
    return set;
  }

  /**
   * An equivalence relation for Collection, based on mapping to Set.
   *
   * Oft-forgotten fact: for all x in Set, y in List, !x.equals(y) even if
   * their elements are the same.
   */
  protected static <T> void
      assertSameContents(Iterable<? extends T> expected, Iterable<? extends T> actual) {
    if (!asSet(expected).equals(asSet(actual))) {
      comparisonFailure("different contents",
          asStringSet(expected).toString(), asStringSet(actual).toString());
    }
  }

  /**
   * Asserts the presence or absence of values in the collection.
   */
  protected <T> void assertPresence(Iterable<T> actual, Iterable<Presence<T>> expectedPresences) {
    for (Presence<T> expected : expectedPresences) {
      if (expected.presence) {
        MoreAsserts.assertContains(actual, expected.value);
      } else {
        MoreAsserts.assertNotContains(actual, expected.value);
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
