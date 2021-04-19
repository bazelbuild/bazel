// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.util.Pair;
import java.lang.ref.Reference;
import java.lang.reflect.Field;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * A helper class for tests providing a simple interface for asserts.
 */
public class MoreAsserts {

  public static <T> void assertEquals(T expected, T actual, Comparator<T> comp) {
    assertThat(comp.compare(expected, actual)).isEqualTo(0);
  }

  /**
   * Scans if an instance of given class is strongly reachable from a given
   * object.
   * <p>Runs breadth-first search in object reachability graph to check if
   * an instance of <code>clz</code> can be reached.
   * <strong>Note:</strong> This method can take a long time if analyzed
   * data structure spans across large part of heap and may need a lot of
   * memory.
   *
   * @param start object to start the search from
   * @param clazz class to look for
   */
  public static void assertInstanceOfNotReachable(
      Object start, final Class<?> clazz) {
    Predicate<Object> p = obj -> clazz.isAssignableFrom(obj.getClass());
    if (isRetained(p, start)) {
      fail("Found an instance of " + clazz.getCanonicalName() + " reachable from " + start);
    }
  }

  private static final Field NON_STRONG_REF;

  static {
    try {
      NON_STRONG_REF = Reference.class.getDeclaredField("referent");
    } catch (SecurityException | NoSuchFieldException e) {
      throw new RuntimeException(e);
    }
  }

  static final Predicate<Field> ALL_STRONG_REFS = Predicates.equalTo(NON_STRONG_REF);

  private static boolean isRetained(Predicate<Object> predicate, Object start) {
    IdentityHashMap<Object, Object> visited = Maps.newIdentityHashMap();
    visited.put(start, start);
    Queue<Object> toScan = new ArrayDeque<>();
    toScan.add(start);

    while (!toScan.isEmpty()) {
      Object current = toScan.poll();
      if (current.getClass().isArray()) {
        if (current.getClass().getComponentType().isPrimitive()) {
          continue;
        }

        for (Object ref : (Object[]) current) {
          if (ref != null) {
            if (predicate.apply(ref)) {
              return true;
            }
            if (visited.put(ref, ref) == null) {
              toScan.add(ref);
            }
          }
        }
      } else {
        // iterate *all* fields (getFields() returns only accessible ones)
        for (Class<?> clazz = current.getClass(); clazz != null;
            clazz = clazz.getSuperclass()) {
          for (Field f : clazz.getDeclaredFields()) {
            if (f.getType().isPrimitive() || ALL_STRONG_REFS.apply(f)) {
              continue;
            }

            try {
              f.setAccessible(true);
            } catch (RuntimeException e) {
              // JDK9 can throw InaccessibleObjectException when internal modules are accessed.
              // This isn't available in JDK8, so catch RuntimeException
              // We can use a JVM arg --add_opens to suppress that, but that involves every
              // test adding every JVM module to the target.
              continue;
            }
            try {
              Object ref = f.get(current);
              if (ref != null) {
                if (predicate.apply(ref)) {
                  return true;
                }
                if (visited.put(ref, ref) == null) {
                  toScan.add(ref);
                }
              }
            } catch (IllegalArgumentException | IllegalAccessException e) {
              throw new IllegalStateException("Error when scanning the heap", e);
            }
          }
        }
      }
    }
    return false;
  }

  private static String getClassDescription(Object object) {
    return object == null
        ? "null"
        : ("instance of " + object.getClass().getName());
  }

  public static String chattyFormat(String message, Object expected, Object actual) {
    String expectedClass = getClassDescription(expected);
    String actualClass = getClassDescription(actual);

    return Joiner.on('\n').join((message != null) ? ("\n" + message) : "",
        "  expected " + expectedClass + ": <" + expected + ">",
        "  but was " + actualClass + ": <" + actual + ">");
  }

  public static void assertEqualsUnifyingLineEnds(String expected, String actual) {
    if (actual != null) {
      actual = actual.replaceAll(System.getProperty("line.separator"), "\n");
    }
    assertThat(actual).isEqualTo(expected);
  }

  public static void assertContainsWordsWithQuotes(String message,
      String... strings) {
    for (String string : strings) {
      assertWithMessage(message + " should contain '" + string + "' (with quotes)")
          .that(message.contains("'" + string + "'"))
          .isTrue();
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

  public static Set<String> asStringSet(Iterable<?> collection) {
    Set<String> set = Sets.newTreeSet();
    for (Object o : collection) {
      set.add("\"" + o + "\"");
    }
    return set;
  }

  /**
   * If the specified EventCollector contains any events, an informative
   * assertion fails in the context of the specified TestCase.
   */
  public static void assertNoEvents(Iterable<Event> eventCollector) {
    String eventsString = eventsToString(eventCollector);
    assertThat(eventsString).isEmpty();
  }

  /**
   * If the specified EventCollector contains an unexpected number of events, an informative
   * assertion fails in the context of the specified TestCase.
   */
  public static void assertEventCount(int expectedCount, EventCollector eventCollector) {
    assertWithMessage(eventsToString(eventCollector))
        .that(eventCollector.count()).isEqualTo(expectedCount);
  }

  /**
   * If the specified EventCollector contains an unexpected number of events, an informative
   * assertion fails in the context of the specified TestCase.
   */
  public static void assertEventCountAtLeast(int minCount, EventCollector eventCollector) {
    assertWithMessage(eventsToString(eventCollector))
        .that(eventCollector.count())
        .isAtLeast(minCount);
  }

  /**
   * If the specified EventCollector does not contain an event which has 'expectedEvent' as a
   * substring, an informative assertion fails. Otherwise the matching event is returned.
   */
  public static Event assertContainsEvent(Iterable<Event> eventCollector, String expectedEvent) {
    return assertContainsEvent(eventCollector, expectedEvent, EventKind.ALL_EVENTS);
  }

  /**
   * If the specified EventCollector does not contain an event which has
   * 'expectedEvent' as a substring, an informative assertion fails. Otherwise
   * the matching event is returned.
   */
  public static Event assertContainsEvent(Iterable<Event> eventCollector,
      String expectedEvent, EventKind kind) {
    return assertContainsEvent(eventCollector, expectedEvent, ImmutableSet.of(kind));
  }

  /**
   * If the specified EventCollector does not contain an event of a kind of 'kinds' which has
   * 'expectedEvent' as a substring, an informative assertion fails. Otherwise
   * the matching event is returned.
   */
  public static Event assertContainsEvent(Iterable<Event> eventCollector,
                                          String expectedEvent,
                                          Set<EventKind> kinds) {
    for (Event event : eventCollector) {
      // We want to be able to check for the location and the message type (error / warning).
      // Consequently, we use toString() instead of getMessage().
      if (event.toString().contains(expectedEvent) && kinds.contains(event.getKind())) {
        return event;
      }
    }
    String eventsString = eventsToString(eventCollector);
    assertWithMessage("Event '" + expectedEvent + "' not found"
        + (eventsString.length() == 0 ? "" : ("; found these though:" + eventsString)))
        .that(false).isTrue();
    return null; // unreachable
  }

  /**
   * If {@code eventCollector} does not contain an event which matches {@code expectedEventPattern},
   * fails with an informative assertion.
   */
  public static Event assertContainsEvent(
      Iterable<Event> eventCollector, Pattern expectedEventPattern, EventKind... kinds) {
    return assertContainsEvent(eventCollector, expectedEventPattern, ImmutableSet.copyOf(kinds));
  }

  /**
   * If {@code eventCollector} does not contain an event which matches {@code expectedEventPattern},
   * fails with an informative assertion.
   */
  public static Event assertContainsEvent(
      Iterable<Event> eventCollector, Pattern expectedEventPattern, Set<EventKind> kinds) {
    for (Event event : eventCollector) {
      // Does the event message match the expected regex?
      if (!expectedEventPattern.matcher(event.toString()).find()) {
        continue;
      }
      // Was an expected kind given, and does the event match?
      if (!kinds.isEmpty() && !kinds.contains(event.getKind())) {
        continue;
      }
      // Return the event, assertion successful
      return event;
    }
    String eventsString = eventsToString(eventCollector);
    String failureMessage = "Event matching '" + expectedEventPattern + "' not found";
    if (!eventsString.isEmpty()) {
      failureMessage += "; found these though: " + eventsString;
    }
    fail(failureMessage);
    return null; // unreachable
  }

  public static void assertNotContainsEvent(
      Iterable<Event> eventCollector, Pattern unexpectedEventPattern) {
    for (Event event : eventCollector) {
      assertThat(event.toString()).doesNotMatch(unexpectedEventPattern);
    }
  }

  /**
   * If the specified EventCollector contains an event which has
   * 'expectedEvent' as a substring, an informative assertion fails.
   */
  public static void assertDoesNotContainEvent(Iterable<Event> eventCollector,
                                          String expectedEvent) {
    for (Event event : eventCollector) {
      assertWithMessage("Unexpected string '" + expectedEvent + "' matched following event:\n"
          + event.getMessage()).that(event.getMessage()).doesNotContain(expectedEvent);
    }
  }

  /**
   * If the specified EventCollector does not contain an event which has
   * each of {@code words} surrounded by single quotes as a substring, an
   * informative assertion fails.  Otherwise the matching event is returned.
   */
  public static Event assertContainsEventWithWordsInQuotes(
      Iterable<Event> eventCollector,
      String... words) {
    for (Event event : eventCollector) {
      boolean found = true;
      for (String word : words) {
        if (!event.getMessage().contains("'" + word + "'")) {
          found = false;
          break;
        }
      }
      if (found) {
        return event;
      }
    }
    String eventsString = eventsToString(eventCollector);
    assertWithMessage("Event containing words " + Arrays.toString(words) + " in "
        + "single quotes not found"
        + (eventsString.length() == 0 ? "" : ("; found these though:" + eventsString)))
        .that(false).isTrue();
    return null; // unreachable
  }

  /**
   * Returns a string consisting of each event in the specified collector,
   * preceded by a newline.
   */
  private static String eventsToString(Iterable<Event> eventCollector) {
    StringBuilder buf = new StringBuilder();
    eventLoop: for (Event event : eventCollector) {
      for (String ignoredPrefix : TestConstants.IGNORED_MESSAGE_PREFIXES) {
        if (event.getMessage().startsWith(ignoredPrefix)) {
          continue eventLoop;
        }
      }
      buf.append('\n').append(event);
    }
    return buf.toString();
  }

  /**
   * If "expectedSublist" is not a sublist of "arguments", an informative assertion is failed in the
   * context of the specified TestCase.
   *
   * <p>Argument order mnemonic: assert(X)ContainsSublist(Y).
   */
  @SuppressWarnings("varargs")
  public static <T> void assertContainsSublist(List<T> arguments, T... expectedSublist) {
    List<T> sublist = Arrays.asList(expectedSublist);
    try {
      assertThat(Collections.indexOfSubList(arguments, sublist)).isNotEqualTo(-1);
    } catch (AssertionError e) {
      throw new AssertionError("Did not find " + sublist + " as a sublist of " + arguments, e);
    }
  }

  /**
   * If "expectedSublist" is a sublist of "arguments", an informative assertion is failed in the
   * context of the specified TestCase.
   *
   * <p>Argument order mnemonic: assert(X)DoesNotContainSublist(Y).
   */
  @SuppressWarnings("varargs")
  public static <T> void assertDoesNotContainSublist(List<T> arguments, T... expectedSublist) {
    List<T> sublist = Arrays.asList(expectedSublist);
    try {
      assertThat(Collections.indexOfSubList(arguments, sublist)).isEqualTo(-1);
    } catch (AssertionError e) {
      throw new AssertionError("Found " + sublist + " as a sublist of " + arguments, e);
    }
  }

  /**
   * Check to see if each element of expectedMessages is the beginning of a message in
   * eventCollector, in order, as in {@link #containsSublistWithGapsAndEqualityChecker}. If not, an
   * informative assertion is failed
   */
  public static void assertContainsEventsInOrder(
      Iterable<Event> eventCollector, String... expectedMessages) {
    String failure =
        containsSublistWithGapsAndEqualityChecker(
            ImmutableList.copyOf(eventCollector),
            pair -> pair.first.getMessage().contains(pair.second),
            expectedMessages);

    String eventsString = eventsToString(eventCollector);
    assertWithMessage("Event '" + failure + "' not found in proper order"
        + (eventsString.length() == 0 ? "" : ("; found these though:" + eventsString)))
        .that(failure).isNull();
  }

  /**
   * Check to see if each element of expectedSublist is in arguments, according to
   * the equalityChecker, in the same order as in expectedSublist (although with
   * other interspersed elements in arguments allowed).
   * @param equalityChecker function that takes a Pair<S, T> element and returns true
   * if the elements of the pair are equal by its lights.
   * @return first element not in arguments in order, or null if success.
   */
  protected static <S, T> T containsSublistWithGapsAndEqualityChecker(List<S> arguments,
      Function<Pair<S, T>, Boolean> equalityChecker, T... expectedSublist) {
    Iterator<S> iter = arguments.iterator();
    outerLoop:
    for (T expected : expectedSublist) {
      while (iter.hasNext()) {
        S actual = iter.next();
        if (equalityChecker.apply(Pair.of(actual, expected))) {
          continue outerLoop;
        }
      }
      return expected;
    }
    return null;
  }

  public static List<Event> assertContainsEventWithFrequency(Iterable<Event> events,
      String expectedMessage, int expectedFrequency) {
    ImmutableList.Builder<Event> builder = ImmutableList.builder();
    for (Event event : events) {
      if (event.getMessage().contains(expectedMessage)) {
        builder.add(event);
      }
    }
    List<Event> foundEvents = builder.build();
    assertWithMessage(events.toString()).that(foundEvents).hasSize(expectedFrequency);
    return foundEvents;
  }
}
