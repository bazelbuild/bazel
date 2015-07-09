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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.util.Pair;

import junit.framework.TestCase;

import java.lang.reflect.AccessibleObject;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * This class contains a utility method {@link #nullifyInstanceFields(Object)}
 * for setting all fields in an instance to {@code null}. This is needed for
 * junit {@code TestCase} instances that keep expensive objects in fields.
 * Basically junit holds onto the instances
 * even after the test methods have run, and it creates one such instance
 * per {@code testFoo} method.
 */
public class JunitTestUtils {

  public static void nullifyInstanceFields(Object instance)
      throws IllegalAccessException {
    /**
     * We're cleaning up this test case instance by assigning null pointers
     * to all fields to reduce the memory overhead of test case instances
     * staying around after the test methods have been executed. This is a
     * bug in junit.
     */
    List<Field> instanceFields = new ArrayList<>();
    for (Class<?> clazz = instance.getClass();
         !clazz.equals(TestCase.class) && !clazz.equals(Object.class);
         clazz = clazz.getSuperclass()) {
      for (Field field : clazz.getDeclaredFields()) {
        if (Modifier.isStatic(field.getModifiers())) {
          continue;
        }
        if (field.getType().isPrimitive()) {
          continue;
        }
        if (Modifier.isFinal(field.getModifiers())) {
          String msg = "Please make field \"" + field + "\" non-final, or, if " +
                       "it's very simple and truly immutable and not too " +
                       "big, make it static.";
          throw new AssertionError(msg);
        }
        instanceFields.add(field);
      }
    }
    // Run setAccessible for efficiency
    AccessibleObject.setAccessible(instanceFields.toArray(new Field[0]), true);
    for (Field field : instanceFields) {
      field.set(instance, null);
    }
  }

  /********************************************************************
   *                                                                  *
   *                         "Mix-in methods"                         *
   *                                                                  *
   ********************************************************************/

  // Java doesn't support mix-ins, but we need them in our tests so that we can
  // inherit a bunch of useful methods, e.g. assertions over an EventCollector.
  // We do this by hand, by delegating from instance methods in each TestCase
  // to the static methods below.

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
   * If the specified EventCollector does not contain an event which has
   * 'expectedEvent' as a substring, an informative assertion fails. Otherwise
   * the matching event is returned.
   */
  public static Event assertContainsEvent(Iterable<Event> eventCollector,
      String expectedEvent) {
    return assertContainsEvent(eventCollector, expectedEvent, EventKind.ALL_EVENTS);
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
      if (event.getMessage().contains(expectedEvent) && kinds.contains(event.getKind())) {
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
   * If "expectedSublist" is not a sublist of "arguments", an informative
   * assertion is failed in the context of the specified TestCase.
   *
   * Argument order mnemonic: assert(X)ContainsSublist(Y).
   */
  @SuppressWarnings({"unchecked", "varargs"})
  public static <T> void assertContainsSublist(List<T> arguments, T... expectedSublist) {
    List<T> sublist = Arrays.asList(expectedSublist);
    try {
      assertThat(Collections.indexOfSubList(arguments, sublist)).isNotEqualTo(-1);
    } catch (AssertionError e) {
      throw new AssertionError("Did not find " + sublist + " as a sublist of " + arguments, e);
    }
  }

  /**
   * If "expectedSublist" is a sublist of "arguments", an informative
   * assertion is failed in the context of the specified TestCase.
   *
   * Argument order mnemonic: assert(X)DoesNotContainSublist(Y).
   */
  @SuppressWarnings({"unchecked", "varargs"})
  public static <T> void assertDoesNotContainSublist(List<T> arguments, T... expectedSublist) {
    List<T> sublist = Arrays.asList(expectedSublist);
    try {
      assertThat(Collections.indexOfSubList(arguments, sublist)).isEqualTo(-1);
    } catch (AssertionError e) {
      throw new AssertionError("Found " + sublist + " as a sublist of " + arguments, e);
    }
  }

  /**
   * If "arguments" does not contain "expectedSubset" as a subset, an
   * informative assertion is failed in the context of the specified TestCase.
   *
   * Argument order mnemonic: assert(X)ContainsSubset(Y).
   */
  public static <T> void assertContainsSubset(Iterable<T> arguments,
                                              Iterable<T> expectedSubset) {
    Set<T> argumentsSet = arguments instanceof Set<?>
        ? (Set<T>) arguments
        : Sets.newHashSet(arguments);

    for (T x : expectedSubset) {
      assertWithMessage("assertContainsSubset failed: did not find element " + x
          + "\nExpected subset = " + expectedSubset + "\nArguments = " + arguments)
          .that(argumentsSet).contains(x);
    }
  }

  /**
   * Check to see if each element of expectedMessages is the beginning of a message
   * in eventCollector, in order, as in {@link #containsSublistWithGapsAndEqualityChecker}.
   * If not, an informative assertion is failed
   */
  protected static void assertContainsEventsInOrder(Iterable<Event> eventCollector,
      String... expectedMessages) {
    String failure = containsSublistWithGapsAndEqualityChecker(
        ImmutableList.copyOf(eventCollector),
        new Function<Pair<Event, String>, Boolean> () {
      @Override
      public Boolean apply(Pair<Event, String> pair) {
        return pair.first.getMessage().contains(pair.second);
      }
    }, expectedMessages);

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
  @SuppressWarnings({"unchecked"})
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
