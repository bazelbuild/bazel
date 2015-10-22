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
package com.google.devtools.build.lib.events.util;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.io.OutErr;

import java.util.List;
import java.util.Set;

/**
 * An apparatus for reporting / collecting events.
 */
public class EventCollectionApparatus {
  private EventCollector eventCollector;
  private Reporter reporter;
  private PrintingEventHandler printingEventHandler;

  /**
   * Determine which events the {@link #collector()} created by this apparatus
   * will collect. Default: {@link EventKind#ERRORS_AND_WARNINGS}.
   */
  public EventCollectionApparatus(Set<EventKind> mask) {
    eventCollector = new EventCollector(mask);
    reporter = new Reporter(eventCollector);
    printingEventHandler = new PrintingEventHandler(EventKind.ERRORS_AND_WARNINGS_AND_OUTPUT);
    reporter.addHandler(printingEventHandler);

    this.setFailFast(true);
  }

  public EventCollectionApparatus() {
    this(EventKind.ERRORS_AND_WARNINGS);
  }

  public void clear() {
    eventCollector.clear();
  }

  /**
   * Determine whether the {#link reporter()} created by this apparatus will
   * fail fast, that is, throw an exception whenever we encounter an event of
   * matching {@link EventKind#ERRORS_AND_WARNINGS}.
   * Default: {@code true}.
   */
  public void setFailFast(boolean failFast) {
    if (failFast) {
      reporter.addHandler(Environment.FAIL_FAST_HANDLER);
    } else {
      reporter.removeHandler(Environment.FAIL_FAST_HANDLER);
    }
  }

  /**
   * @return the event reporter for this apparatus
   */
  public Reporter reporter() {
    return reporter;
  }

  /**
   * @return the event collector for this apparatus.
   */
  public EventCollector collector() {
    return eventCollector;
  }

  public Iterable<Event> errors() {
    return eventCollector.filtered(EventKind.ERROR);
  }

  public Iterable<Event> warnings() {
    return eventCollector.filtered(EventKind.WARNING);
  }

  /**
   * Redirects all output to the specified OutErr stream pair.
   * Returns the previous OutErr.
   */
  public OutErr setOutErr(OutErr outErr) {
    return printingEventHandler.setOutErr(outErr);
  }

  /**
   * Utility method: Asserts that the {@link #collector()} has not collected
   * any events.
   *
   * @throws IllegalStateException If the apparatus has not yet been
   *    initialized by calling {@link #reporter()} or {@link #collector()}.
   */
  public void assertNoEvents() {
    MoreAsserts.assertNoEvents(eventCollector);
  }

  /**
   * Utility method: Assert that the {@link #collector()} has received an
   * info message with the {@code expectedMessage}.
   */
  public Event assertContainsInfo(String expectedMessage) {
    return MoreAsserts.assertContainsEvent(eventCollector, expectedMessage, EventKind.INFO);
  }

  /**
   * Utility method: Assert that the {@link #collector()} has received an
   * error with the {@code expectedMessage}.
   */
  public Event assertContainsError(String expectedMessage) {
    return MoreAsserts.assertContainsEvent(eventCollector, expectedMessage, EventKind.ERROR);
  }

  /**
   * Utility method: Assert that the {@link #collector()} has received a
   * warning with the {@code expectedMessage}.
   */
  public Event assertContainsWarning(String expectedMessage) {
    return MoreAsserts.assertContainsEvent(eventCollector, expectedMessage, EventKind.WARNING);
  }

  public List<Event> assertContainsEventWithFrequency(String expectedMessage,
      int expectedFrequency) {
    return MoreAsserts.assertContainsEventWithFrequency(eventCollector, expectedMessage,
        expectedFrequency);
  }

  /**
   * Utility method: Assert that the {@link #collector()} has received an
   * event with the {@code expectedMessage} in quotes.
   */

  public Event assertContainsEventWithWordsInQuotes(String... words) {
    return MoreAsserts.assertContainsEventWithWordsInQuotes(
        eventCollector, words);
  }

  public void assertDoesNotContainEvent(String unexpectedEvent) {
    MoreAsserts.assertDoesNotContainEvent(eventCollector, unexpectedEvent);
  }
}
