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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.io.OutErr;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * An apparatus for reporting / collecting events.
 */
public final class EventCollectionApparatus {
  private EventCollector eventCollector;
  private Reporter reporter;
  private PrintingEventHandler printingEventHandler;

  private boolean failFast;
  private List<EventHandler> handlers = new ArrayList<>();

  /**
   * Determine which events the {@link #collector()} created by this apparatus
   * will collect. Default: {@link EventKind#ERRORS_AND_WARNINGS}.
   */
  public EventCollectionApparatus(Set<EventKind> mask) {
    eventCollector = new EventCollector(mask);
    printingEventHandler = new PrintingEventHandler(EventKind.ERRORS_AND_WARNINGS_AND_OUTPUT);
    reporter = new Reporter(new EventBus(), eventCollector, printingEventHandler);
    this.setFailFast(true);
  }

  public EventCollectionApparatus() {
    this(EventKind.ERRORS_WARNINGS_AND_INFO);
  }

  public void clear() {
    eventCollector.clear();
  }

  public void initExternal(Reporter reporter) {
    // TODO(ulfjack): Changes to the EventCollectionApparatus are not reflected in the external
    // reporter, i.e., this is a one-shot change. Maybe we should store the external reporter here?
    reporter.addHandler(eventCollector);
    reporter.addHandler(printingEventHandler);
    for (EventHandler handler : handlers) {
      reporter.addHandler(handler);
    }
    if (failFast) {
      reporter.addHandler(FAIL_FAST_HANDLER);
    }
  }

  /**
   * Determine whether the {#link reporter()} created by this apparatus will
   * fail fast, that is, throw an exception whenever we encounter an event of
   * matching {@link EventKind#ERRORS_AND_WARNINGS}.
   * Default: {@code true}.
   */
  public void setFailFast(boolean failFast) {
    this.failFast = failFast;
    if (failFast) {
      reporter.addHandler(FAIL_FAST_HANDLER);
    } else {
      reporter.removeHandler(FAIL_FAST_HANDLER);
    }
  }

  public void addHandler(EventHandler eventHandler) {
    reporter.addHandler(eventHandler);
    handlers.add(eventHandler);
  }

  /** An exception thrown by {@link #FAIL_FAST_HANDLER}. */
  // TODO(bazel-team): Possibly extend RuntimeException instead of IllegalArgumentException.
  public static class FailFastException extends IllegalArgumentException {
    public FailFastException(String s) {
      super(s);
    }
  }

  /**
   * A handler that immediately throws {@link FailFastException} whenever an error or warning
   * occurs.
   *
   * <p>We do not reuse an existing unchecked exception type, because callers (e.g., test
   * assertions) need to be able to distinguish between organically occurring exceptions and
   * exceptions thrown by this handler.
   */
  private static final EventHandler FAIL_FAST_HANDLER =
      new EventHandler() {
        @Override
        public void handle(Event event) {
          if (EventKind.ERRORS_AND_WARNINGS.contains(event.getKind())) {
            throw new FailFastException(event.toString());
          }
        }
      };

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

  public Iterable<Event> infos() {
    return eventCollector.filtered(EventKind.INFO);
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
   * any warnings or errors.
   */
  public void assertNoWarningsOrErrors() {
    MoreAsserts.assertNoEvents(warnings());
    MoreAsserts.assertNoEvents(errors());
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
   * Utility method: Assert that the {@link #collector()} has received an error that matches {@code
   * expectedPattern}.
   */
  public Event assertContainsError(Pattern expectedPattern) {
    return MoreAsserts.assertContainsEvent(eventCollector, expectedPattern, EventKind.ERROR);
  }

  /**
   * Utility method: Assert that the {@link #collector()} has received a warning with the {@code
   * expectedMessage}.
   */
  public Event assertContainsWarning(String expectedMessage) {
    return MoreAsserts.assertContainsEvent(eventCollector, expectedMessage, EventKind.WARNING);
  }

  /**
   * Utility method: Assert that the {@link #collector()} has received an event of the given type
   * and with the {@code expectedMessage}.
   */
  public Event assertContainsEvent(EventKind kind, String expectedMessage) {
    return MoreAsserts.assertContainsEvent(eventCollector, expectedMessage, kind);
  }

  public List<Event> assertContainsEventWithFrequency(String expectedMessage,
      int expectedFrequency) {
    return MoreAsserts.assertContainsEventWithFrequency(eventCollector, expectedMessage,
        expectedFrequency);
  }

  public void assertDoesNotContainEvent(String unexpectedEvent) {
    MoreAsserts.assertDoesNotContainEvent(eventCollector, unexpectedEvent);
  }

  public void assertContainsEventsInOrder(String... expectedMessages) {
    MoreAsserts.assertContainsEventsInOrder(eventCollector, expectedMessages);
  }
}
