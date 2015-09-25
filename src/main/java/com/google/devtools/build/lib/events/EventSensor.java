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
package com.google.devtools.build.lib.events;

import java.util.Set;

/**
 * A "latch" that just detects whether or not a particular type of event has happened, based on its
 * kind.
 *
 * <p>Be careful when using this class to track errors reported during some operation. Namely, this
 * pattern is not thread-safe:
 *
 * <pre><code>
 * EventSensor sensor = new EventSensor(EventKind.ERRORS);
 * reporter.addHandler(sensor);
 * someActionThatMightCreateErrors(reporter)
 * reporter.removeHandler(sensor);
 * boolean containsErrors = sensor.wasTriggered();
 * </code></pre>
 *
 * <p>If other threads generate errors on the reporter, then containsErrors may be true even if
 * someActionThatMightCreateErrors() did not cause any errors.
 *
 * <p>As a workaround, run someActionThatMightCreateErrors() with a local reporter, merging its
 * events with those of the shared reporter.
 */
public class EventSensor extends AbstractEventHandler {

  private int triggerCount;

  /**
   * Constructs a sensor that will register all events matching the mask.
   */
  public EventSensor(Set<EventKind> mask) {
    super(mask);
  }

  /**
   * Implements {@link EventHandler#handle(Event)}.
   */
  @Override
  public void handle(Event event) {
    if (getEventMask().contains(event.getKind())) {
      triggerCount++;
    }
  }

  /**
   * Returns true iff a qualifying event was handled.
   */
  public boolean wasTriggered() {
    return triggerCount > 0;
  }

  /**
   * Returns the number of times the qualifying event was handled.
   */
  public int getTriggerCount() {
    return triggerCount;
  }
}
