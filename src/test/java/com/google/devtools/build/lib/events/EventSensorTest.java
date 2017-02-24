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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.eventbus.EventBus;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link EventSensor}.
 */
@RunWith(JUnit4.class)
public class EventSensorTest extends EventTestTemplate {

  @Test
  public void sensorStartsOutWithFalse() {
    assertFalse(new EventSensor(EventKind.ALL_EVENTS).wasTriggered());
    assertFalse(new EventSensor(EventKind.ERRORS).wasTriggered());
    assertFalse(new EventSensor(EventKind.ERRORS_AND_WARNINGS).wasTriggered());
  }

  @Test
  public void sensorNoticesEventsInItsMask() {
    EventSensor sensor = new EventSensor(EventKind.ERRORS);
    Reporter reporter = new Reporter(new EventBus(), sensor);
    reporter.handle(Event.error(location, "An ERROR event."));
    assertTrue(sensor.wasTriggered());
  }

  @Test
  public void sensorNoticesEventsInItsMask2() {
    EventSensor sensor = new EventSensor(EventKind.ALL_EVENTS);
    Reporter reporter = new Reporter(new EventBus(), sensor);
    reporter.handle(Event.error(location, "An ERROR event."));
    reporter.handle(Event.warn(location, "A warning event."));
    assertTrue(sensor.wasTriggered());
  }

  @Test
  public void sensorIgnoresEventsNotInItsMask() {
    EventSensor sensor = new EventSensor(EventKind.ERRORS_AND_WARNINGS);
    Reporter reporter = new Reporter(new EventBus(), sensor);
    reporter.handle(Event.info(location, "An INFO event."));
    assertFalse(sensor.wasTriggered());
  }

  @Test
  public void sensorCanCount() {
    EventSensor sensor = new EventSensor(EventKind.ERRORS_AND_WARNINGS);
    Reporter reporter = new Reporter(new EventBus(), sensor);
    reporter.handle(Event.error(location, "An ERROR event."));
    reporter.handle(Event.error(location, "Another ERROR event."));
    reporter.handle(Event.warn(location, "A warning event."));
    reporter.handle(Event.info(location, "An info event.")); // not in mask
    assertEquals(3, sensor.getTriggerCount());
    assertTrue(sensor.wasTriggered());
  }

}
