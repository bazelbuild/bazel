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

import static com.google.common.truth.Truth.assertThat;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests the {@link EventCollector} class.
 */
@RunWith(JUnit4.class)
public class EventCollectorTest extends EventTestTemplate {

  @Test
  public void usesPassedInCollection() {
    Collection<Event> events = new ArrayList<>();
    EventCollector collector = new EventCollector(EventKind.ALL_EVENTS, events);
    collector.handle(event);
    Event onlyEvent = events.iterator().next();
    assertThat(onlyEvent.getMessage()).isEqualTo(event.getMessage());
    assertThat(onlyEvent.getLocation()).isSameInstanceAs(location);
    assertThat(onlyEvent.getKind()).isEqualTo(event.getKind());
    assertThat(onlyEvent.getLocation().getStartOffset())
        .isEqualTo(event.getLocation().getStartOffset());
    assertThat(collector.count()).isEqualTo(1);
    assertThat(events).hasSize(1);
  }

  @Test
  public void collectsEvents() {
    EventCollector collector = new EventCollector();
    collector.handle(event);
    Iterator<Event> collectedEventIt = collector.iterator();
    Event onlyEvent = collectedEventIt.next();
    assertThat(onlyEvent.getMessage()).isEqualTo(event.getMessage());
    assertThat(onlyEvent.getLocation()).isSameInstanceAs(location);
    assertThat(onlyEvent.getKind()).isEqualTo(event.getKind());
    assertThat(onlyEvent.getLocation().getStartOffset())
        .isEqualTo(event.getLocation().getStartOffset());
    assertThat(collectedEventIt.hasNext()).isFalse();
  }
}
