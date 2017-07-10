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

import com.google.common.collect.ImmutableList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the {@link StoredEventHandler} class. */
@RunWith(JUnit4.class)
public class StoredErrorEventHandlerTest {

  @Test
  public void hasErrors() {
    StoredEventHandler eventHandler = new StoredEventHandler();
    assertThat(eventHandler.hasErrors()).isFalse();
    eventHandler.handle(Event.warn("warning"));
    assertThat(eventHandler.hasErrors()).isFalse();
    eventHandler.handle(Event.info("info"));
    assertThat(eventHandler.hasErrors()).isFalse();
    eventHandler.handle(Event.error("error"));
    assertThat(eventHandler.hasErrors()).isTrue();
  }

  @Test
  public void replayOnWithoutEvents() {
    StoredEventHandler eventHandler = new StoredEventHandler();
    StoredEventHandler sink = new StoredEventHandler();

    eventHandler.replayOn(sink);
    assertThat(sink.isEmpty()).isTrue();
  }

  @Test
  public void replayOn() {
    StoredEventHandler eventHandler = new StoredEventHandler();
    StoredEventHandler sink = new StoredEventHandler();

    List<Event> events = ImmutableList.of(
        Event.warn("a"),
        Event.error("b"),
        Event.info("c"),
        Event.warn("d"));
    for (Event e : events) {
      eventHandler.handle(e);
    }

    eventHandler.replayOn(sink);
    assertThat(sink.getEvents()).isEqualTo(events);
  }
}
