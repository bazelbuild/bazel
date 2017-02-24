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

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;

/** Stores error and warning events, and later replays them. Thread-safe. */
public class StoredEventHandler implements ExtendedEventHandler {

  private final List<Event> events = new ArrayList<>();
  private final List<ExtendedEventHandler.Postable> posts = new ArrayList<>();
  private boolean hasErrors;

  public synchronized ImmutableList<Event> getEvents() {
    return ImmutableList.copyOf(events);
  }

  /** Returns true if there are no stored events. */
  public synchronized boolean isEmpty() {
    return events.isEmpty() && posts.isEmpty();
  }


  @Override
  public synchronized void handle(Event e) {
    hasErrors |= e.getKind() == EventKind.ERROR;
    events.add(e);
  }

  @Override
  public synchronized void post(ExtendedEventHandler.Postable e) {
    posts.add(e);
  }

  /** Replay all events stored in this object on the given eventHandler, in the same order. */
  public synchronized void replayOn(ExtendedEventHandler eventHandler) {
    Event.replayEventsOn(eventHandler, events);
    for (ExtendedEventHandler.Postable obj : posts) {
      eventHandler.post(obj);
    }
  }

  /**
   * Returns whether any of the events on this objects were errors.
   */
  public synchronized boolean hasErrors() {
    return hasErrors;
  }

  public synchronized void clear() {
    events.clear();
    posts.clear();
    hasErrors = false;
  }
}
