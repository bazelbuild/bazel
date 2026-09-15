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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;

/**
 * An {@link EventHandler} that collects all events it encounters, and makes them available via the
 * {@link Iterable} interface. The collected events contain not just the original event information
 * but also the location context.
 */
public final class EventCollector extends AbstractEventHandler implements Iterable<Event> {
  private final Collection<Event> collected;

  /**
   * This collector will collect all events that match the event mask.
   */
  public EventCollector(Set<EventKind> mask) {
    this(mask, new ArrayList<Event>());
  }

  /**
   * This collector will collect all events.
   */
  public EventCollector() {
    this(EventKind.ALL_EVENTS, new ArrayList<Event>());
  }

  /**
   * This collector will collect all events that match the event mask.
   */
  public EventCollector(EventKind... mask) {
    this(ImmutableSet.copyOf(mask), new ArrayList<Event>());
  }

  /**
   * This collector will save the Event instances in the provided
   * collection.
   */
  public EventCollector(Set<EventKind> mask, Collection<Event> collected) {
    super(mask);
    this.collected = collected;
  }

  /**
   * Implements {@link EventHandler#handle(Event)}.
   */
  @Override
  public synchronized void handle(Event event) {
    if (getEventMask().contains(event.getKind())) {
      collected.add(event);
    }
    if (event.getStdErr() != null) {
      handle(Event.of(EventKind.STDERR, null, event.getStdErr()));
    }
    if (event.getStdOut() != null) {
      handle(Event.of(EventKind.STDOUT, null, event.getStdOut()));
    }
  }

  /**
   * Returns an iterator over the collected events. This must not be called in a scenario where
   * there may still be concurrent modifications to the collector.
   */
  @Override
  public Iterator<Event> iterator() {
    return collected.iterator();
  }

  /**
   * Returns an iterator over the collected events of the given kind. This must not be called in a
   * scenario where there may still be concurrent modifications to the collector.
   */
  public Iterable<Event> filtered(final EventKind eventKind) {
    return Iterables.filter(collected, event -> event.getKind() == eventKind);
  }

  /**
   * Returns the number of events collected.
   */
  public synchronized int count() {
    return collected.size();
  }

  /*
   * Clears the collected events
   */
  public synchronized void clear() {
    collected.clear();
  }

  @Override
  public synchronized String toString() {
    return "EventCollector: " + Iterables.toString(collected);
  }
}
