// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.eventbus.EventBus;

/**
 * An event handler that forwards {@link Postable} events to an {@link EventBus}.
 *
 * <p>Lifetime: 1 command.
 */
public final class EventBusEventHandler implements ExtendedEventHandler {
  // Non-final for cleanup.
  private volatile EventBus eventBus;

  public EventBusEventHandler(EventBus eventBus) {
    this.eventBus = eventBus;
  }

  /** Creates a {@link EventBusEventHandler} with a new {@link EventBus} enclosed. */
  public static EventBusEventHandler createWithNewEventBus() {
    return new EventBusEventHandler(new EventBus());
  }

  @Override
  public void handle(Event event) {
    // Do nothing. We only handle {@link Postable} events.
  }

  @Override
  public void post(Postable obj) {
    if (eventBus != null) {
      eventBus.post(obj);
    }
  }

  @Override
  public void cleanup() {
    eventBus = null;
  }
}
