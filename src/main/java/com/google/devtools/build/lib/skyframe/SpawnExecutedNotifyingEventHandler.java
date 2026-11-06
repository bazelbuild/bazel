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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * Notifies an {@link ImportantOutputHandler} of every {@link SpawnExecutedEvent} posted to it, then
 * delegates all calls to its wrapped {@link ExtendedEventHandler}.
 *
 * <p>This guarantees the handler observes every spawn even when the delegate suppresses the event,
 * like {@link ProgressSuppressingEventHandler} does for rewound actions).
 */
final class SpawnExecutedNotifyingEventHandler implements ExtendedEventHandler {
  private final ExtendedEventHandler delegate;
  private final ImportantOutputHandler importantOutputHandler;

  SpawnExecutedNotifyingEventHandler(
      ExtendedEventHandler delegate, ImportantOutputHandler importantOutputHandler) {
    this.delegate = checkNotNull(delegate);
    this.importantOutputHandler = checkNotNull(importantOutputHandler);
  }

  @Override
  public void post(Postable obj) {
    if (obj instanceof SpawnExecutedEvent spawnExecutedEvent) {
      importantOutputHandler.spawnExecuted(spawnExecutedEvent);
    }
    delegate.post(obj);
  }

  @Override
  public void handle(Event event) {
    delegate.handle(event);
  }
}
