// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.common.LostInputsEvent;

/**
 * Suppresses {@link #post} when the provided {@link Postable} represents a progress event (denoted
 * by a return of {@code false} from {@link Postable#storeForReplay}), but otherwise delegates calls
 * to its wrapped {@link ExtendedEventHandler}.
 */
final class ProgressSuppressingEventHandler implements ExtendedEventHandler {
  private final ExtendedEventHandler delegate;

  ProgressSuppressingEventHandler(ExtendedEventHandler listener) {
    this.delegate = listener;
  }

  @Override
  public void post(Postable obj) {
    // A LostInputsEvent isn't replayable, but still has to be delivered to ensure that rewound
    // actions can recover their lost inputs by triggering further rewinding.
    if (obj.storeForReplay() || obj instanceof LostInputsEvent) {
      delegate.post(obj);
    }
  }

  @Override
  public void handle(Event event) {
    delegate.handle(event);
  }
}
