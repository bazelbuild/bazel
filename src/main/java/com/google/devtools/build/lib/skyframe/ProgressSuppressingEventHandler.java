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

/**
 * Suppresses {@link #post} when the provided {@link ExtendedEventHandler.Postable} is a {@link
 * ProgressLike}, but otherwise delegates calls to its wrapped {@link ExtendedEventHandler}.
 */
class ProgressSuppressingEventHandler implements ExtendedEventHandler {
  private final ExtendedEventHandler delegate;

  ProgressSuppressingEventHandler(ExtendedEventHandler listener) {
    this.delegate = listener;
  }

  @Override
  public void post(Postable obj) {
    if (obj instanceof ProgressLike) {
      return;
    }
    delegate.post(obj);
  }

  @Override
  public void handle(Event event) {
    delegate.handle(event);
  }
}
