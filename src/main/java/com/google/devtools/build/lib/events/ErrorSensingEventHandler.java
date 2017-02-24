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

/**
 * Passes through any events, and keeps a flag if any of them were errors. It is thread-safe as long
 * as the target eventHandler is thread-safe.
 */
public final class ErrorSensingEventHandler extends DelegatingEventHandler {

  private volatile boolean hasErrors;

  public ErrorSensingEventHandler(ExtendedEventHandler eventHandler) {
    super(eventHandler);
  }

  @Override
  public void handle(Event e) {
    hasErrors |= e.getKind() == EventKind.ERROR;
    super.handle(e);
  }

  /**
   * Returns whether any of the events on this objects were errors.
   */
  public boolean hasErrors() {
    return hasErrors;
  }

  /**
   * Reset the error flag. Don't call this while other threads are accessing the same object.
   */
  public void resetErrors() {
    hasErrors = false;
  }
}
