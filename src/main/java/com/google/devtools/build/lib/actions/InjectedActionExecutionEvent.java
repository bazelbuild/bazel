// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * Event fired when a remotely fetched action execution result is injected into the graph.
 *
 * <p>This event is only fired for actions that are remotely fetched and injected, preventing action
 * execution.
 */
public final class InjectedActionExecutionEvent implements ExtendedEventHandler.Postable {

  private final Action action;

  public InjectedActionExecutionEvent(Action action) {
    this.action = action;
  }

  public Action getAction() {
    return action;
  }
}
