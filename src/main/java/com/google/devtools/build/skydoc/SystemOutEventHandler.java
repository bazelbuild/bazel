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

package com.google.devtools.build.skydoc;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

/**
 * A simple {@link EventHandler} which outputs log information to system.out and system.err.
 */
class SystemOutEventHandler implements EventHandler {

  @Override
  public void handle(Event event) {
    switch (event.getKind()) {
      case ERROR:
      case WARNING:
      case STDERR:
        System.err.println(messageWithLocation(event));
        break;
      case DEBUG:
      case INFO:
      case PROGRESS:
      case STDOUT:
        System.out.println(messageWithLocation(event));
        break;
      default:
        System.err.println("Unknown message type: " + event);
    }
  }

  private String messageWithLocation(Event event) {
    String location =
        event.getLocation() == null ? "<no location>" : event.getLocation().toString();
    return location + ": " + event.getMessage();
  }
}
