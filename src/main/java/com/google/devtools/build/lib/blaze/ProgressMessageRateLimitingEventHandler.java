// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.blaze;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;

/**
 * An event handler that rate limits events.
 */
public class ProgressMessageRateLimitingEventHandler implements EventHandler {

  private final EventHandler outputHandler;
  private final int rateLimitation;
  private final Clock clock;
  private long lastEvent = -1;

  @Override
  public boolean showOutput(String tag) {
    return true;
  }

  /**
   * Creates a new Event handler that rate limits the events of type PROGRESS
   * to one per event "rateLimitation" seconds.
   * All events that remain after rate limiting are forwarded to the handler "delegateTo".
   *
   * @param delegateTo  The event handler that ultimately handles the events
   * @param rateLimitation The number of seconds in which at most one event of any kind
   *                    in limitEvents will be forwarded to the delegateTo-handler.
   *                    If -1, all events will be forwarded.
   */
  public static EventHandler createRateLimitingEventHandler(EventHandler delegateTo,
      int rateLimitation) {
    if (rateLimitation == -1) {
      return delegateTo;
    }
    return new ProgressMessageRateLimitingEventHandler(delegateTo, rateLimitation);
  }

  private ProgressMessageRateLimitingEventHandler(EventHandler delegateTo,
      int rateLimitation) {
    clock = BlazeClock.instance();
    outputHandler = delegateTo;
    this.rateLimitation = rateLimitation * 1000;
  }

  @Override
  public void handle(Event event) {
    switch (event.getKind()) {
      case PROGRESS:
      case START:
      case FINISH:
        long currentTime = clock.currentTimeMillis();
        if (lastEvent + rateLimitation <= currentTime) {
          lastEvent = currentTime;
          outputHandler.handle(event);
        }
        break;
      default:
        outputHandler.handle(event);
        break;
    }
  }
}
