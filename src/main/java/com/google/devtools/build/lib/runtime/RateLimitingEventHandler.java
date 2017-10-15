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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

/**
 * An event handler that rate limits events.
 */
public class RateLimitingEventHandler implements EventHandler {

  private final EventHandler outputHandler;
  private final double intervalMillis;
  private final Clock clock;
  private long lastEventMillis = -1;

  /**
   * Creates a new Event handler that rate limits the events of type PROGRESS
   * to one per event "rateLimitation" seconds.  Events that arrive too quickly are dropped;
   * all others are are forwarded to the handler "delegateTo".
   *
   * @param delegateTo  The event handler that ultimately handles the events
   * @param rateLimitation The minimum number of seconds between events that will be forwarded
   *                    to the delegateTo-handler.
   *                    If less than zero (or NaN), all events will be forwarded.
   */
  public static EventHandler create(EventHandler delegateTo, double rateLimitation) {
    if (rateLimitation < 0.0 || Double.isNaN(rateLimitation)) {
      return delegateTo;
    }
    return new RateLimitingEventHandler(delegateTo, rateLimitation);
  }

  private RateLimitingEventHandler(EventHandler delegateTo, double rateLimitation) {
    clock = BlazeClock.instance();
    outputHandler = delegateTo;
    this.intervalMillis = rateLimitation * 1000;
  }

  @Override
  public void handle(Event event) {
    switch (event.getKind()) {
      case PROGRESS:
      case START:
      case FINISH:
        long currentTime = clock.currentTimeMillis();
        if (lastEventMillis + intervalMillis <= currentTime) {
          lastEventMillis = currentTime;
          outputHandler.handle(event);
        }
        break;
      default:
        outputHandler.handle(event);
        break;
    }
  }
}
