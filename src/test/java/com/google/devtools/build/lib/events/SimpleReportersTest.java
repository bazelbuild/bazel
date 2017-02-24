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

import static org.junit.Assert.assertEquals;

import com.google.common.eventbus.EventBus;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link Reporter}.
 */
@RunWith(JUnit4.class)
public class SimpleReportersTest extends EventTestTemplate {

  private int handlerCount = 0;

  @Test
  public void addsHandlers() {
    EventHandler handler = new EventHandler() {
      @Override
      public void handle(Event event) {
        handlerCount++;
      }
    };

    Reporter reporter = new Reporter(new EventBus(), handler);
    reporter.handle(Event.info(location, "Add to handlerCount."));
    reporter.handle(Event.info(location, "Add to handlerCount."));
    reporter.handle(Event.info(location, "Add to handlerCount."));
    assertEquals(3, handlerCount);
  }
}
