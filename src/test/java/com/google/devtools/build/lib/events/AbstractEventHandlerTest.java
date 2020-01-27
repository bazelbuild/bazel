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

import static com.google.common.truth.Truth.assertThat;

import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link AbstractEventHandler}. */
@RunWith(JUnit4.class)
public class AbstractEventHandlerTest {

  private static AbstractEventHandler create(Set<EventKind> mask) {
    return new AbstractEventHandler(mask) {
        @Override
        public void handle(Event event) {}
      };
  }

  @Test
  public void retainsEventMask() {
    assertThat(create(EventKind.ALL_EVENTS).getEventMask()).isEqualTo(EventKind.ALL_EVENTS);
    assertThat(create(EventKind.ERRORS_AND_WARNINGS).getEventMask())
        .isEqualTo(EventKind.ERRORS_AND_WARNINGS);
    assertThat(create(EventKind.ERRORS).getEventMask()).isEqualTo(EventKind.ERRORS);
  }

}
