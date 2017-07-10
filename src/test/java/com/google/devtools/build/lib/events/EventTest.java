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
import static java.nio.charset.StandardCharsets.UTF_8;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A super simple little test for the {@link Event} class.
 */
@RunWith(JUnit4.class)
public class EventTest extends EventTestTemplate {

  @Test
  public void eventRetainsEventKind() {
    assertThat(event.getKind()).isEqualTo(EventKind.WARNING);
  }

  @Test
  public void eventRetainsMessage() {
    assertThat(event.getMessage()).isEqualTo("This is not an error message.");
  }

  @Test
  public void eventRetainsLocation() {
    assertThat(event.getLocation().getStartOffset()).isEqualTo(21);
    assertThat(event.getLocation().getEndOffset()).isEqualTo(31);
  }

  @Test
  public void eventEncoding() {
    String message = "Bazel \u1f33f";
    Event ev1 = Event.of(EventKind.WARNING, null, message);
    assertThat(ev1.getMessage()).isEqualTo(message);
    assertThat(ev1.getMessageBytes()).isEqualTo(message.getBytes(UTF_8));
    Event ev2 = Event.of(EventKind.WARNING, null, message.getBytes(UTF_8));
    assertThat(ev2.getMessage()).isEqualTo(message);
    assertThat(ev2.getMessageBytes()).isEqualTo(message.getBytes(UTF_8));
  }
}
