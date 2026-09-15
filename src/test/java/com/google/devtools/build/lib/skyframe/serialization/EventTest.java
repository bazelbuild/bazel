// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Event} serialization.
 *
 * <p>At the time of this test class's writing there is no custom EventCodec. However, event
 * property value insertion order should not affect events' serialized representation, and this
 * tests for that.
 */
@RunWith(JUnit4.class)
public class EventTest {
  @Test
  public void smoke() throws Exception {
    Event propertylessEvent = Event.of(EventKind.INFO, "myMessage");
    Event byteArrayEvent = Event.of(EventKind.INFO, "myMessage".getBytes(UTF_8));
    Event labelEvent =
        Event.of(
            EventKind.WARNING,
            "myOtherMessage",
            Label.class,
            Label.create("myPackage", "myTarget"));
    Event labelStringEvent = labelEvent.withProperty(String.class, "myTag");

    new SerializationTester(propertylessEvent, byteArrayEvent, labelEvent, labelStringEvent)
        .runTests();
  }

  @Test
  public void serializationIsPropertyOrderAgnostic() throws Exception {
    Event labelStringEvent =
        Event.of(EventKind.WARNING, "myMessage")
            .withProperty(Label.class, Label.create("myPackage", "myTarget"))
            .withProperty(String.class, "myTag");

    Event stringLabelEvent =
        Event.of(EventKind.WARNING, "myMessage")
            .withProperty(String.class, "myTag")
            .withProperty(Label.class, Label.create("myPackage", "myTarget"));

    ObjectCodecs codecs = new ObjectCodecs(AutoRegistry.get());
    assertThat(codecs.serialize(labelStringEvent)).isEqualTo(codecs.serialize(stringLabelEvent));
  }
}
