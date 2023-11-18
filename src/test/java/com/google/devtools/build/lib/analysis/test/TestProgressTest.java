// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.analysis.test;

import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;

import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.TestProgressId;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TestProgressTest {
  @Test
  public void testTestProgress_convertsToEventId() {
    TestProgress progress =
        new TestProgress(
            "alabel", ConfigurationId.newBuilder().setId("configid").build(), 1, 2, 3, 4, "auri");

    BuildEventId id = progress.getEventId();

    assertThat(id)
        .isEqualTo(
            BuildEventId.newBuilder()
                .setTestProgress(
                    TestProgressId.newBuilder()
                        .setLabel("alabel")
                        .setConfiguration(ConfigurationId.newBuilder().setId("configid"))
                        .setRun(1)
                        .setShard(2)
                        .setAttempt(3)
                        .setOpaqueCount(4))
                .build());
  }

  @Test
  public void testTestProgress_convertsToEvent() {
    TestProgress progress =
        new TestProgress(
            "alabel", ConfigurationId.newBuilder().setId("configid").build(), 1, 2, 3, 4, "auri");

    BuildEvent event = progress.asStreamProto(null);

    assertThat(event)
        .isEqualTo(
            BuildEvent.newBuilder()
                .setId(
                    BuildEventId.newBuilder()
                        .setTestProgress(
                            TestProgressId.newBuilder()
                                .setLabel("alabel")
                                .setConfiguration(ConfigurationId.newBuilder().setId("configid"))
                                .setRun(1)
                                .setShard(2)
                                .setAttempt(3)
                                .setOpaqueCount(4)))
                .setTestProgress(BuildEventStreamProtos.TestProgress.newBuilder().setUri("auri"))
                .build());
  }
}
