// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestStatus;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link TargetSummaryPublisher}. */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public final class TargetSummaryEventTest {
  private static final String PATH = "package";
  private static final String TARGET_NAME = "name";
  private static final String CHECKSUM = "abcdef";

  @Test
  public void testGetEventId() throws Exception {
    TargetSummaryEvent event =
        TargetSummaryEvent.create(target(PATH, TARGET_NAME, CHECKSUM), false, false, null);
    assertThat(event.getEventId())
        .isEqualTo(
            BuildEventIdUtil.targetSummary(
                Label.create(PATH, TARGET_NAME), BuildEventIdUtil.configurationId(CHECKSUM)));
  }

  @Test
  public void testGetEventId_nullConfig() throws Exception {
    TargetSummaryEvent event =
        TargetSummaryEvent.create(target(PATH, TARGET_NAME, null), false, false, null);
    assertThat(event.getEventId())
        .isEqualTo(
            BuildEventIdUtil.targetSummary(
                Label.create(PATH, TARGET_NAME), BuildEventIdUtil.nullConfigurationId()));
  }

  @Test
  public void testPostedAfter_noTestSummary() throws Exception {
    TargetSummaryEvent event = TargetSummaryEvent.create(stubTarget(), false, false, null);
    assertThat(event.postedAfter())
        .containsExactly(
            BuildEventIdUtil.targetCompleted(
                Label.create(PATH, TARGET_NAME), BuildEventIdUtil.configurationId(CHECKSUM)));
  }

  @Test
  public void testPostedAfter_expectTestSummary() throws Exception {
    TargetSummaryEvent event = TargetSummaryEvent.create(stubTarget(), false, true, null);
    assertThat(event.postedAfter())
        .containsExactly(
            BuildEventIdUtil.targetCompleted(
                Label.create(PATH, TARGET_NAME), BuildEventIdUtil.configurationId(CHECKSUM)),
            BuildEventIdUtil.testSummary(
                Label.create(PATH, TARGET_NAME), BuildEventIdUtil.configurationId(CHECKSUM)));
  }

  @Test
  public void testPostedAfter_nullConfig() throws Exception {
    TargetSummaryEvent event =
        TargetSummaryEvent.create(target(PATH, TARGET_NAME, null), false, true, null);
    assertThat(event.postedAfter())
        .containsExactly(
            BuildEventIdUtil.targetCompleted(
                Label.create(PATH, TARGET_NAME), BuildEventIdUtil.nullConfigurationId()),
            BuildEventIdUtil.testSummary(
                Label.create(PATH, TARGET_NAME), BuildEventIdUtil.nullConfigurationId()));
  }

  @Test
  public void testAsStreamProto_forTest() throws Exception {
    TargetSummaryEvent event =
        TargetSummaryEvent.create(stubTarget(), true, true, BlazeTestStatus.FLAKY);
    BuildEvent proto = event.asStreamProto(null);
    assertThat(proto.getId()).isEqualTo(event.getEventId());
    assertThat(proto.getTargetSummary().getOverallBuildSuccess()).isTrue();
    assertThat(proto.getTargetSummary().getOverallTestStatus()).isEqualTo(TestStatus.FLAKY);
  }

  @Test
  public void testAsStreamProto_forBuildSuccess() throws Exception {
    TargetSummaryEvent event = TargetSummaryEvent.create(stubTarget(), true, false, null);
    BuildEvent proto = event.asStreamProto(null);
    assertThat(proto.getId()).isEqualTo(event.getEventId());
    assertThat(proto.getTargetSummary().getOverallBuildSuccess()).isTrue();
    assertThat(proto.getTargetSummary().getOverallTestStatus()).isEqualTo(TestStatus.NO_STATUS);
  }

  @Test
  public void testAsStreamProto_failedBuildIgnoresTestResult() throws Exception {
    TargetSummaryEvent event =
        TargetSummaryEvent.create(stubTarget(), false, true, BlazeTestStatus.PASSED);
    BuildEvent proto = event.asStreamProto(null);
    assertThat(proto.getId()).isEqualTo(event.getEventId());
    assertThat(proto.getTargetSummary().getOverallBuildSuccess()).isFalse();
    assertThat(proto.getTargetSummary().getOverallTestStatus()).isEqualTo(TestStatus.NO_STATUS);
  }

  private static ConfiguredTarget stubTarget() throws Exception {
    return target(PATH, TARGET_NAME, CHECKSUM);
  }

  private static ConfiguredTarget target(String path, String targetName, String configChecksum)
      throws Exception {
    ConfiguredTarget target = mock(ConfiguredTarget.class);
    when(target.getOriginalLabel()).thenReturn(Label.create(path, targetName));
    when(target.getConfigurationChecksum()).thenReturn(configChecksum);
    return target;
  }
}
