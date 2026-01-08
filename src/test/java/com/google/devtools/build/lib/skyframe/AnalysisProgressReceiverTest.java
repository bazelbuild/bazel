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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link AnalysisProgressReceiver}. */
@RunWith(JUnit4.class)
public class AnalysisProgressReceiverTest {

  @Test
  public void testTargetCounted() {
    // If the configuration of a target is completed it is counted as fully configured target.
    AnalysisProgressReceiver progress = new AnalysisProgressReceiver();
    progress.doneConfigureTarget();
    String progressString1 = progress.getProgressString();

    assertWithMessage("One configured target should be visible in progress.")
        .that(progressString1.contains("1 target configured"))
        .isTrue();

    progress.doneConfigureTarget();
    String progressString2 = progress.getProgressString();

    assertWithMessage("Two configured targets should be visible in progress.")
        .that(progressString2.contains("2 targets configured"))
        .isTrue();
  }

  @Test
  public void testDownloadedTargetCounted() {
    AnalysisProgressReceiver progress = new AnalysisProgressReceiver();
    progress.doneDownloadedConfiguredTarget();
    String progressString1 = progress.getProgressString();

    assertThat(progressString1).contains("1 target configured (1 remote cache hits)");

    progress.doneConfigureTarget();
    String progressString2 = progress.getProgressString();

    assertThat(progressString2).contains("2 targets configured (1 remote cache hits)");
  }

  @Test
  public void testAspectCounted() {
    AnalysisProgressReceiver progress = new AnalysisProgressReceiver();
    progress.doneConfigureAspect();
    String progressString1 = progress.getProgressString();

    assertWithMessage("One configured aspect should be visible in progress.")
        .that(progressString1.contains("0 targets configured, 1 aspect application"))
        .isTrue();

    progress.doneConfigureAspect();
    String progressString2 = progress.getProgressString();

    assertWithMessage("Two configured aspects should be visible in progress.")
        .that(progressString2.contains("0 targets configured, 2 aspect applications"))
        .isTrue();
  }

  @Test
  public void testDownloadedAspectCounted() {
    AnalysisProgressReceiver progress = new AnalysisProgressReceiver();
    progress.doneDownloadedConfiguredAspect();
    String progressString1 = progress.getProgressString();

    assertThat(progressString1)
        .contains("0 targets configured, 1 aspect application (1 remote cache hits)");

    progress.doneConfigureAspect();
    String progressString2 = progress.getProgressString();

    assertThat(progressString2)
        .contains("0 targets configured, 2 aspect applications (1 remote cache hits)");
  }

  @Test
  public void testTargetAndAspectCounted() {
    AnalysisProgressReceiver progress = new AnalysisProgressReceiver();
    String progressString1 = progress.getProgressString();
    assertThat(progressString1).contains("0 targets configured");

    progress.doneConfigureTarget();
    String progressString2 = progress.getProgressString();

    assertThat(progressString2).contains("1 target configured");

    progress.doneConfigureAspect();
    String progressString3 = progress.getProgressString();

    assertThat(progressString3).contains("1 target configured, 1 aspect application");
  }

  @Test
  public void testReset() {
    // After resetting, messages should be as immediately after creation.
    AnalysisProgressReceiver progress = new AnalysisProgressReceiver();
    String defaultProgress = progress.getProgressString();
    progress.doneConfigureTarget();
    assertThat(progress.getProgressString()).isNotEqualTo(defaultProgress);
    progress.reset();
    assertThat(progress.getProgressString()).isEqualTo(defaultProgress);
  }
}
