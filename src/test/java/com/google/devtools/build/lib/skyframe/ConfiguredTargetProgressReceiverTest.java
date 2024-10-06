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

/** Tests {@link ConfiguredTargetProgressReceiver}. */
@RunWith(JUnit4.class)
public class ConfiguredTargetProgressReceiverTest {

  @Test
  public void testTargetCounted() {
    // If the configuration of a target is completed it is counted as fully configured target.
    ConfiguredTargetProgressReceiver progress = new ConfiguredTargetProgressReceiver();
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
  public void testReset() {
    // After resetting, messages should be as immediately after creation.
    ConfiguredTargetProgressReceiver progress = new ConfiguredTargetProgressReceiver();
    String defaultProgress = progress.getProgressString();
    progress.doneConfigureTarget();
    assertThat(progress.getProgressString()).isNotEqualTo(defaultProgress);
    progress.reset();
    assertThat(progress.getProgressString()).isEqualTo(defaultProgress);
  }
}
