// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link PackageProgressReceiver}.
 */
@RunWith(JUnit4.class)
public class PackageProgressReceiverTest {

  @Test
  public void testPackageVisible() {
    // If there is only a single package being loaded, it is visible in
    // the activity part of the progress state.
    PackageIdentifier id = PackageIdentifier.createInMainRepo("foo/bar/baz");
    PackageProgressReceiver progress = new PackageProgressReceiver();
    progress.startReadPackage(id);
    String activity = progress.progressState().getSecond();

    assertWithMessage("Unfinished package '" + id + "' should be visible in activity: " + activity)
        .that(activity.contains(id.toString()))
        .isTrue();
  }

  @Test
  public void testPackageCounted() {
    // If the loading of a package is completed, it is no longer visible as activity,
    // but counted as one package fully loaded.
    PackageIdentifier id = PackageIdentifier.createInMainRepo("foo/bar/baz");
    PackageProgressReceiver progress = new PackageProgressReceiver();
    progress.startReadPackage(id);
    progress.doneReadPackage(id);
    String state = progress.progressState().getFirst();
    String activity = progress.progressState().getSecond();

    assertWithMessage(
            "Finished package '" + id + "' should not be visible in activity: " + activity)
        .that(activity.contains(id.toString()))
        .isFalse();
    assertWithMessage("Number of completed packages should be visible in state")
        .that(state.contains("1 package"))
        .isTrue();
  }

  @Test
  public void testReset() {
    // After resetting, messages should be as immediately after creation.
    PackageProgressReceiver progress = new PackageProgressReceiver();
    String defaultState = progress.progressState().getFirst();
    String defaultActivity = progress.progressState().getSecond();
    PackageIdentifier id = PackageIdentifier.createInMainRepo("foo/bar/baz");
    progress.startReadPackage(id);
    progress.doneReadPackage(id);
    progress.reset();
    assertThat(progress.progressState().getFirst()).isEqualTo(defaultState);
    assertThat(progress.progressState().getSecond()).isEqualTo(defaultActivity);
  }
}
