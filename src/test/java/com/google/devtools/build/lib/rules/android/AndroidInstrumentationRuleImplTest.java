// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link AndroidInstrumentation}.
 *
 * <p>Note that this class is not named {@code AndroidInstrumentationTest} per the usual naming
 * pattern due to a conflict with the upcoming {@link AndroidInstrumentationTest} class which
 * implements the {@code android_instrumentation_test} rule.
 */
@RunWith(JUnit4.class)
public class AndroidInstrumentationRuleImplTest extends AndroidBuildViewTestCase {

  @Before
  public void createFiles() throws Exception {
    scratch.file(
        "java/com/app/BUILD",
        "android_library(",
        "  name = 'lib',",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "android_binary(",
        "  name = 'app',",
        "  manifest = 'AndroidManifest.xml',",
        "  deps = [':lib'],",
        ")");
    scratch.file(
        "javatests/com/app/BUILD",
        "android_library(",
        "  name = 'lib',",
        "  manifest = 'AndroidManifest.xml',",
        ")",
        "android_binary(",
        "  name = 'instrumentation',",
        "  manifest = 'AndroidManifest.xml',",
        "  deps = [':lib'],",
        ")");
  }

  @Test
  public void testTargetAndInstrumentationAreBinaries() throws Exception {
    scratch.file(
        "javatests/com/app/instrumentation/BUILD",
        "android_instrumentation(",
        "  name = 'instrumentation',",
        "  target = '//java/com/app',",
        "  instrumentation = '//javatests/com/app:instrumentation',",
        ")");
    ConfiguredTarget instrumentation = getConfiguredTarget("//javatests/com/app/instrumentation");
    assertThat(instrumentation).isNotNull();
    AndroidInstrumentationInfoProvider instrumentationProvider =
        instrumentation.get(AndroidInstrumentationInfoProvider.ANDROID_INSTRUMENTATION_INFO);
    assertThat(instrumentationProvider.getTargetApk()).isNotNull();
    assertThat(instrumentationProvider.getTargetApk().prettyPrint())
        .isEqualTo("javatests/com/app/instrumentation/instrumentation-target.apk");
    assertThat(instrumentationProvider.getInstrumentationApk()).isNotNull();
    assertThat(instrumentationProvider.getInstrumentationApk().prettyPrint())
        .isEqualTo("javatests/com/app/instrumentation/instrumentation-instrumentation.apk");
  }

  // TODO(b/37856762): Re-enable and expand on this test when android_instrumentation is fullly
  // implemented to build APKs from libraries.
  @Ignore
  @Test
  public void testTargetAndInstrumentationAreLibraries() throws Exception {
    scratch.file(
        "javatests/com/app/instrumentation/BUILD",
        "android_instrumentation(",
        "  name = 'instrumentation',",
        "  target_library = '//java/com/app:lib',",
        "  instrumentation_library = '//javatests/com/app:lib',",
        ")");
    ConfiguredTarget instrumentation = getConfiguredTarget("//javatests/com/app/instrumentation");
    assertThat(instrumentation).isNotNull();
    AndroidInstrumentationInfoProvider instrumentationProvider =
        instrumentation.get(AndroidInstrumentationInfoProvider.ANDROID_INSTRUMENTATION_INFO);

    Artifact targetApk = instrumentationProvider.getTargetApk();
    assertThat(targetApk).isNotNull();
    assertThat(targetApk.prettyPrint())
        .isEqualTo("javatests/com/app/instrumentation/instrumentation-target.apk");
    SpawnAction targetSpawnAction = getGeneratingSpawnAction(targetApk);
    assertThat(targetSpawnAction).isNotNull();

    Artifact instrumentationApk = instrumentationProvider.getInstrumentationApk();
    assertThat(instrumentationApk).isNotNull();
    assertThat(instrumentationApk.prettyPrint())
        .isEqualTo("javatests/com/app/instrumentation/instrumentation-instrumentation.apk");
    SpawnAction instrumentationSpawnAction = getGeneratingSpawnAction(instrumentationApk);
    assertThat(instrumentationSpawnAction).isNotNull();
  }

  @Test
  public void testInvalidAttributeCombinations() throws Exception {
    checkError(
        "javatests/com/app/instrumentation1",
        "instrumentation",
        "android_instrumentation requires that exactly one of the instrumentation and "
            + "instrumentation_library attributes be specified.",
        "android_instrumentation(",
        "  name = 'instrumentation',",
        "  target = '//java/com/app',",
        ")");
    checkError(
        "javatests/com/app/instrumentation2",
        "instrumentation",
        "android_instrumentation requires that exactly one of the instrumentation and "
            + "instrumentation_library attributes be specified.",
        "android_instrumentation(",
        "  name = 'instrumentation',",
        "  target = '//java/com/app',",
        "  instrumentation = '//javatests/com/app:instrumentation',",
        "  instrumentation_library = '//javatests/com/app:lib',",
        ")");
    checkError(
        "javatests/com/app/instrumentation3",
        "instrumentation",
        "android_instrumentation requires that exactly one of the target and target_library "
            + "attributes be specified.",
        "android_instrumentation(",
        "  name = 'instrumentation',",
        "  instrumentation = '//javatests/com/app:instrumentation',",
        ")");
    checkError(
        "javatests/com/app/instrumentation4",
        "instrumentation",
        "android_instrumentation requires that exactly one of the target and target_library "
            + "attributes be specified.",
        "android_instrumentation(",
        "  name = 'instrumentation',",
        "  target = '//java/com/app',",
        "  target_library = '//java/com/app:lib',",
        "  instrumentation = '//javatests/com/app:instrumentation',",
        ")");
  }
}
