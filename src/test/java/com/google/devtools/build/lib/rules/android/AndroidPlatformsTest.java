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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.packages.util.BazelMockAndroidSupport;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests Android support for Blaze's platforms API (https://bazel.build/concepts/platforms-intro).
 *
 * <p>This only provides the first-level of testing: that <code>--platforms</code> settings directly
 * impact toolchain selection in expected ways. Devs can lean on this test for quick interactive
 * feedback on their changes.
 */
@RunWith(JUnit4.class)
public class AndroidPlatformsTest extends AndroidBuildViewTestCase {

  private static final String EXTRA_SDK_TOOLCHAINS_FLAG =
      "--extra_toolchains=//platform_selected_android_sdks/toolchains:all";

  @Before
  public void setupPlatformsAndToolchains() throws Exception {
    scratch.file(
        "android_platforms/BUILD",
        "platform(",
        "    name = 'x86_platform',",
        "    constraint_values = [",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64',",
        "    ])",
        "platform(",
        "    name = 'arm_platform',",
        "    constraint_values = [",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7',",
        "    ])");
    BazelMockAndroidSupport.setupPlatformResolvableSdks(mockToolsConfig);

    analysisMock.setupMockClient(mockToolsConfig);
    // This line is necessary so an ARM C++ toolchain is available for dependencies under an Android
    // split transition.
    analysisMock.ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
  }

  @Test
  public void chooseSdk() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml')");

    useConfiguration(EXTRA_SDK_TOOLCHAINS_FLAG, "--platforms=//android_platforms:x86_platform");
    Artifact apkX86 =
        getImplicitOutputArtifact(
            getConfiguredTarget("//java/a:a"), AndroidRuleClasses.ANDROID_BINARY_APK);
    assertThat(getGeneratingSpawnActionArgs(apkX86).get(0))
        .isEqualTo("platform_selected_android_sdks/apksigner_x86_64");

    useConfiguration(EXTRA_SDK_TOOLCHAINS_FLAG, "--platforms=//android_platforms:arm_platform");
    Artifact apkArm =
        getImplicitOutputArtifact(
            getConfiguredTarget("//java/a:a"), AndroidRuleClasses.ANDROID_BINARY_APK);
    assertThat(getGeneratingSpawnActionArgs(apkArm).get(0))
        .isEqualTo("platform_selected_android_sdks/apksigner_arm");
  }

  @Test
  public void chooseNdk() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "cc_library(",
        "    name = 'cclib',",
        "    srcs  = ['cclib.cc'])",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    deps = [':cclib'],",
        "    manifest = 'AndroidManifest.xml')");

    useConfiguration(EXTRA_SDK_TOOLCHAINS_FLAG, "--platforms=//android_platforms:x86_platform");
    ConfiguredTarget x86Binary = getConfiguredTarget("//java/a:a");
    CppLinkAction x86Link =
        (CppLinkAction) getGeneratingAction(getPrerequisiteArtifacts(x86Binary, "deps").get(0));
    // TODO(blaze-team): replace with the commented line below when platform-based resolution works.
    assertThat(x86Link.getArguments().get(0)).isEqualTo("/usr/bin/mock-ar");
    // assertThat(cppLinkAction.getLinkCommandLine().getLinkerPathString())
    //    .isEqualTo("android/crosstool/x86/bin/i686-linux-android-ar");

    useConfiguration(EXTRA_SDK_TOOLCHAINS_FLAG, "--platforms=//android_platforms:arm_platform");
    ConfiguredTarget armBinary = getConfiguredTarget("//java/a:a");
    CppLinkAction armLink =
        (CppLinkAction) getGeneratingAction(getPrerequisiteArtifacts(armBinary, "deps").get(0));
    // TODO(blaze-team): replace with the commented line below when platform-based resolution works.
    assertThat(armLink.getArguments().get(0)).isEqualTo("/usr/bin/mock-ar");
    // assertThat(cppLinkAction.getLinkCommandLine().getLinkerPathString())
    //    .isEqualTo("android/crosstool/arm/bin/arm-linux-androideabi-ar");
  }
}
