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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.packages.util.MockPlatformSupport;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests Android support for Blaze's platforms API
 * (https://docs.bazel.build/versions/master/platforms-intro.html).
 *
 * <p>This only provides the first-level of testing: that <code>--platforms</code> settings directly
 * impact toolchain selection in expected ways. Devs can lean on this test for quick interactive
 * feedback on their changes.
 *
 * <p>More broadly, we also need to ensure a) all other Android tests continue to pass and b) CI
 * continues working.
 *
 * <p>For a), we need to split any tests that set legacy flags (<code>--android_sdk</code>, <code>
 * --android_cpu</code>, <code>--fat_apk_cpu</code>, <code>--android_crosstool_top</code>, etc.) to
 * also run with the equivalent <code>--platforms</code> settings.
 *
 * <p>For b), we similarly need to split CI projects that build with legacy flags to run with
 * equivalent platform settings.
 */
@RunWith(JUnit4.class)
public class AndroidPlatformsTest extends AndroidBuildViewTestCase {
  private static final ImmutableList<String> MOCK_SDK_VERSIONS = ImmutableList.of("foo", "bar");
  private static final ImmutableList<String> MOCK_NDK_CPUS = ImmutableList.of("x86", "armeabi-v7a");

  private static final String PLATFORM_TEMPLATE =
      String.join(
          "\n",
          "platform(",
          "    name = '%s',",
          "    constraint_values = [",
          "        '" + TestConstants.PLATFORM_PACKAGE_ROOT + "/java/constraints:java8',",
          "    ])");

  @Before
  public void writeMockPlatforms() throws Exception {
    MockPlatformSupport.setup(mockToolsConfig);
    // This line is necessary so an ARM C++ toolchain is available for dependencies under an Android
    // split transition. BazelMockAndroidSupport.setupNdk(mockToolsConfig) isn't sufficient for this
    // because that sets up the NDK in a special package //android/crosstool that tests then have to
    // manually trigger with --android_crosstool_top=//android/crosstool:everything. Since the point
    // of this test is to test that --platforms sets the correct NDK toolchain, we don't want these
    // tests to have to explicitly set --android_crosstool_top. Until --platforms correctly does
    // that, NDKs default to the default C++ toolchain. That's what this line configures.
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");

    for (String sdkVersion : MOCK_SDK_VERSIONS) {
      scratch.appendFile("platforms/BUILD", String.format(PLATFORM_TEMPLATE, "sdk_" + sdkVersion));
    }
    for (String ndkCpu : MOCK_NDK_CPUS) {
      scratch.appendFile("platforms/BUILD", String.format(PLATFORM_TEMPLATE, "ndk_" + ndkCpu));
    }
  }

  @Before
  public void writeMockSDKs() throws Exception {
    for (String sdkVersion : MOCK_SDK_VERSIONS) {
      scratch.appendFile(
          "sdk/BUILD",
          "android_sdk(",
          String.format("    name = 'sdk_%s',", sdkVersion),
          "    aapt = 'aapt',",
          "    aapt2 = 'aapt2',",
          "    adb = 'adb',",
          "    aidl = 'aidl',",
          "    android_jar = 'android.jar',",
          String.format("    apksigner = 'apksigner_%s',", sdkVersion),
          "    dx = 'dx',",
          "    framework_aidl = 'framework_aidl',",
          "    main_dex_classes = 'main_dex_classes',",
          "    main_dex_list_creator = 'main_dex_list_creator',",
          "    proguard = 'proguard',",
          "    shrinked_android_jar = 'shrinked_android_jar',",
          "    zipalign = 'zipalign',",
          "    tags = ['__ANDROID_RULES_MIGRATION__'])");
    }
  }

  @Test
  public void chooseSdk() throws Exception {
    scratch.file(
        "java/a/BUILD",
        "android_binary(",
        "    name = 'a',",
        "    srcs = ['A.java'],",
        "    manifest = 'AndroidManifest.xml')");

    useConfiguration("--platforms=//platforms:sdk_foo");
    Artifact apkFoo =
        getImplicitOutputArtifact(
            getConfiguredTarget("//java/a:a"), AndroidRuleClasses.ANDROID_BINARY_APK);
    // TODO(blaze-team): replace with the commented line below when platform-based resolution works.
    assertThat(getGeneratingSpawnActionArgs(apkFoo).get(0)).endsWith("/ApkSignerBinary");
    // assertThat(getGeneratingSpawnActionArgs(apkFoo).get(0)).isEqualTo("sdk/apksigner_foo");

    useConfiguration("--platforms=//platforms:sdk_bar");
    Artifact apkBar =
        getImplicitOutputArtifact(
            getConfiguredTarget("//java/a:a"), AndroidRuleClasses.ANDROID_BINARY_APK);
    // TODO(blaze-team): replace with the commented line below when platform-based resolution works.
    assertThat(getGeneratingSpawnActionArgs(apkBar).get(0)).endsWith("/ApkSignerBinary");
    // assertThat(getGeneratingSpawnActionArgs(apkBar).get(0)).isEqualTo("sdk/apksigner_foo");
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

    // See BazelMockAndroidSupport for the NDK toolchain this should imply. This replaces
    // "--fat_apk_cpu=x86", "--android_crosstool_top=//android/crosstool:everything".
    useConfiguration("--platforms=//platforms:ndk_x86");
    ConfiguredTarget x86Binary = getConfiguredTarget("//java/a:a");
    CppLinkAction x86Link =
        (CppLinkAction) getGeneratingAction(getPrerequisiteArtifacts(x86Binary, "deps").get(0));
    // TODO(blaze-team): replace with the commented line below when platform-based resolution works.
    assertThat(x86Link.getLinkCommandLine().getLinkerPathString()).isEqualTo("/usr/bin/mock-ar");
    // assertThat(cppLinkAction.getLinkCommandLine().getLinkerPathString())
    //    .isEqualTo("android/crosstool/x86/bin/i686-linux-android-ar");

    // See BazelMockAndroidSupport for the NDK toolchain this should imply. This replaces
    // "--fat_apk_cpu=armeabi-v7a", "--android_crosstool_top=//android/crosstool:everything".
    useConfiguration("--platforms=//platforms:ndk_armeabi-v7a");
    ConfiguredTarget armBinary = getConfiguredTarget("//java/a:a");
    CppLinkAction armLink =
        (CppLinkAction) getGeneratingAction(getPrerequisiteArtifacts(armBinary, "deps").get(0));
    // TODO(blaze-team): replace with the commented line below when platform-based resolution works.
    assertThat(armLink.getLinkCommandLine().getLinkerPathString()).isEqualTo("/usr/bin/mock-ar");
    // assertThat(cppLinkAction.getLinkCommandLine().getLinkerPathString())
    //    .isEqualTo("android/crosstool/arm/bin/arm-linux-androideabi-ar");
  }
}
