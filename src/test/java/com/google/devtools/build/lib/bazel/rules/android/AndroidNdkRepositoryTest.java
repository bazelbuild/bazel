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
package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RepositoryFetchException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.util.BazelMockCcSupport;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidNdkRepositoryTest}. */
@RunWith(JUnit4.class)
public class AndroidNdkRepositoryTest extends BuildViewTestCase {
  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    return builder.addRuleDefinition(new AndroidNdkRepositoryRule()).build();
  }

  @Before
  public void setup() throws Exception {
    BazelMockCcSupport.INSTANCE.setup(mockToolsConfig);
    scratch.overwriteFile("bazel_tools_workspace/tools/build_defs/cc/BUILD");
    scratch.overwriteFile(
        "bazel_tools_workspace/tools/build_defs/cc/action_names.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/action_names.bzl"));

    scratch.overwriteFile(
        "bazel_tools_workspace/tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/cc_toolchain_config_lib.bzl"));
    scratch.file("/ndk/source.properties", "Pkg.Desc = Android NDK", "Pkg.Revision = 13.1.3345770");
  }

  private void scratchPlatformsDirectories(String arch, int... apiLevels) throws Exception {
    for (int apiLevel : apiLevels) {
      scratch.dir("/ndk/platforms/android-" + apiLevel);
      scratch.file(
          String.format("/ndk/platforms/android-%s/%s/usr/lib/libandroid.so", apiLevel, arch));
    }
  }

  @Test
  public void testApiLevelHighestVersionDetection() throws Exception {
    scratchPlatformsDirectories("arch-x86", 19, 20, 22, 24);
    String bazelToolsWorkspace = scratch.dir("bazel_tools_workspace").getPathString();
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(name = 'bazel_tools', path = '" + bazelToolsWorkspace + "')",
        "android_ndk_repository(",
        "    name = 'androidndk',",
        "    path = '/ndk',",
        ")");
    invalidatePackages();

    NestedSet<Artifact> x86ClangHighestApiLevelFilesToRun =
        getConfiguredTarget("@androidndk//:x86-clang3.8-gnu-libstdcpp-all_files")
            .getProvider(FilesToRunProvider.class)
            .getFilesToRun();
    assertThat(artifactsToStrings(x86ClangHighestApiLevelFilesToRun))
        .contains(
            "src external/androidndk/ndk/platforms/android-24/arch-x86/usr/lib/libandroid.so");
    assertThat(artifactsToStrings(x86ClangHighestApiLevelFilesToRun))
        .doesNotContain(
            "src external/androidndk/ndk/platforms/android-22/arch-x86/usr/lib/libandroid.so");
  }

  @Test
  public void testInvalidNdkReleaseTxt() throws Exception {
    scratchPlatformsDirectories("arch-x86", 24);
    String bazelToolsWorkspace = scratch.dir("bazel_tools_workspace").getPathString();
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(name = 'bazel_tools', path = '" + bazelToolsWorkspace + "')",
        "android_ndk_repository(",
        "    name = 'androidndk',",
        "    path = '/ndk',",
        "    api_level = 24,",
        ")");

    scratch.deleteFile("/ndk/source.properties");
    scratch.file("/ndk/RELEASE.TXT", "not a valid release string");

    invalidatePackages();

    assertThat(getConfiguredTarget("@androidndk//:files")).isNotNull();
    MoreAsserts.assertContainsEvent(
        eventCollector,
        "The revision of the Android NDK referenced by android_ndk_repository rule 'androidndk' "
            + "could not be determined (the revision string found is 'not a valid release string')."
            + " Bazel will attempt to treat the NDK as if it was r21.");
  }

  @Test
  public void testInvalidNdkSourceProperties() throws Exception {
    scratchPlatformsDirectories("arch-x86", 24);
    String bazelToolsWorkspace = scratch.dir("bazel_tools_workspace").getPathString();
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(name = 'bazel_tools', path = '" + bazelToolsWorkspace + "')",
        "android_ndk_repository(",
        "    name = 'androidndk',",
        "    path = '/ndk',",
        "    api_level = 24,",
        ")");

    scratch.overwriteFile(
        "/ndk/source.properties",
        "Pkg.Desc = Android NDK",
        "Pkg.Revision = invalid package revision");

    invalidatePackages();

    assertThat(getConfiguredTarget("@androidndk//:files")).isNotNull();
    MoreAsserts.assertContainsEvent(
        eventCollector,
        "The revision of the Android NDK referenced by android_ndk_repository rule 'androidndk' "
            + "could not be determined (the revision string found is 'invalid package revision'). "
            + "Bazel will attempt to treat the NDK as if it was r21.");
  }

  @Test
  public void testUnsupportedNdkVersion() throws Exception {
    scratchPlatformsDirectories("arch-x86", 24);
    String bazelToolsWorkspace = scratch.dir("bazel_tools_workspace").getPathString();
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(name = 'bazel_tools', path = '" + bazelToolsWorkspace + "')",
        "android_ndk_repository(",
        "    name = 'androidndk',",
        "    path = '/ndk',",
        "    api_level = 24,",
        ")");

    scratch.overwriteFile(
        "/ndk/source.properties", "Pkg.Desc = Android NDK", "Pkg.Revision = 22.0.3675639-beta2");
    invalidatePackages();

    assertThat(getConfiguredTarget("@androidndk//:files")).isNotNull();
    MoreAsserts.assertContainsEvent(
        eventCollector,
        "The major revision of the Android NDK referenced by android_ndk_repository rule "
            + "'androidndk' is 22. The major revisions supported by Bazel are "
            + "[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]. "
            + "Bazel will attempt to treat the NDK as if it was r21.");
  }

  @Test
  public void testMiscLibraries() throws Exception {
    scratchPlatformsDirectories("arch-x86", 19, 20, 22, 24);
    scratch.file(String.format("/ndk/sources/android/cpufeatures/cpu-features.c"));
    scratch.file(String.format("/ndk/sources/android/cpufeatures/cpu-features.h"));
    String bazelToolsWorkspace = scratch.dir("bazel_tools_workspace").getPathString();
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(name = 'bazel_tools', path = '" + bazelToolsWorkspace + "')",
        "android_ndk_repository(",
        "    name = 'androidndk',",
        "    path = '/ndk',",
        ")");
    invalidatePackages();

    ConfiguredTargetAndData cpufeatures = getConfiguredTargetAndData("@androidndk//:cpufeatures");
    assertThat(cpufeatures).isNotNull();
    Rule rule = cpufeatures.getTarget().getAssociatedRule();
    assertThat(rule.isAttributeValueExplicitlySpecified("srcs")).isTrue();
    assertThat(rule.getAttr("srcs").toString())
        .isEqualTo("[@androidndk//:ndk/sources/android/cpufeatures/cpu-features.c]");
    assertThat(rule.isAttributeValueExplicitlySpecified("hdrs")).isTrue();
    assertThat(rule.getAttr("hdrs").toString())
        .isEqualTo("[@androidndk//:ndk/sources/android/cpufeatures/cpu-features.h]");
  }

  @Test
  public void testMissingPlatformsDirectory() throws Exception {
    String bazelToolsWorkspace = scratch.dir("bazel_tools_workspace").getPathString();
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "local_repository(name = 'bazel_tools', path = '" + bazelToolsWorkspace + "')",
        "android_ndk_repository(",
        "    name = 'androidndk',",
        "    path = '/ndk',",
        ")");
    invalidatePackages(false);
    RepositoryFetchException e =
        assertThrows(RepositoryFetchException.class, () -> getTarget("@androidndk//:files"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "Expected directory at /ndk/platforms but it is not a directory or it does not "
                + "exist.");
  }
}
