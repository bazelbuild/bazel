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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
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
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    return builder.addRuleDefinition(new AndroidNdkRepositoryRule()).build();
  }

  @Before
  public void setup() throws Exception {
    scratch.file(
        "/ndk/source.properties",
        "Pkg.Desc = Android NDK",
        "Pkg.Revision = 13.1.3345770");
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
    FileSystemUtils.appendIsoLatin1(
        scratch.resolve("WORKSPACE"),
        "android_ndk_repository(",
        "    name = 'androidndk',",
        "    path = '/ndk',",
        ")");
    invalidatePackages();

    ImmutableList<Artifact> x86ClangHighestApiLevelFilesToRun =
        getConfiguredTarget("@androidndk//:x86-clang3.8-gnu-libstdcpp-all_files")
            .getProvider(FilesToRunProvider.class).getFilesToRun();
    assertThat(artifactsToStrings(x86ClangHighestApiLevelFilesToRun))
        .contains(
            "src external/androidndk/ndk/platforms/android-24/arch-x86/usr/lib/libandroid.so");
    assertThat(artifactsToStrings(x86ClangHighestApiLevelFilesToRun))
        .doesNotContain(
            "src external/androidndk/ndk/platforms/android-22/arch-x86/usr/lib/libandroid.so");
  }
}
