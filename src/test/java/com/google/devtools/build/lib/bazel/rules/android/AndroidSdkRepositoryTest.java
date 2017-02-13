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

package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AndroidSdkRepositoryFunction}. */
@RunWith(JUnit4.class)
public class AndroidSdkRepositoryTest extends BuildViewTestCase {
  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    return builder.addRuleDefinition(new AndroidSdkRepositoryRule()).build();
  }

  @Before
  public void setup() throws Exception {
    scratch.setWorkingDir("/sdk");
    scratch.dir("platforms/android-25");
    scratch.file("extras/google/m2repository/com/google/android/foo/1.0.0/foo.pom",
        "<project>",
        "  <groupId>com.google.android</groupId>",
        "  <artifactId>foo</artifactId>",
        "  <version>1.0.0</version>",
        "  <packaging>aar</packaging>",
        "</project>");

    scratch.setWorkingDir("/bazel_tools_workspace");
    if (!scratch.resolve("WORKSPACE").exists()) {
      scratch.file("WORKSPACE");
    }
    if (!scratch.resolve("tools/android/BUILD").exists()) {
      scratch.file("tools/android/BUILD");
    }
    scratch.file(
        "tools/android/android_sdk_repository_template.bzl",
        "def create_android_sdk_rules(name, build_tools_version, build_tools_directory, ",
        "        api_levels, default_api_level):",
        "    pass",
        "def create_android_device_rules(system_image_dirs):",
        "    native.filegroup(",
        "        name = 'test_android_devices_filegroup',",
        "        srcs = system_image_dirs)");

    scratch.setWorkingDir("/workspace");
    FileSystemUtils.appendIsoLatin1(scratch.resolve("WORKSPACE"),
        "local_repository(name = 'bazel_tools', path = '/bazel_tools_workspace')",
        "android_sdk_repository(",
        "    name = 'androidsdk',",
        "    path = '/sdk',",
        "    build_tools_version = '25.0.0',",
        "    api_level = 25,",
        ")");
    invalidatePackages();
  }

  private void setupSystemImages() throws Exception {
    scratch.dir("/sdk/system-images/android-25/default/armeabi-v7a");
    scratch.dir("/sdk/system-images/android-24/google_apis/x86");
    scratch.dir("/sdk/system-images/android-24/google_apis/x86_64");
  }

  @Test
  public void testGeneratedAarImport() throws Exception {
    ConfiguredTarget aarImportTarget =
        getConfiguredTarget("@androidsdk//com.google.android:foo-1.0.0");
    assertThat(aarImportTarget).isNotNull();
    assertThat(aarImportTarget.getTarget().getAssociatedRule().getRuleClass())
        .isEqualTo("aar_import");
  }

  @Test
  public void testExportsFiles() throws Exception {
    ConfiguredTarget aarTarget = getConfiguredTarget(
        "@androidsdk//:extras/google/m2repository/com/google/android/foo/1.0.0/foo.aar");
    assertThat(aarTarget).isNotNull();
  }

  @Test
  public void testSystemImageDirectoriesAreFound() throws Exception {
    setupSystemImages();
    ConfiguredTarget androidDevicesFilegroupTarget =
        getConfiguredTarget("@androidsdk//:test_android_devices_filegroup");
    ImmutableList<Artifact> systemImagesDirectories =
        androidDevicesFilegroupTarget.getProvider(FilesToRunProvider.class).getFilesToRun();
    assertThat(artifactsToStrings(systemImagesDirectories))
        .containsExactly(
            "src external/androidsdk/system-images/android-25/default/armeabi-v7a",
            "src external/androidsdk/system-images/android-24/google_apis/x86",
            "src external/androidsdk/system-images/android-24/google_apis/x86_64");
  }

  @Test
  public void testNoSystemImageDirectory() throws Exception {
    ConfiguredTarget androidDevicesFilegroupTarget =
        getConfiguredTarget("@androidsdk//:test_android_devices_filegroup");
    assertThat(androidDevicesFilegroupTarget).isNotNull();
    assertThat(
        androidDevicesFilegroupTarget
            .getProvider(FilesToRunProvider.class)
            .getFilesToRun())
        .isEmpty();
  }
}
