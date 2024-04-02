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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the output groups of cc_library. */
@RunWith(JUnit4.class)
public class CppOutputGroupsTest extends BuildViewTestCase {

  @Test
  public void testStaticLibraryOnlyOutputGroups() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));
    scratch.file("src.cc");
    scratch.file(
        "a/BUILD",
        """
        cc_library(
            name = "lib",
            srcs = ["src.cc"],
            linkstatic = 1,
            alwayslink = 0,
        )

        filegroup(
            name = "group_archive",
            srcs = [":lib"],
            output_group = "archive",
        )

        filegroup(
            name = "group_dynamic",
            srcs = [":lib"],
            output_group = "dynamic_library",
        )
        """);

    ConfiguredTarget groupArchive = getConfiguredTarget("//a:group_archive");
    ConfiguredTarget groupDynamic = getConfiguredTarget("//a:group_dynamic");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupArchive)))
        .containsExactly("a/liblib.a");
    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupDynamic))).isEmpty();
  }

  @Test
  public void testSharedLibraryOnlyOutputGroups() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));
    scratch.file("src.cc");
    scratch.file(
        "a/BUILD",
        """
        cc_library(
            name = "lib",
            srcs = ["src.cc"],
            linkstatic = 1,
            alwayslink = 1,
        )

        filegroup(
            name = "group_archive",
            srcs = [":lib"],
            output_group = "archive",
        )

        filegroup(
            name = "group_dynamic",
            srcs = [":lib"],
            output_group = "dynamic_library",
        )
        """);

    ConfiguredTarget groupArchive = getConfiguredTarget("//a:group_archive");
    ConfiguredTarget groupDynamic = getConfiguredTarget("//a:group_dynamic");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupArchive)))
        .containsExactly("a/liblib.lo");
    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupDynamic))).isEmpty();
  }

  @Test
  public void testStaticAndDynamicLibraryOutputGroups() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));
    scratch.file("src.cc");
    scratch.file(
        "a/BUILD",
        """
        cc_library(
            name = "lib",
            srcs = ["src.cc"],
            linkstatic = 0,
            alwayslink = 0,
        )

        filegroup(
            name = "group_archive",
            srcs = [":lib"],
            output_group = "archive",
        )

        filegroup(
            name = "group_dynamic",
            srcs = [":lib"],
            output_group = "dynamic_library",
        )
        """);

    ConfiguredTarget groupArchive = getConfiguredTarget("//a:group_archive");
    ConfiguredTarget groupDynamic = getConfiguredTarget("//a:group_dynamic");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupArchive)))
        .containsExactly("a/liblib.a");
    // If supports_interface_shared_objects is true, .ifso could also be generated.
    // So we here use contains instead containsExactly.
    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupDynamic)))
        .contains("a/liblib.so");
  }

  @Test
  public void testSharedAndDynamicLibraryOutputGroups() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_DYNAMIC_LINKER));
    scratch.file("src.cc");
    scratch.file(
        "a/BUILD",
        """
        cc_library(
            name = "lib",
            srcs = ["src.cc"],
            linkstatic = 0,
            alwayslink = 1,
        )

        filegroup(
            name = "group_archive",
            srcs = [":lib"],
            output_group = "archive",
        )

        filegroup(
            name = "group_dynamic",
            srcs = [":lib"],
            output_group = "dynamic_library",
        )
        """);

    ConfiguredTarget groupArchive = getConfiguredTarget("//a:group_archive");
    ConfiguredTarget groupDynamic = getConfiguredTarget("//a:group_dynamic");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupArchive)))
        .containsExactly("a/liblib.lo");
    // If supports_interface_shared_objects is true, .ifso could also be generated.
    // So we here use contains instead containsExactly.
    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupDynamic)))
        .contains("a/liblib.so");
  }

  @Test
  public void testModuleOutputGroups() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures("header_modules_feature_configuration"));
    scratch.file("header.h");
    scratch.file(
        "a/BUILD",
        """
        cc_library(
            name = "lib",
            hdrs = ["src.h"],
            features = ["header_modules"],
        )

        filegroup(
            name = "group_modules",
            srcs = [":lib"],
            output_group = "module_files",
        )
        """);

    ConfiguredTarget groupArchive = getConfiguredTarget("//a:group_modules");

    assertThat(ActionsTestUtil.prettyArtifactNames(getFilesToBuild(groupArchive)))
        .containsExactly("a/_objs/lib/lib.pcm");
  }
}
