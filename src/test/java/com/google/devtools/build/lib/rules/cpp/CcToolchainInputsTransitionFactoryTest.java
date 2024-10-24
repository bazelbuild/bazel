// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@code CcToolchainInputsTransitionFactory}. */
@RunWith(JUnit4.class)
public class CcToolchainInputsTransitionFactoryTest extends BuildViewTestCase {
  @Test
  public void testToolchain_usesTargetPlatform() throws Exception {
    scratch.file(
        "a/BUILD",
        """
        load(":cc_toolchain_config.bzl", "cc_toolchain_config")

        filegroup(
            name = "all_files",
            srcs = ["a.txt"],
        )

        cc_toolchain(
            name = "toolchain",
            all_files = ":all_files",
            ar_files = ":all_files",
            as_files = ":all_files",
            compiler_files = ":all_files",
            compiler_files_without_includes = ":all_files",
            dwp_files = ":all_files",
            linker_files = ":all_files",
            objcopy_files = ":all_files",
            strip_files = ":all_files",
            toolchain_config = ":does-not-matter-config",
            toolchain_identifier = "does-not-matter",
        )

        cc_toolchain_config(name = "does-not-matter-config")
        """);

    scratch.file("a/cc_toolchain_config.bzl", MockCcSupport.EMPTY_CC_TOOLCHAIN);

    ConfiguredTarget toolchainTarget = getConfiguredTarget("//a:toolchain");
    assertThat(toolchainTarget).isNotNull();

    ConfiguredTarget allFiles = getDirectPrerequisite(toolchainTarget, "//a:all_files");
    assertThat(allFiles).isNotNull();

    CoreOptions coreOptions = getConfiguration(allFiles).getOptions().get(CoreOptions.class);
    assertThat(coreOptions).isNotNull();
    assertThat(coreOptions.isExec).isFalse();
    // if isExec is false, then allFiles is building for the target platform
  }
}
