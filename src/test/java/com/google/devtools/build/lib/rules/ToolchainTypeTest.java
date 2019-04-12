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
package com.google.devtools.build.lib.rules;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the {@code toolchain_type} rule. */
@RunWith(JUnit4.class)
public class ToolchainTypeTest extends BuildViewTestCase {

  @Test
  public void testCcTargetsDependOnCcToolchainAutomatically() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "   name='empty')",
        "package(default_visibility=['//visibility:public'])",
        "constraint_setting(name = 'mock_setting')",
        "constraint_value(name = 'mock_value', constraint_setting = ':mock_setting')",
        "platform(",
        "   name = 'mock-platform',",
        "   constraint_values = [':mock_value'],",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    proto='''",
        "      toolchain_identifier: 'banana'",
        "      abi_version: 'banana'",
        "      abi_libc_version: 'banana'",
        "      compiler: 'banana'",
        "      host_system_name: 'banana'",
        "      target_system_name: 'banana'",
        "      target_cpu: 'banana'",
        "      target_libc: 'banana'",
        "    ''')",
        "toolchain(",
        "   name = 'toolchain_b',",
        "   toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "   toolchain = ':b',",
        "   target_compatible_with = [':mock_value'],",
        ")");

    useConfiguration(
        "--incompatible_enable_cc_toolchain_resolution",
        "--experimental_platforms=//a:mock-platform",
        "--extra_toolchains=//a:toolchain_b",
        "--noincompatible_disable_crosstool_file");

    // for cc_library, cc_binary, and cc_test, we check that $(TARGET_CPU) is a valid Make variable
    ConfiguredTarget cclibrary =
        ScratchAttributeWriter.fromLabelString(this, "cc_library", "//cclib")
            .setList("srcs", "a.cc")
            .setList("copts", "foobar-$(ABI)")
            .write();
    CppCompileAction compileAction =
        (CppCompileAction) getGeneratingAction(getBinArtifact("_objs/cclib/a.o", cclibrary));
    assertThat(compileAction.getArguments()).contains("foobar-banana");

    ConfiguredTarget ccbinary =
        ScratchAttributeWriter.fromLabelString(this, "cc_binary", "//ccbin")
            .setList("srcs", "a.cc")
            .setList("copts", "foobar-$(ABI)")
            .write();
    compileAction =
        (CppCompileAction) getGeneratingAction(getBinArtifact("_objs/ccbin/a.o", ccbinary));
    assertThat(compileAction.getArguments()).contains("foobar-banana");

    ConfiguredTarget cctest =
        ScratchAttributeWriter.fromLabelString(this, "cc_test", "//cctest")
            .setList("srcs", "a.cc")
            .setList("copts", "foobar-$(ABI)")
            .write();
    compileAction =
        (CppCompileAction) getGeneratingAction(getBinArtifact("_objs/cctest/a.o", cctest));
    assertThat(compileAction.getArguments()).contains("foobar-banana");
  }
}
