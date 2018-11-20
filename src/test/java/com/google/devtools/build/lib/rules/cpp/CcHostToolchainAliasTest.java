// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for toolchain features. */
@RunWith(JUnit4.class)
public class CcHostToolchainAliasTest extends BuildViewTestCase {

  @Test
  public void testCcHostToolchainAliasRuleHasHostConfiguration() throws Exception {
    scratch.file("a/BUILD", "cc_host_toolchain_alias(name='current_cc_host_toolchain')");

    ConfiguredTarget target = getConfiguredTarget("//a:current_cc_host_toolchain");
    CcToolchainProvider toolchainProvider =
        (CcToolchainProvider) target.get(ToolchainInfo.PROVIDER);

    assertThat(toolchainProvider.isHostConfiguration()).isTrue();
  }

  @Test
  public void testThatHostCrosstoolTopCommandLineArgumentWorks() throws Exception {
    scratch.file(
        "b/BUILD",
        "cc_toolchain_suite(",
        "  name = 'my_custom_toolchain_suite',",
        "  toolchains = {",
        "    'k8': '//b:toolchain_b',",
        "    'k8|gcc-4.4.0': '//b:toolchain_b',",
        "    'k8|compiler': '//b:toolchain_b',",
        "    'x64_windows|windows_msys64': '//b:toolchain_b',",
        "    'darwin|compiler': '//b:toolchain_b',",
        "})",
        "cc_toolchain(",
        "    name = 'toolchain_b',",
        "    toolchain_identifier = 'toolchain-identifier-k8',",
        "    cpu = 'ED-E',",
        "    all_files = ':banana',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'])");
    scratch.file("b/CROSSTOOL", AnalysisMock.get().ccSupport().readCrosstoolFile());

    scratch.file("a/BUILD", "cc_host_toolchain_alias(name='current_cc_host_toolchain')");

    useConfiguration("--host_crosstool_top=//b:my_custom_toolchain_suite", "--host_cpu=k8");
    ConfiguredTarget target = getConfiguredTarget("//a:current_cc_host_toolchain");

    assertThat(target.getLabel())
        .isEqualTo(Label.parseAbsoluteUnchecked("//b:my_custom_toolchain_suite"));
  }
}
