// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@code cc_toolchain_suite} rule.
 */
@RunWith(JUnit4.class)
public class CcToolchainSuiteTest extends BuildViewTestCase {

  @Test
  public void testCcToolchainLabelFromCpuCompilerAttributes() throws Exception {
    scratch.file(
        "cc/BUILD",
        "filegroup(name='empty')",
        "filegroup(name='everything')",
        "cc_toolchain(",
        "    name = 'cc-compiler-fruitie',",
        "    cpu = 'banana',",
        "    compiler = 'avocado',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'],",
        ")",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = {",
        "       'k8': ':cc-compiler-fruitie',",
        "    },",
        "    proto = \"\"\"",
        "major_version: 'v1'",
        "minor_version: '0'",
        "toolchain {",
        "  compiler: 'avocado'",
        "  target_cpu: 'banana'",
        "  toolchain_identifier: 'not-used-identifier'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "}",
        // Stub toolchain to make default cc toolchains happy
        // TODO(b/113849758): Remove once CppConfiguration doesn't load packages
        "toolchain {",
        "  compiler: 'orange'",
        "  target_cpu: 'banana'",
        "  toolchain_identifier: 'toolchain-identifier-k8'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "}",
        "\"\"\"",
        ")");

    scratch.file("a/BUILD", "cc_binary(name='b', srcs=['b.cc'])");

    useConfiguration("--crosstool_top=//cc:suite", "--cpu=k8", "--host_cpu=k8");
    ConfiguredTarget c = getConfiguredTarget("//a:b");
    CppConfiguration config = getConfiguration(c).getFragment(CppConfiguration.class);
    assertThat(config.getRuleProvidingCcToolchainProvider().toString())
        .isEqualTo("//cc:cc-compiler-fruitie");

    useConfiguration(
        "--crosstool_top=//cc:suite",
        "--cpu=k8",
        "--host_cpu=k8",
        "--incompatible_provide_cc_toolchain_info_from_cc_toolchain_suite");
    c = getConfiguredTarget("//a:b");
    config = getConfiguration(c).getFragment(CppConfiguration.class);
    assertThat(config.getRuleProvidingCcToolchainProvider().toString()).isEqualTo("//cc:suite");
  }

  @Test
  public void testCcToolchainFromToolchainIdentifierOverridesCpuCompiler() throws Exception {
    scratch.file(
        "cc/BUILD",
        "filegroup(name='empty')",
        "filegroup(name='everything')",
        "cc_toolchain(",
        "    name = 'cc-compiler-fruitie',",
        "    toolchain_identifier = 'toolchain-identifier-fruitie',",
        "    cpu = 'banana',",
        "    compiler = 'avocado',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_libs = [':empty'],",
        "    static_runtime_libs = [':empty'],",
        ")",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = {",
        "       'k8': ':cc-compiler-fruitie',",
        "    },",
        "    proto = \"\"\"",
        "major_version: 'v1'",
        "minor_version: '0'",
        "toolchain {",
        "  compiler: 'avocado'",
        "  target_cpu: 'banana'",
        "  toolchain_identifier: 'boring-non-fuitie-identifier'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "}",
        "toolchain {",
        "  compiler: 'orange'",
        "  target_cpu: 'banana'",
        "  toolchain_identifier: 'toolchain-identifier-fruitie'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "}",
        // Stub toolchain to make default cc toolchains happy
        // TODO(b/113849758): Remove once CppConfiguration doesn't load packages
        "toolchain {",
        "  compiler: 'orange'",
        "  target_cpu: 'banana'",
        "  toolchain_identifier: 'toolchain-identifier-k8'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "}",
        "\"\"\"",
        ")");

    scratch.file("a/BUILD", "cc_binary(name='b', srcs=['b.cc'])");

    useConfiguration("--crosstool_top=//cc:suite", "--cpu=k8", "--host_cpu=k8");
    ConfiguredTarget c = getConfiguredTarget("//a:b");
    CppConfiguration config = getConfiguration(c).getFragment(CppConfiguration.class);
    assertThat(config.getToolchainIdentifier()).isEqualTo("toolchain-identifier-fruitie");

    useConfiguration(
        "--crosstool_top=//cc:suite",
        "--cpu=k8",
        "--host_cpu=k8",
        "--incompatible_provide_cc_toolchain_info_from_cc_toolchain_suite");
    c = getConfiguredTarget("//a:b");
    config = getConfiguration(c).getFragment(CppConfiguration.class);
    assertThat(config.getToolchainIdentifier()).isEqualTo("toolchain-identifier-fruitie");
  }
}
