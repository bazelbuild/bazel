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
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
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
  public void testCcToolchainLabelFromAttributes() throws Exception {
    scratch.file(
        "cc/BUILD",
        "filegroup(name='empty')",
        "filegroup(name='everything')",
        "TOOLCHAIN_NAMES = [",
        "  'darwin-from-crosstool',",
        "  'windows-from-crosstool',",
        "  'k8-compiler',",
        "  'k8-default-from-cpu',",
        "  'ppc-compiler',",
        "  'ppc-default-from-cpu']",
        "[cc_toolchain(",
        "    name = NAME,",
        "    cpu = 'banana',",
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
        ") for NAME in TOOLCHAIN_NAMES]",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = {",
        "       'k8|compiler': ':k8-compiler',",
        "       'ppc': ':invalid-label',",
        "       'ppc|compiler': ':ppc-compiler',",
        "       'k8': ':k8-default-from-cpu',",
        "       'x64_windows' : ':windows-from-crosstool',",
        "       'darwin' : ':darwin-from-crosstool',",
        "       'x64_windows|compiler' : ':windows-from-crosstool',",
        "       'darwin|compiler' : ':darwin-from-crosstool',",
        "    },",
        "    proto = \"\"\"",
        "major_version: 'v1'",
        "minor_version: '0'",
        "default_target_cpu: 'k8'",
        "default_toolchain {",
        "  cpu: 'k8'",
        "  toolchain_identifier: 'k8-from-crosstool'",
        "}",
        "default_toolchain {",
        "  cpu: 'ppc'",
        "  toolchain_identifier: 'ppc-from-crosstool'",
        "}",
        "default_toolchain {",
        "  cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-from-crosstool'",
        "}",
        "default_toolchain {",
        "  cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-from-crosstool'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'k8-from-crosstool'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'ppc'",
        "  toolchain_identifier: 'ppc-from-crosstool'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-from-crosstool'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-from-crosstool'",
        "  host_system_name: 'windows'",
        "  target_system_name: 'windows'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "\"\"\"",
        ")");

    scratch.file("a/BUILD", "cc_binary(name='b', srcs=['b.cc'])");

    useConfiguration("--crosstool_top=//cc:suite", "--cpu=k8");
    ConfiguredTarget c = getConfiguredTarget("//a:b");
    CppConfiguration config = getConfiguration(c).getFragment(CppConfiguration.class);
    assertThat(config.getRuleProvidingCcToolchainProvider().toString())
        .isEqualTo("//cc:k8-default-from-cpu");

    useConfiguration("--crosstool_top=//cc:suite", "--compiler=compiler", "--cpu=ppc");
    config = getConfiguration(getConfiguredTarget("//a:b")).getFragment(CppConfiguration.class);
    assertThat(config.getRuleProvidingCcToolchainProvider().toString())
        .isEqualTo("//cc:ppc-compiler");

    useConfiguration("--crosstool_top=//cc:suite", "--compiler=compiler", "--cpu=k8");
    config = getConfiguration(getConfiguredTarget("//a:b")).getFragment(CppConfiguration.class);
    assertThat(config.getRuleProvidingCcToolchainProvider().toString())
        .isEqualTo("//cc:k8-compiler");

    try {
      useConfiguration("--crosstool_top=//cc:suite", "--cpu=ppc");
      getConfiguration(getConfiguredTarget("//a:b")).getFragment(CppConfiguration.class);
      fail("expected failure because 'ppc' entry points to an invalid label");
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("The label '//cc:invalid-label' is not a cc_toolchain rule");
    }
  }

  @Test
  public void testCcToolchainFromToolchainIdentifier() throws Exception {
    scratch.file(
        "cc/BUILD",
        "filegroup(name='empty')",
        "filegroup(name='everything')",
        "TOOLCHAIN_NAMES = [",
        "  'k8',",
        "  'ppc',",
        "  'darwin',",
        "  'windows']",
        "[cc_toolchain(",
        "    name = NAME,",
        "    cpu = 'banana',",
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
        ") for NAME in TOOLCHAIN_NAMES]",
        "[cc_toolchain(",
        "    name = NAME + '-override',",
        "    cpu = NAME,",
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
        "    toolchain_identifier = NAME + '-from-attribute'",
        ") for NAME in TOOLCHAIN_NAMES]",
        "cc_toolchain(",
        "    name = 'invalid',",
        "    cpu = 'banana',",
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
        "    toolchain_identifier = 'invalid-toolchain',",
        ")",
        "cc_toolchain(",
        "    name = 'duplicate',",
        "    cpu = 'banana',",
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
        "    toolchain_identifier = 'duplicate-toolchain',",
        ")",
        "cc_toolchain(",
        "    name = 'wrong-compiler',",
        "    cpu = 'k8',",
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
        "    toolchain_identifier = 'wrong-compiler-toolchain',",
        ")",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = {",
        "       'k8': ':k8-override',",
        "       'k8|compiler': ':k8',",
        "       'k8|compiler-from-attribute': ':k8-override',",
        "       'ppc|compiler': ':ppc',",
        "       'ppc|compiler-from-attribute': ':ppc-override',",
        "       'ppc_invalid|compiler': ':invalid',",
        "       'k8|compiler1': ':duplicate',",
        "       'k8|right-compiler': ':wrong-compiler',",
        "       'x64_windows' : ':windows',",
        "       'x64_windows|compiler' : ':windows',",
        "       'x64_windows|compiler-from-attribute' : ':windows',",
        "       'x64_windows|compiler1' : ':windows',",
        "       'x64_windows|right-compiler' : ':windows',",
        "       'darwin' : ':darwin',",
        "       'darwin|compiler' : ':darwin',",
        "       'darwin|compiler-from-attribute' : ':darwin',",
        "       'darwin|compiler1' : ':darwin',",
        "       'darwin|right-compiler' : ':darwin',",
        "    },",
        "    proto = \"\"\"",
        "major_version: 'v1'",
        "minor_version: '0'",
        "default_target_cpu: 'k8'",
        "default_toolchain {",
        "  cpu: 'k8'",
        "  toolchain_identifier: 'k8-toolchain-identifier'",
        "}",
        "default_toolchain {",
        "  cpu: 'ppc'",
        "  toolchain_identifier: 'ppc-toolchain-identifier'",
        "}",
        "default_toolchain {",
        "  cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-toolchain-identifier'",
        "}",
        "default_toolchain {",
        "  cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-toolchain-identifier'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'k8-toolchain-identifier'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler-from-attribute'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'k8-from-attribute'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'ppc'",
        "  toolchain_identifier: 'ppc-toolchain-identifier'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler-from-attribute'",
        "  target_cpu: 'ppc'",
        "  toolchain_identifier: 'ppc-from-attribute'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-toolchain-identifier'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler-from-attribute'",
        "  target_cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-from-attribute'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-toolchain-identifier'",
        "  host_system_name: 'windows'",
        "  target_system_name: 'windows'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler-from-attribute'",
        "  target_cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-from-attribute'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler1'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'duplicate-toolchain'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler2'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'duplicate-toolchain'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'wrong-compiler'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'wrong-compiler-toolchain'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "\"\"\"",
        ")");

    scratch.file("a/BUILD", "cc_binary(name='b', srcs=['b.cc'])");

    useConfiguration("--crosstool_top=//cc:suite", "--cpu=k8");
    ConfiguredTarget c = getConfiguredTarget("//a:b");
    CppConfiguration config = getConfiguration(c).getFragment(CppConfiguration.class);
    assertThat(config.getToolchainIdentifier()).isEqualTo("k8-from-attribute");

    useConfiguration("--crosstool_top=//cc:suite", "--cpu=ppc");
    c = getConfiguredTarget("//a:b");
    config = getConfiguration(c).getFragment(CppConfiguration.class);
    assertThat(config.getToolchainIdentifier()).isEqualTo("ppc-toolchain-identifier");

    useConfiguration(
        "--crosstool_top=//cc:suite", "--compiler=compiler-from-attribute", "--cpu=ppc");
    config = getConfiguration(getConfiguredTarget("//a:b")).getFragment(CppConfiguration.class);
    assertThat(config.getToolchainIdentifier()).isEqualTo("ppc-from-attribute");

    try {
      useConfiguration("--crosstool_top=//cc:suite", "--compiler=compiler", "--cpu=ppc_invalid");
      getConfiguration(getConfiguredTarget("//a:b")).getFragment(CppConfiguration.class);
      fail(
          "expected failure because ppc_invalid|compiler entry points to an invalid toolchain "
              + "identifier");
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Toolchain identifier 'invalid-toolchain' was not found");
    }

    try {
      useConfiguration("--crosstool_top=//cc:suite", "--compiler=compiler1", "--cpu=k8");
      getConfiguration(getConfiguredTarget("//a:b")).getFragment(CppConfiguration.class);
      fail("expected failure because of duplicate toolchain identifier");
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Multiple toolchains with 'duplicate-toolchain' identifier");
    }
  }

  @Test
  public void testDisableCcToolchainLabelFromCrosstoolFile() throws Exception {
    scratch.file(
        "cc/BUILD",
        "filegroup(name='empty')",
        "filegroup(name='everything')",
        "TOOLCHAIN_NAMES = [",
        "  'darwin',",
        "  'windows',",
        "  'k8',",
        "  'ppc-compiler',",
        "  'ppc']",
        "[cc_toolchain(",
        "    name = NAME,",
        "    cpu = 'banana',",
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
        ") for NAME in TOOLCHAIN_NAMES]",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = {",
        "       'ppc': ':ppc',",
        "       'ppc|compiler': ':ppc-compiler',",
        "       'k8': ':k8',",
        "       'x64_windows' : ':windows',",
        "       'darwin' : ':darwin',",
        "       'x64_windows|compiler' : ':windows',",
        "       'darwin|compiler' : ':darwin',",
        "    },",
        "    proto = \"\"\"",
        "major_version: 'v1'",
        "minor_version: '0'",
        "default_target_cpu: 'k8'",
        "default_toolchain {",
        "  cpu: 'k8'",
        "  toolchain_identifier: 'k8-from-crosstool'",
        "}",
        "default_toolchain {",
        "  cpu: 'ppc'",
        "  toolchain_identifier: 'ppc-from-crosstool'",
        "}",
        "default_toolchain {",
        "  cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-from-crosstool'",
        "}",
        "default_toolchain {",
        "  cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-from-crosstool'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'k8-from-crosstool'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'ppc'",
        "  toolchain_identifier: 'ppc-from-crosstool'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: 'local'",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-from-crosstool'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "toolchain {",
        "  compiler: 'compiler'",
        "  target_cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-from-crosstool'",
        "  host_system_name: 'windows'",
        "  target_system_name: 'windows'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "}",
        "\"\"\"",
        ")");

    scratch.file("a/BUILD", "cc_binary(name='b', srcs=['b.cc'])");

    try {
      useConfiguration(
          "--crosstool_top=//cc:suite",
          "--cpu=k8",
          "--compiler=compiler",
          "--incompatible_disable_cc_toolchain_label_from_crosstool_proto");
      getConfiguredTarget("//a:b");
      fail("Expected failure because selecting cc_toolchain label from CROSSTOOL is disabled");
    } catch (InvalidConfigurationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "you may want to add an entry for 'k8|compiler' into toolchains and "
                  + "toolchain_identifier 'k8-from-crosstool' into the corresponding "
                  + "cc_toolchain rule");
    }
  }
}
