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
  public void testSmoke() throws Exception {
    scratch.file("cc/BUILD",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = { ",
        "       'cpu': ':cc-toolchain', 'k8': ':cc-toolchain', 'darwin': ':cc-toolchain' ",
        "    },",
        "    proto = \"\"\"",
        "major_version: 'v1'",
        "minor_version: '0'",
        "default_target_cpu: 'cpu'",
        "default_toolchain {",
        "  cpu: 'cpu'",
        "  toolchain_identifier: 'cpu-toolchain'",
        "}",
        "default_toolchain {",
        "  cpu: 'darwin'",
        "  toolchain_identifier: 'cpu-toolchain'",
        "}",
        "default_toolchain {",
        "  cpu: 'k8'",
        "  toolchain_identifier: 'k8-toolchain'",
        "}",
        "toolchain {",
        "  compiler: 'cpu-compiler'",
        "  target_cpu: 'cpu'",
        "  toolchain_identifier: 'cpu-toolchain'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "  tool_path { name: 'cpu-compiler', path: 'cpu/compiler' }",
        "  tool_path { name: 'ar', path: 'cpu/ar' }",
        "  tool_path { name: 'cpp', path: 'cpu/cpp' }",
        "  tool_path { name: 'gcc', path: 'cpu/gcc' }",
        "  tool_path { name: 'gcov', path: 'cpu/gcov' }",
        "  tool_path { name: 'ld', path: 'cpu/ld' }",
        "  tool_path { name: 'nm', path: 'cpu/nm' }",
        "  tool_path { name: 'objcopy', path: 'cpu/objcopy' }",
        "  tool_path { name: 'objdump', path: 'cpu/objdump' }",
        "  tool_path { name: 'strip', path: 'cpu/strip' }",
        "}",
        "toolchain {",
        "  compiler: 'k8-compiler'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'k8-toolchain'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "  tool_path { name: 'k8-compiler', path: 'k8/compiler' }",
        "  tool_path { name: 'ar', path: 'k8/ar' }",
        "  tool_path { name: 'cpp', path: 'k8/cpp' }",
        "  tool_path { name: 'gcc', path: 'k8/gcc' }",
        "  tool_path { name: 'gcov', path: 'k8/gcov' }",
        "  tool_path { name: 'ld', path: 'k8/ld' }",
        "  tool_path { name: 'nm', path: 'k8/nm' }",
        "  tool_path { name: 'objcopy', path: 'k8/objcopy' }",
        "  tool_path { name: 'objdump', path: 'k8/objdump' }",
        "  tool_path { name: 'strip', path: 'k8/strip' }",
        "}",
        "\"\"\")",
        "cc_toolchain(",
        "    name = 'cc-toolchain',",
        "    module_map = 'map',",
        "    cpu = 'cpu',",
        "    compiler_files = 'compile',",
        "    dwp_files = 'dwp',",
        "    linker_files = 'link',",
        "    strip_files = ':strip',",
        "    objcopy_files = 'objcopy',",
        "    all_files = ':every-file',",
        "    dynamic_runtime_libs = ['dynamic-runtime-libs'],",
        "    static_runtime_libs = ['static-runtime-libs'])",
        "",
        "filegroup(name = 'every-file', srcs = ['//cc:everything'])");

    invalidatePackages();
    useConfiguration("--crosstool_top=//cc:suite");
    CppConfiguration cppConfig = getTargetConfiguration().getFragment(CppConfiguration.class);
    assertThat(cppConfig.getTargetCpu()).isEqualTo("cpu");
    assertThat(cppConfig.getAbi()).isEqualTo("cpu-abi");
  }
}
