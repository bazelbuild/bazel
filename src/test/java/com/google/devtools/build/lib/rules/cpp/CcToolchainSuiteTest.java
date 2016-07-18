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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@code cc_toolchain_suite} rule.
 */
@RunWith(JUnit4.class)
public class CcToolchainSuiteTest extends BuildViewTestCase {
  @Test
  public void testFilesToBuild() throws Exception {
    scratch.file(
        "cc/BUILD",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = { ",
        "       'k8|k8-compiler': ':k8-toolchain',",
        "       'darwin|darwin-compiler': ':darwin-toolchain',",
        "       'x64_windows|windows-compiler': ':windows-toolchain',",
        "    },",
        "    proto = \"\"\"",
        "major_version: 'v1'",
        "minor_version: '0'",
        "default_target_cpu: 'k8'",
        "default_toolchain {",
        "  cpu: 'k8'",
        "  toolchain_identifier: 'k8-toolchain'",
        "}",
        "default_toolchain {",
        "  cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-toolchain'",
        "}",
        "default_toolchain {",
        "  cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-toolchain'",
        "}",
        "toolchain {",
        "  compiler: 'k8-compiler'",
        "  target_cpu: 'k8'",
        "  toolchain_identifier: 'k8-toolchain'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: 'cpu-abi'",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "  tool_path { name: 'cpu-compiler', path: 'k8/compiler' }",
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
        "toolchain {",
        "  compiler: 'darwin-compiler'",
        "  target_cpu: 'darwin'",
        "  toolchain_identifier: 'darwin-toolchain'",
        "  host_system_name: 'linux'",
        "  target_system_name: 'linux'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "  tool_path { name: 'darwin-compiler', path: 'darwin/compiler' }",
        "  tool_path { name: 'ar', path: 'darwin/ar' }",
        "  tool_path { name: 'cpp', path: 'darwin/cpp' }",
        "  tool_path { name: 'gcc', path: 'darwin/gcc' }",
        "  tool_path { name: 'gcov', path: 'darwin/gcov' }",
        "  tool_path { name: 'ld', path: 'darwin/ld' }",
        "  tool_path { name: 'nm', path: 'darwin/nm' }",
        "  tool_path { name: 'objcopy', path: 'darwin/objcopy' }",
        "  tool_path { name: 'objdump', path: 'darwin/objdump' }",
        "  tool_path { name: 'strip', path: 'darwin/strip' }",
        "}",
        "toolchain {",
        "  compiler: 'windows-compiler'",
        "  target_cpu: 'x64_windows'",
        "  toolchain_identifier: 'windows-toolchain'",
        "  host_system_name: 'windows'",
        "  target_system_name: 'windows'",
        "  abi_version: ''",
        "  abi_libc_version: ''",
        "  target_libc: ''",
        "  builtin_sysroot: 'sysroot'",
        "  default_grte_top: '//cc:grtetop'",
        "  tool_path { name: 'windows-compiler', path: 'windows/compiler' }",
        "  tool_path { name: 'ar', path: 'windows/ar' }",
        "  tool_path { name: 'cpp', path: 'windows/cpp' }",
        "  tool_path { name: 'gcc', path: 'windows/gcc' }",
        "  tool_path { name: 'gcov', path: 'windows/gcov' }",
        "  tool_path { name: 'ld', path: 'windows/ld' }",
        "  tool_path { name: 'nm', path: 'windows/nm' }",
        "  tool_path { name: 'objcopy', path: 'windows/objcopy' }",
        "  tool_path { name: 'objdump', path: 'windows/objdump' }",
        "  tool_path { name: 'strip', path: 'windows/strip' }",
        "}",
        "\"\"\")",
        "cc_toolchain(",
        "    name = 'k8-toolchain',",
        "    module_map = 'map',",
        "    cpu = 'cpu',",
        "    compiler_files = 'compile',",
        "    dwp_files = 'dwp',",
        "    linker_files = 'link',",
        "    strip_files = ':strip',",
        "    objcopy_files = 'objcopy',",
        "    all_files = ':k8-files',",
        "    dynamic_runtime_libs = ['k8-dynamic-runtime-libs'],",
        "    static_runtime_libs = ['k8-static-runtime-libs'])",
        "filegroup(",
        "    name = 'k8-files',",
        "    srcs = ['k8-marker', 'everything'])",
        "",
        "cc_toolchain(",
        "    name = 'darwin-toolchain',",
        "    module_map = 'map',",
        "    cpu = 'cpu',",
        "    compiler_files = 'compile',",
        "    dwp_files = 'dwp',",
        "    linker_files = 'link',",
        "    strip_files = ':strip',",
        "    objcopy_files = 'objcopy',",
        "    all_files = ':darwin-files',",
        "    dynamic_runtime_libs = ['darwin-dynamic-runtime-libs'],",
        "    static_runtime_libs = ['darwin-static-runtime-libs'])",
        "filegroup(",
        "    name = 'darwin-files',",
        "    srcs = ['darwin-marker', 'everything'])",
        "cc_toolchain(",
        "    name = 'windows-toolchain',",
        "    module_map = 'map',",
        "    cpu = 'cpu',",
        "    compiler_files = 'compile',",
        "    dwp_files = 'dwp',",
        "    linker_files = 'link',",
        "    strip_files = ':strip',",
        "    objcopy_files = 'objcopy',",
        "    all_files = ':windows-files',",
        "    dynamic_runtime_libs = ['windows-dynamic-runtime-libs'],",
        "    static_runtime_libs = ['windows-static-runtime-libs'])",
        "filegroup(",
        "    name = 'windows-files',",
        "    srcs = ['windows-marker', 'everything'])");

    scratch.file("a/BUILD",
        "genrule(name='a', srcs=[], outs=['ao'], tools=['//tools/defaults:crosstool'], cmd='x')");
    invalidatePackages();
    useConfiguration("--crosstool_top=//cc:suite");
    Action action = getGeneratingAction(getConfiguredTarget("//a:a"), "a/ao");
    assertThat(ActionsTestUtil.baseArtifactNames(action.getInputs()))
        .containsAllOf("k8-marker", "darwin-marker", "windows-marker");

    NestedSet<Artifact> suiteFiles = getFilesToBuild(getConfiguredTarget("//cc:suite"));
    assertThat(ActionsTestUtil.baseArtifactNames(suiteFiles))
        .containsAllOf("k8-marker", "darwin-marker", "windows-marker");
  }

  @Test
  public void testSmoke() throws Exception {
    scratch.file(
        "cc/BUILD",
        "cc_toolchain_suite(",
        "    name = 'suite',",
        "    toolchains = { ",
        "       'cpu|cpu-compiler': ':cc-toolchain',",
        "       'k8|k8-compiler': ':cc-toolchain',",
        "       'darwin|cpu-compiler': ':cc-toolchain',",
        "       'x64_windows|cpu-compiler': ':cc-toolchain',",
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
        "  cpu: 'x64_windows'",
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
    assertThat(cppConfig.getCompiler()).isEqualTo("cpu-compiler");
  }
}
