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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@code CcToolchainProvider}
 */
@RunWith(JUnit4.class)
public class CcToolchainProviderTest extends BuildViewTestCase {
  @Test
  public void testSkylarkCallables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            CrosstoolConfig.CToolchain.newBuilder()
                .setBuiltinSysroot("/usr/local/custom-sysroot")
                .addToolPath(ToolPath.newBuilder().setName("ar").setPath("foo/ar/path").build())
                .buildPartial());
    useConfiguration("--cpu=k8", "--force_pic");
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  provider = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  return struct(",
        "    dirs = provider.built_in_include_directories,",
        "    sysroot = provider.sysroot,",
        "    cpu = provider.cpu,",
        "    ar_executable = provider.ar_executable,",
        "    use_pic_for_dynamic_libraries = provider.use_pic_for_dynamic_libraries,",
        "  )",
        "",
        "my_rule = rule(",
        "  _impl,",
        "  attrs = {'_cc_toolchain': attr.label(default=Label('//test:toolchain')) }",
        ")");

    scratch.file("test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "cc_toolchain_alias(name = 'toolchain')",
        "my_rule(name = 'target')");

    ConfiguredTarget ct = getConfiguredTarget("//test:target");

    assertThat((String) ct.get("ar_executable")).endsWith("foo/ar/path");

    assertThat(ct.get("cpu")).isEqualTo("k8");

    assertThat(ct.get("sysroot")).isEqualTo("/usr/local/custom-sysroot");

    @SuppressWarnings("unchecked")
    boolean usePicForDynamicLibraries = (boolean) ct.get("use_pic_for_dynamic_libraries");
    assertThat(usePicForDynamicLibraries).isTrue();
  }

  @Test
  public void testDisablingCompilationModeFlags() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "compilation_mode_flags { mode: OPT compiler_flag: '-foo_from_compilation_mode' }",
            "compilation_mode_flags { mode: OPT cxx_flag: '-bar_from_compilation_mode' }",
            "compilation_mode_flags { mode: OPT linker_flag: '-baz_from_compilation_mode' }");
    scratch.file("a/BUILD", "cc_library(name='a', srcs=['a.cc'])");

    useConfiguration("-c", "opt");
    CcToolchainProvider ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getLegacyCompileOptionsWithCopts())
        .contains("-foo_from_compilation_mode");
    assertThat(ccToolchainProvider.getLegacyCxxOptions()).contains("-bar_from_compilation_mode");
    assertThat(ccToolchainProvider.getLegacyMostlyStaticLinkFlags(CompilationMode.OPT))
        .contains("-baz_from_compilation_mode");

    useConfiguration("-c", "opt", "--experimental_disable_compilation_mode_flags");
    ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getLegacyCxxOptions())
        .doesNotContain("-bar_from_compilation_mode");
    assertThat(ccToolchainProvider.getLegacyMostlyStaticLinkFlags(CompilationMode.OPT))
        .doesNotContain("-baz_from_compilation_mode");
  }

  private CcToolchainProvider getCcToolchainProvider() throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//a");
    RuleContext ruleContext = getRuleContext(target);
    return CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
  }

  @Test
  public void testDisablingLinkingModeFlags() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "linking_mode_flags { mode: MOSTLY_STATIC linker_flag: '-foo_from_linking_mode' }");
    scratch.file("a/BUILD", "cc_library(name='a', srcs=['a.cc'])");

    useConfiguration();
    CcToolchainProvider ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getLegacyMostlyStaticLinkFlags(CompilationMode.OPT))
        .contains("-foo_from_linking_mode");

    useConfiguration("--experimental_disable_linking_mode_flags");
    ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getLegacyMostlyStaticLinkFlags(CompilationMode.OPT))
        .doesNotContain("-foo_from_linking_mode");
  }

  @Test
  public void testDisablingLegacyCrosstoolFields() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "compiler_flag: '-foo_compiler_flag'",
            "cxx_flag: '-foo_cxx_flag'",
            "unfiltered_cxx_flag: '-foo_unfiltered_cxx_flag'",
            "linker_flag: '-foo_linker_flag'",
            "dynamic_library_linker_flag: '-foo_dynamic_library_linker_flag'",
            "test_only_linker_flag: '-foo_test_only_linker_flag'");
    scratch.file("a/BUILD", "cc_library(name='a', srcs=['a.cc'])");

    useConfiguration();
    CcToolchainProvider ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getLegacyCompileOptions()).contains("-foo_compiler_flag");
    assertThat(ccToolchainProvider.getLegacyCxxOptions()).contains("-foo_cxx_flag");
    assertThat(ccToolchainProvider.getUnfilteredCompilerOptions())
        .contains("-foo_unfiltered_cxx_flag");
    assertThat(ccToolchainProvider.getLegacyLinkOptions()).contains("-foo_linker_flag");
    assertThat(ccToolchainProvider.getSharedLibraryLinkOptions(/* flags= */ ImmutableList.of()))
        .contains("-foo_dynamic_library_linker_flag");
    assertThat(ccToolchainProvider.getTestOnlyLinkOptions()).contains("-foo_test_only_linker_flag");

    useConfiguration("--experimental_disable_legacy_crosstool_fields");
    ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getLegacyCompileOptions()).doesNotContain("-foo_compiler_flag");
    assertThat(ccToolchainProvider.getLegacyCxxOptions()).doesNotContain("-foo_cxx_flag");
    assertThat(ccToolchainProvider.getUnfilteredCompilerOptions())
        .doesNotContain("-foo_unfiltered_cxx_flag");
    assertThat(ccToolchainProvider.getLegacyLinkOptions()).doesNotContain("-foo_linker_flag");
    assertThat(ccToolchainProvider.getSharedLibraryLinkOptions(/* flags= */ ImmutableList.of()))
        .doesNotContain("-foo_dynamic_library_linker_flag");
    assertThat(ccToolchainProvider.getTestOnlyLinkOptions())
        .doesNotContain("-foo_test_only_linker_flag");
  }

  /*
   * Crosstools should load fine with or without 'gcov-tool'. Those that define 'gcov-tool'
   * should also add a make variable.
   */
  @Test
  public void testOptionalGcovTool() throws Exception {
    // Crosstool without gcov-tool
    scratch.file(
        "a/BUILD",
        "filegroup(",
        "   name='empty')",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    dynamic_runtime_lib = ':empty',",
        "    static_runtime_lib = ':empty',",
        "    proto=\"\"\"",
        "      feature { name: 'no_legacy_features' }",
        "      tool_path { name: 'gcc' path: 'path-to-gcc-tool' }",
        "      tool_path { name: 'ar' path: 'ar' }",
        "      tool_path { name: 'cpp' path: 'cpp' }",
        "      tool_path { name: 'gcov' path: 'gcov' }",
        "      tool_path { name: 'ld' path: 'ld' }",
        "      tool_path { name: 'nm' path: 'nm' }",
        "      tool_path { name: 'objdump' path: 'objdump' }",
        "      tool_path { name: 'strip' path: 'strip' }",
        "      toolchain_identifier: \"banana\"",
        "      abi_version: \"banana\"",
        "      abi_libc_version: \"banana\"",
        "      compiler: \"banana\"",
        "      host_system_name: \"banana\"",
        "      target_system_name: \"banana\"",
        "      target_cpu: \"banana\"",
        "      target_libc: \"banana\"",
        "    \"\"\")");
    useConfiguration("--cpu=k8", "--host_cpu=k8");
    CcToolchainProvider ccToolchainProvider =
        (CcToolchainProvider) getConfiguredTarget("//a:a").get(ToolchainInfo.PROVIDER);
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    ccToolchainProvider.addGlobalMakeVariables(builder);
    assertThat(builder.build().get("GCOVTOOL")).isNull();

    // Crosstool with gcov-tool
    scratch.file(
        "b/BUILD",
        "filegroup(",
        "   name='empty')",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    proto=\"\"\"",
        "      feature { name: 'no_legacy_features' }",
        "      tool_path { name: 'gcc' path: 'path-to-gcc-tool' }",
        "      tool_path { name: 'gcov-tool' path: 'path-to-gcov-tool' }",
        "      tool_path { name: 'ar' path: 'ar' }",
        "      tool_path { name: 'cpp' path: 'cpp' }",
        "      tool_path { name: 'gcov' path: 'gcov' }",
        "      tool_path { name: 'ld' path: 'ld' }",
        "      tool_path { name: 'nm' path: 'nm' }",
        "      tool_path { name: 'objdump' path: 'objdump' }",
        "      tool_path { name: 'strip' path: 'strip' }",
        "      toolchain_identifier: \"banana\"",
        "      abi_version: \"banana\"",
        "      abi_libc_version: \"banana\"",
        "      compiler: \"banana\"",
        "      host_system_name: \"banana\"",
        "      target_system_name: \"banana\"",
        "      target_cpu: \"banana\"",
        "      target_libc: \"banana\"",
        "    \"\"\")");
    useConfiguration("--cpu=k8", "--host_cpu=k8");
    ccToolchainProvider =
        (CcToolchainProvider) getConfiguredTarget("//b:a").get(ToolchainInfo.PROVIDER);
    builder = ImmutableMap.builder();
    ccToolchainProvider.addGlobalMakeVariables(builder);
    assertThat(builder.build().get("GCOVTOOL")).isNotNull();
  }

  @Test
  public void testUnsupportedSysrootErrorMessage() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(name='empty') ",
        "filegroup(name='everything')",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        // Not specifying `builtin_sysroot` means the toolchain doesn't support --grte_top,
        "\"\"\")");
    reporter.removeHandler(failFastHandler);
    useConfiguration("--grte_top=//a", "--cpu=k8", "--host_cpu=k8");
    getConfiguredTarget("//a:a");
    assertContainsEvent("does not support setting --grte_top");
  }

  @Test
  public void testConfigWithMissingToolDefs() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(name='empty') ",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        "      tool_path { name: 'gcc' path: 'path-to-gcc-tool' }",
        "      tool_path { name: 'ar' path: 'ar' }",
        "      tool_path { name: 'cpp' path: 'cpp' }",
        "      tool_path { name: 'gcov' path: 'gcov' }",
        "      tool_path { name: 'ld' path: 'ld' }",
        "      tool_path { name: 'nm' path: 'nm' }",
        "      tool_path { name: 'objdump' path: 'objdump' }",
        // "      tool_path { name: 'strip' path: 'strip' }",
        "\"\"\")");
    reporter.removeHandler(failFastHandler);
    useConfiguration("--cpu=k8", "--host_cpu=k8");
    getConfiguredTarget("//a:a");
    assertContainsEvent("Tool path for 'strip' is missing");
  }

  /** For a fission-supporting crosstool: check the dwp tool path. */
  @Test
  public void testFissionConfigWithMissingDwp() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(name='empty') ",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        "      supports_fission: 1",
        "      tool_path { name: 'gcc' path: 'path-to-gcc-tool' }",
        "      tool_path { name: 'ar' path: 'ar' }",
        "      tool_path { name: 'cpp' path: 'cpp' }",
        "      tool_path { name: 'gcov' path: 'gcov' }",
        "      tool_path { name: 'ld' path: 'ld' }",
        "      tool_path { name: 'nm' path: 'nm' }",
        "      tool_path { name: 'objdump' path: 'objdump' }",
        "      tool_path { name: 'strip' path: 'strip' }",
        "\"\"\")");
    reporter.removeHandler(failFastHandler);
    useConfiguration("--cpu=k8", "--host_cpu=k8");
    getConfiguredTarget("//a:a");
    assertContainsEvent("Tool path for 'dwp' is missing");
  }

  @Test
  public void testRuntimeLibsAttributesAreNotObligatory() throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(name='empty') ",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        "\"\"\")");
    reporter.removeHandler(failFastHandler);
    useConfiguration("--cpu=k8", "--host_cpu=k8");
    getConfiguredTarget("//a:a");
    assertNoEvents();
  }

  @Test
  public void testWhenRuntimeLibsAtttributesMandatoryWhenSupportsEmbeddedRuntimes()
      throws Exception {
    scratch.file(
        "a/BUILD",
        "filegroup(name='empty') ",
        "cc_toolchain_suite(",
        "    name = 'a',",
        "    toolchains = { 'k8': ':b', 'k9': ':c' },",
        ")",
        "cc_toolchain(",
        "    name = 'b',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        "      supports_embedded_runtimes: true,",
        "\"\"\")",
        "cc_toolchain(",
        "    name = 'c',",
        "    cpu = 'banana',",
        "    all_files = ':empty',",
        "    ar_files = ':empty',",
        "    as_files = ':empty',",
        "    compiler_files = ':empty',",
        "    dwp_files = ':empty',",
        "    linker_files = ':empty',",
        "    strip_files = ':empty',",
        "    objcopy_files = ':empty',",
        "    static_runtime_libs = [ ':yolo' ],",
        "    proto = \"\"\"",
        "      toolchain_identifier: \"a\"",
        "      host_system_name: \"a\"",
        "      target_system_name: \"a\"",
        "      target_cpu: \"a\"",
        "      target_libc: \"a\"",
        "      compiler: \"a\"",
        "      abi_version: \"a\"",
        "      abi_libc_version: \"a\"",
        "      supports_embedded_runtimes: true,",
        "\"\"\")");
    reporter.removeHandler(failFastHandler);
    useConfiguration("--cpu=k8", "--host_cpu=k8");
    getConfiguredTarget("//a:a");
    assertContainsEvent(
        "Toolchain supports embedded runtimes, but didn't provide static_runtime_lib attribute.");

    useConfiguration("--cpu=k9", "--host_cpu=k9");
    getConfiguredTarget("//a:a");
    assertContainsEvent(
        "Toolchain supports embedded runtimes, but didn't provide dynamic_runtime_lib attribute.");
  }
}
