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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import java.util.List;
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
                .addCompilerFlag("-foo_compiler")
                .addCxxFlag("-foo_cxx")
                .setBuiltinSysroot("/usr/local/custom-sysroot")
                .addToolPath(ToolPath.newBuilder().setName("ar").setPath("foo/ar/path").build())
                .buildPartial());
    useConfiguration(
        "--cpu=k8", "--force_pic", "--noincompatible_disable_legacy_flags_cc_toolchain_api");
    scratch.file(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  provider = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  return struct(",
        "    dirs = provider.built_in_include_directories,",
        "    link_options = provider.link_options_do_not_use,",
        "    unfiltered_compiler_options = provider.unfiltered_compiler_options([]),",
        "    sysroot = provider.sysroot,",
        "    cpu = provider.cpu,",
        "    compiler_options = provider.compiler_options(),",
        "    cxx_options = provider.cxx_options(),",
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

    @SuppressWarnings("unchecked")
    List<String> compilerOptions = (List<String>) ct.get("compiler_options");
    assertThat(compilerOptions).contains("-foo_compiler");

    @SuppressWarnings("unchecked")
    List<String> cxxOptions = (List<String>) ct.get("cxx_options");
    assertThat(cxxOptions).contains("-foo_cxx");

    assertThat((String) ct.get("ar_executable")).endsWith("foo/ar/path");

    assertThat(ct.get("cpu")).isEqualTo("k8");

    assertThat(ct.get("sysroot")).isEqualTo("/usr/local/custom-sysroot");

    @SuppressWarnings("unchecked")
    List<String> linkOptions = (List<String>) ct.get("link_options");
    assertThat(linkOptions).contains("--sysroot=/usr/local/custom-sysroot");

    @SuppressWarnings("unchecked")
    List<String> unfilteredCompilerOptions = (List<String>) ct.get("unfiltered_compiler_options");
    assertThat(unfilteredCompilerOptions).contains("--sysroot=/usr/local/custom-sysroot");

    @SuppressWarnings("unchecked")
    boolean usePicForDynamicLibraries = (boolean) ct.get("use_pic_for_dynamic_libraries");
    assertThat(usePicForDynamicLibraries).isTrue();
  }

  @Test
  public void testDisablingMostlyStaticLinkOptions() throws Exception {
    testDisablingLinkingApiMethod("provider.mostly_static_link_options(False)");
  }

  @Test
  public void testDisablingFullyStaticLinkOptions() throws Exception {
    testDisablingLinkingApiMethod("provider.fully_static_link_options(True)");
  }

  @Test
  public void testDisablingDynamicLinkOptions() throws Exception {
    testDisablingLinkingApiMethod("provider.dynamic_link_options(False)");
  }

  @Test
  public void testDisablingLinkOptions() throws Exception {
    testDisablingLinkingApiMethod("provider.link_options_do_not_use");
  }

  private void testDisablingLinkingApiMethod(String method) throws Exception {
    useConfiguration("--experimental_disable_legacy_cc_linking_api");
    testDisablingLinkingApiMethodWithConfiguration(method);
    useConfiguration("--incompatible_disable_legacy_flags_cc_toolchain_api");
    testDisablingLinkingApiMethodWithConfiguration(method);
  }

  private void testDisablingLinkingApiMethodWithConfiguration(String method) throws Exception {
    scratch.overwriteFile(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  provider = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  return struct(",
        "    link_options = " + method + ",",
        "  )",
        "",
        "my_rule = rule(",
        "  _impl,",
        "  fragments = [ 'cpp' ],",
        "  attrs = {'_cc_toolchain': attr.label(default=Label('//test:toolchain')) }",
        ")");
    scratch.overwriteFile(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "cc_toolchain_alias(name = 'toolchain')",
        "my_rule(name = 'target')");
    invalidatePackages();
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:target");
    assertContainsEvent("Skylark APIs accessing linking flags has been removed.");
  }

  @Test
  public void testDisablingCompilerOptions() throws Exception {
    testDisablingCompilationApiMethod("provider.compiler_options()");
  }

  @Test
  public void testDisablingCxxOptions() throws Exception {
    testDisablingCompilationApiMethod("provider.cxx_options()");
  }

  @Test
  public void testDisablingCOptions() throws Exception {
    testDisablingCompilationApiMethod("provider.c_options()");
  }

  @Test
  public void testDisablingUnfilteredOptions() throws Exception {
    testDisablingCompilationApiMethod("provider.unfiltered_compiler_options([])");
  }
  private void testDisablingCompilationApiMethod(String method) throws Exception {
    useConfiguration("--experimental_disable_legacy_cc_compilation_api");
    testDisablingCompilationApiMethodWithConfiguration(method);
    useConfiguration("--incompatible_disable_legacy_flags_cc_toolchain_api");
    testDisablingCompilationApiMethodWithConfiguration(method);
  }

  private void testDisablingCompilationApiMethodWithConfiguration(String method) throws Exception {
    scratch.overwriteFile(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  provider = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  return struct(",
        "    compile_options = " + method + ",",
        "  )",
        "",
        "my_rule = rule(",
        "  _impl,",
        "  fragments = [ 'cpp' ],",
        "  attrs = {'_cc_toolchain': attr.label(default=Label('//test:toolchain')) }",
        ")");
    scratch.overwriteFile(
        "test/BUILD",
        "load(':rule.bzl', 'my_rule')",
        "cc_toolchain_alias(name = 'toolchain')",
        "my_rule(name = 'target')");
    invalidatePackages();
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:target");
    assertContainsEvent("Skylark APIs accessing compilation flags has been removed.");
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

    useConfiguration("-c", "opt", "--noincompatible_disable_legacy_flags_cc_toolchain_api");
    CcToolchainProvider ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getCompilerOptions()).contains("-foo_from_compilation_mode");
    assertThat(ccToolchainProvider.getLegacyCxxOptions()).contains("-bar_from_compilation_mode");
    assertThat(ccToolchainProvider.getLegacyMostlyStaticLinkFlags(CompilationMode.OPT))
        .contains("-baz_from_compilation_mode");

    useConfiguration(
        "-c",
        "opt",
        "--experimental_disable_compilation_mode_flags",
        "--noincompatible_disable_legacy_flags_cc_toolchain_api");
    ccToolchainProvider = getCcToolchainProvider();
    assertThat(ccToolchainProvider.getCompilerOptions())
        .doesNotContain("-foo_from_compilation_mode");
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
}
