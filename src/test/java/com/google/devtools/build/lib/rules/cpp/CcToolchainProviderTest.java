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
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoMode;
import com.google.devtools.build.lib.vfs.PathFragment;
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
  public void equalityIsObjectIdentity() throws Exception {
    CcToolchainProvider a =
        new CcToolchainProvider(
            /* values= */ ImmutableMap.of(),
            /* cppConfiguration= */ null,
            /* toolchainInfo= */ null,
            /* crosstoolTopPathFragment= */ null,
            /* crosstool= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* crosstoolMiddleman= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* compile= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* strip= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* objCopy= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* as= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* ar= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* link= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* interfaceSoBuilder= */ null,
            /* dwp= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* coverage= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* libcLink= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* dynamicRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeSolibDir= */ PathFragment.EMPTY_FRAGMENT,
            CcCompilationContext.EMPTY,
            /* supportsParamFiles= */ false,
            /* supportsHeaderParsing= */ false,
            CcToolchainVariables.EMPTY,
            /* builtinIncludeFiles= */ ImmutableList.<Artifact>of(),
            /* coverageEnvironment= */ NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
            /* linkDynamicLibraryTool= */ null,
            /* builtInIncludeDirectories= */ ImmutableList.<PathFragment>of(),
            /* sysroot= */ null,
            FdoMode.OFF,
            /* useLLVMCoverageMapFormat= */ false,
            /* codeCoverageEnabled= */ false,
            /* isHostConfiguration= */ false);

    CcToolchainProvider b =
        new CcToolchainProvider(
            /* values= */ ImmutableMap.of(),
            /* cppConfiguration= */ null,
            /* toolchainInfo= */ null,
            /* crosstoolTopPathFragment= */ null,
            /* crosstool= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* crosstoolMiddleman= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* compile= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* strip= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* objCopy= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* as= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* ar= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* link= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* interfaceSoBuilder= */ null,
            /* dwp= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* coverage= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* libcLink= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* staticRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeLinkInputs= */ NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
            /* dynamicRuntimeLinkMiddleman= */ null,
            /* dynamicRuntimeSolibDir= */ PathFragment.EMPTY_FRAGMENT,
            CcCompilationContext.EMPTY,
            /* supportsParamFiles= */ false,
            /* supportsHeaderParsing= */ false,
            CcToolchainVariables.EMPTY,
            /* builtinIncludeFiles= */ ImmutableList.<Artifact>of(),
            /* coverageEnvironment= */ NestedSetBuilder.emptySet(Order.COMPILE_ORDER),
            /* linkDynamicLibraryTool= */ null,
            /* builtInIncludeDirectories= */ ImmutableList.<PathFragment>of(),
            /* sysroot= */ null,
            FdoMode.OFF,
            /* useLLVMCoverageMapFormat= */ false,
            /* codeCoverageEnabled= */ false,
            /* isHostConfiguration= */ false);

    new EqualsTester()
        .addEqualityGroup(a)
        .addEqualityGroup(b)
        .testEquals();
  }

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
    useConfiguration("--cpu=k8");
    scratch.file("test/rule.bzl",
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
  }
}
