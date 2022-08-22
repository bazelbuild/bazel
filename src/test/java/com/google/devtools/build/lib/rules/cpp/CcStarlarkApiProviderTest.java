// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark providers for cpp rules. */
@RunWith(JUnit4.class)
public class CcStarlarkApiProviderTest extends BuildViewTestCase {
  private CcStarlarkApiProvider getApi(String label) throws Exception {
    RuleConfiguredTarget rule = (RuleConfiguredTarget) getConfiguredTarget(label);
    return (CcStarlarkApiProvider) rule.get(CcStarlarkApiProvider.NAME);
  }

  private CcStarlarkApiInfo getApiForBuiltin(String label) throws Exception {
    RuleConfiguredTarget rule = (RuleConfiguredTarget) getConfiguredTarget(label);
    return (CcStarlarkApiInfo) rule.get(CcStarlarkApiProvider.NAME);
  }

  @Before
  public void setUp() throws Exception {
    MockProtoSupport.setupWorkspace(scratch);
    invalidatePackages();
  }

  @Test
  public void testDisableInCcLibrary() throws Exception {
    useConfiguration("--incompatible_disable_legacy_cc_provider");
    scratch.file("a/BUILD", "cc_library(name='a', srcs=['a.cc'])");
    assertThat(getApi("//a:a")).isNull();
  }

  @Test
  public void testDisableInCcBinary() throws Exception {
    useConfiguration("--incompatible_disable_legacy_cc_provider");
    scratch.file("a/BUILD", "cc_binary(name='a', srcs=['a.cc'])");
    assertThat(getApi("//a:a")).isNull();
  }

  @Test
  public void testDisableInCcImport() throws Exception {
    useConfiguration("--incompatible_disable_legacy_cc_provider");
    scratch.file("a/BUILD", "cc_import(name='a', static_library='a.a')");
    assertThat(getApi("//a:a")).isNull();
  }

  @Test
  public void testDisableInCcProtoLibrary() throws Exception {
    if (!analysisMock.isThisBazel()) {
      // Our internal version does not have this rule
      return;
    }

    mockToolsConfig.create("protobuf_workspace/WORKSPACE");
    mockToolsConfig.overwrite(
        "protobuf_workspace/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "package(default_visibility=['//visibility:public'])",
        "exports_files(['protoc'])",
        "proto_lang_toolchain(",
        "    name = 'cc_toolchain',",
        "    command_line = '--cpp_out=$(OUT)',",
        "    blacklisted_protos = [],",
        "    progress_message = 'Generating C++ proto_library %{label}',",
        ")");

    String existingWorkspace =
        new String(FileSystemUtils.readContentAsLatin1(rootDirectory.getRelative("WORKSPACE")));
    mockToolsConfig.overwrite(
        "WORKSPACE",
        "local_repository(name = 'com_google_protobuf', path = 'protobuf_workspace/')",
        existingWorkspace);
    invalidatePackages(); // A dash of magic to re-evaluate the WORKSPACE file.

    useConfiguration("--incompatible_disable_legacy_cc_provider");
    scratch.file(
        "a/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "cc_proto_library(name='a', deps=[':p'])",
        "proto_library(name='p', srcs=['p.proto'])");
    assertThat(getApi("//a:a")).isNull();
  }

  @Test
  public void testTransitiveHeaders() throws Exception {
    setBuildLanguageOptions(
        "--experimental_builtins_injection_override=+cc_binary",
        "--experimental_builtins_injection_override=+cc_library");
    useConfiguration("--noincompatible_disable_legacy_cc_provider");
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    srcs = ['lib.cc', 'lib.h'],",
        ")");
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                getApiForBuiltin("//pkg:check").getTransitiveHeaders()))
        .containsAtLeast("lib.h", "bin.h");
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                getApiForBuiltin("//pkg:check_lib").getTransitiveHeaders()))
        .contains("lib.h");
  }

  @Test
  public void testLinkFlags() throws Exception {
    setBuildLanguageOptions(
        "--experimental_builtins_injection_override=+cc_binary",
        "--experimental_builtins_injection_override=+cc_library");
    useConfiguration("--noincompatible_disable_legacy_cc_provider");
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    linkopts = ['-lm'],",
        "    deps = [':dependent_lib'],",
        ")",
        "cc_binary(",
        "    name = 'check_no_srcs',",
        "    linkopts = ['-lm'],",
        "    deps = [':dependent_lib'],",
        ")",
        "cc_library(",
        "    name = 'dependent_lib',",
        "    linkopts = ['-lz'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    defines = ['foo'],",
        "    linkopts = ['-Wl,-M'],",
        ")");
    assertThat(getApiForBuiltin("//pkg:check_lib").getLinkopts()).contains("-Wl,-M");
    assertThat(getApiForBuiltin("//pkg:dependent_lib").getLinkopts())
        .containsAtLeast("-lz", "-Wl,-M")
        .inOrder();
    assertThat(getApiForBuiltin("//pkg:check").getLinkopts()).isEmpty();
    assertThat(getApiForBuiltin("//pkg:check_no_srcs").getLinkopts()).isEmpty();
  }

  @Test
  public void testLibraries() throws Exception {
    setBuildLanguageOptions(
        "--experimental_builtins_injection_override=+cc_binary",
        "--experimental_builtins_injection_override=+cc_library");
    useConfiguration("--noincompatible_disable_legacy_cc_provider");
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_binary(",
        "    name = 'check_no_srcs',",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    srcs = ['lib.cc', 'lib.h'],",
        ")");
    assertThat(
            ActionsTestUtil.baseArtifactNames(getApiForBuiltin("//pkg:check_lib").getLibraries()))
        .containsExactly("libcheck_lib.a");
    assertThat(ActionsTestUtil.baseArtifactNames(getApiForBuiltin("//pkg:check").getLibraries()))
        .isEmpty();
    assertThat(
            ActionsTestUtil.baseArtifactNames(
                getApiForBuiltin("//pkg:check_no_srcs").getLibraries()))
        .isEmpty();
  }

  @Test
  public void testCcFlags() throws Exception {
    setBuildLanguageOptions("--experimental_builtins_injection_override=+cc_binary");
    useConfiguration("--noincompatible_disable_legacy_cc_provider");
    scratch.file(
        "pkg/BUILD",
        "cc_binary(",
        "    name = 'check',",
        "    srcs = ['bin.cc', 'bin.h'],",
        "    deps = [':check_lib'],",
        ")",
        "cc_library(",
        "    name = 'check_lib',",
        "    defines = ['foo'],",
        ")");
    assertThat(getApiForBuiltin("//pkg:check").getCcFlags()).contains("-Dfoo");
  }

  @Test
  public void testCcInfoTransitiveNativeLibsIsPrivateAPI() throws Exception {
    scratch.file(
        "pkg/custom_rule.bzl",
        "def _impl(ctx):",
        "  libs = ctx.attr.dep[CcInfo].transitive_native_libraries()",
        "  return []",
        "",
        "custom_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'dep' : attr.label(providers = [CcInfo])",
        "  }",
        ")");
    scratch.file(
        "pkg/BUILD",
        "load(':custom_rule.bzl', 'custom_rule')",
        "custom_rule(name = 'foo', dep = ':lib')",
        "cc_library(name = 'lib', defines = ['foo'])");
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//pkg:foo");

    assertContainsEvent("Rule in 'pkg' cannot use private API");
  }
}
