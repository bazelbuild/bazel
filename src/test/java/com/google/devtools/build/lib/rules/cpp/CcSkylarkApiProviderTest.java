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

/**
 * Tests for Skylark providers for cpp rules.
 */
@RunWith(JUnit4.class)
public class CcSkylarkApiProviderTest extends BuildViewTestCase {
  private CcSkylarkApiProvider getApi(String label) throws Exception {
    RuleConfiguredTarget rule = (RuleConfiguredTarget) getConfiguredTarget(label);
    return (CcSkylarkApiProvider) rule.get(CcSkylarkApiProvider.NAME);
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
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check").getTransitiveHeaders()))
        .containsAtLeast("lib.h", "bin.h");
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check_lib").getTransitiveHeaders()))
        .contains("lib.h");
  }

  @Test
  public void testLinkFlags() throws Exception {
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
    assertThat(getApi("//pkg:check_lib").getLinkopts())
        .contains("-Wl,-M");
    assertThat(getApi("//pkg:dependent_lib").getLinkopts())
        .containsAtLeast("-lz", "-Wl,-M")
        .inOrder();
    assertThat(getApi("//pkg:check").getLinkopts())
        .isEmpty();
    assertThat(getApi("//pkg:check_no_srcs").getLinkopts())
        .isEmpty();
  }

  @Test
  public void testLibraries() throws Exception {
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
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check_lib").getLibraries()))
        .containsExactly("libcheck_lib.a");
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check").getLibraries()))
        .isEmpty();
    assertThat(ActionsTestUtil.baseArtifactNames(getApi("//pkg:check_no_srcs").getLibraries()))
        .isEmpty();
  }

  @Test
  public void testCcFlags() throws Exception {
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
    assertThat(getApi("//pkg:check").getCcFlags()).contains("-Dfoo");
  }
}
