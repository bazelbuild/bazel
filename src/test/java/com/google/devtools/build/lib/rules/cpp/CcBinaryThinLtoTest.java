// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static java.util.Arrays.stream;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import java.io.IOException;
import java.util.stream.Stream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc_binary with ThinLTO. */
@RunWith(JUnit4.class)
public class CcBinaryThinLtoTest extends BuildViewTestCase {

  private String targetName = "bin";

  private ConfiguredTarget getCurrentTarget() throws Exception {
    return getConfiguredTarget("//pkg:" + targetName);
  }

  private SpawnAction getLinkAction() throws Exception {
    ConfiguredTarget pkg = getCurrentTarget();
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    SpawnAction linkAction = (SpawnAction) getGeneratingAction(pkgArtifact);
    assertThat(linkAction.getOutputs()).containsExactly(pkgArtifact);
    return linkAction;
  }

  private LtoBackendAction getBackendAction(String path) throws Exception {
    return (LtoBackendAction) getPredecessorByInputName(getLinkAction(), path);
  }

  private String getRootExecPath() throws Exception {
    ConfiguredTarget pkg = getCurrentTarget();
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    return pkgArtifact.getRoot().getExecPathString();
  }

  private SpawnAction getIndexAction(LtoBackendAction backendAction) throws Exception {
    return (SpawnAction)
        getPredecessorByInputName(
            backendAction, backendAction.getPrimaryOutput().getExecPathString() + ".thinlto.bc");
  }

  @Before
  public void createBasePkg() throws IOException {
    scratch.overwriteFile(
        "base/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        cc_library(
            name = "system_malloc",
            visibility = ["//visibility:public"],
        )

        cc_library(
            name = "empty_lib",
            visibility = ["//visibility:public"],
        )
        """);
  }

  public void createBuildFiles(String... extraCcBinaryParameters) throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_binary(name = '" + targetName + "',",
        "          srcs = ['binfile.cc', ],",
        "          deps = [ ':lib' ], ",
        String.join("", extraCcBinaryParameters),
        "          link_extra_lib = '//base:empty_lib', ",
        "          malloc = '//base:system_malloc')",
        "cc_library(name = 'lib',",
        "        srcs = ['libfile.cc'],",
        "        hdrs = ['libfile.h'],",
        "        linkstamp = 'linkstamp.cc',",
        "       )");

    scratch.file("pkg/binfile.cc", "#include \"pkg/libfile.h\"", "int main() { return pkg(); }");
    scratch.file("pkg/libfile.cc", "int pkg() { return 42; }");
    scratch.file("pkg/libfile.h", "int pkg();");
    scratch.file("pkg/linkstamp.cc");
  }

  public void createTestFiles(String extraTestParameters, String extraLibraryParameters)
      throws Exception {
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "load('@rules_cc//cc:cc_test.bzl', 'cc_test')",
        "cc_test(",
        "    name = 'bin_test',",
        "    srcs = ['bin_test.cc', ],",
        "    deps = [ ':lib' ], ",
        extraTestParameters,
        "    link_extra_lib = '//base:empty_lib', ",
        "    malloc = '//base:system_malloc'",
        ")",
        "cc_test(",
        "    name = 'bin_test2',",
        "    srcs = ['bin_test2.cc', ],",
        "    deps = [ ':lib' ], ",
        extraTestParameters,
        "    link_extra_lib = '//base:empty_lib', ",
        "    malloc = '//base:system_malloc'",
        ")",
        "cc_library(",
        "    name = 'lib',",
        "    srcs = ['libfile.cc'],",
        "    hdrs = ['libfile.h'],",
        extraLibraryParameters,
        "    linkstamp = 'linkstamp.cc',",
        ")");

    scratch.file("pkg/bin_test.cc", "#include \"pkg/libfile.h\"", "int main() { return pkg(); }");
    scratch.file("pkg/bin_test2.cc", "#include \"pkg/libfile.h\"", "int main() { return pkg(); }");
    scratch.file("pkg/libfile.cc", "int pkg() { return 42; }");
    scratch.file("pkg/libfile.h", "int pkg();");
    scratch.file("pkg/linkstamp.cc");
  }

  private Action getPredecessorByInputName(Action action, String str) {
    for (Artifact a : action.getInputs().toList()) {
      if (a.getExecPathString().contains(str)) {
        return getGeneratingAction(a);
      }
    }
    return null;
  }

  @Test
  public void testNoUseLtoIndexingBitcodeFile() throws Exception {
    createBuildFiles();

    setupThinLTOCrosstool(
        CppRuleClasses.NO_USE_LTO_INDEXING_BITCODE_FILE, CppRuleClasses.SUPPORTS_PIC);
    useConfiguration("--features=no_use_lto_indexing_bitcode_file");
    String rootExecPath = getRootExecPath();

    /*
    We follow the chain from the final product backwards.

    binary <=[Link]=
    .lto/...o <=[LTOBackend]=
    {.o.thinlto.bc,.o.imports} <=[LTOIndexing]=
    .o <= [CppCompile] .cc
    */
    SpawnAction indexAction =
        getIndexAction(
            getBackendAction("pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.pic.o"));

    assertThat(indexAction.getArguments()).doesNotContain("object_suffix_replace");

    assertThat(artifactsToStrings(indexAction.getInputs()))
        .containsAtLeast("bin pkg/_objs/bin/binfile.pic.o", "bin pkg/_objs/lib/libfile.pic.o");

    CppCompileAction bitcodeAction =
        (CppCompileAction) getPredecessorByInputName(indexAction, "pkg/_objs/bin/binfile.pic.o");
    assertThat(bitcodeAction.getArguments()).doesNotContain("lto_indexing_bitcode=");
  }

  private void setupThinLTOCrosstool(String... extraFeatures) throws Exception {
    String[] allFeatures =
        Stream.concat(
                Stream.of(
                    CppRuleClasses.THIN_LTO,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES),
                stream(extraFeatures))
            .toArray(String[]::new);
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(allFeatures));
  }

  private void testLLVMCachePrefetchBackendOption(String extraOption) throws Exception {
    createBuildFiles();
    scratch.file(
        "fdo/BUILD",
        "load('@rules_cc//cc/toolchains:fdo_prefetch_hints.bzl',"
            + " 'fdo_prefetch_hints')",
        "fdo_prefetch_hints(name='test_profile', profile=':prefetch.afdo')");

    setupThinLTOCrosstool(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.AUTOFDO);
    useConfiguration(
        "--fdo_prefetch_hints=//fdo:test_profile", "--compilation_mode=opt", extraOption);

    String rootExecPath = getRootExecPath();
    LtoBackendAction backendAction =
        getBackendAction("pkg/bin.lto/" + rootExecPath + "/pkg/_objs/bin/binfile.o");

    assertThat(Joiner.on(" ").join(backendAction.getArguments()))
        .containsMatch("-mllvm -prefetch-hints-file=.*/prefetch.afdo");

    assertThat(ActionsTestUtil.baseArtifactNames(backendAction.getInputs()))
        .contains("prefetch.afdo");
  }

  @Test
  public void testFdoCachePrefetchLLVMOptionsToBackendFromLabel() throws Exception {
    testLLVMCachePrefetchBackendOption("");
  }

  @Test
  public void testFdoCachePrefetchAndFdoLLVMOptionsToBackendFromLabel() throws Exception {
    testLLVMCachePrefetchBackendOption("--fdo_optimize=/profile.zip");
  }

  @Test
  public void testThinLtoWithoutSupportsStartEndLibError() throws Exception {
    createBuildFiles("testonly = 1,");
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.THIN_LTO));
    checkError("//pkg:bin", "The feature supports_start_end_lib must be enabled.");
  }
}
