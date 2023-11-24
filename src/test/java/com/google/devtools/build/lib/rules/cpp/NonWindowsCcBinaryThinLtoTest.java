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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for cc_binary with ThinLTO.
 *
 * <p>As of 2020-02-06, these tests do not work on Windows, hence the "NonWindows" part of the class
 * name.
 */
@RunWith(JUnit4.class)
public final class NonWindowsCcBinaryThinLtoTest extends BuildViewTestCase {

  public void createTestFiles(String extraTestParameters, String extraLibraryParameters)
      throws Exception {
    scratch.overwriteFile(
        "base/BUILD", "cc_library(name = 'system_malloc', visibility = ['//visibility:public'])");
    scratch.file(
        "pkg/BUILD",
        "package(features = ['thin_lto'])",
        "cc_test(",
        "    name = 'bin_test',",
        "    srcs = ['bin_test.cc', ],",
        "    deps = [ ':lib' ], ",
        extraTestParameters,
        "    malloc = '//base:system_malloc'",
        ")",
        "cc_test(",
        "    name = 'bin_test2',",
        "    srcs = ['bin_test2.cc', ],",
        "    deps = [ ':lib' ], ",
        extraTestParameters,
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

  /** Helper method to get the root prefix from the given dwpFile. */
  private static PathFragment dwpRootPrefix(Artifact dwpFile) throws Exception {
    return dwpFile
        .getExecPath()
        .subFragment(
            0, dwpFile.getExecPath().segmentCount() - dwpFile.getRootRelativePath().segmentCount());
  }

  /** Helper method that checks that a .dwp has the expected generating action structure. */
  private void validateDwp(
      RuleContext ruleContext,
      Artifact dwpFile,
      CcToolchainProvider toolchain,
      List<String> expectedInputs)
      throws Exception {
    SpawnAction dwpAction = (SpawnAction) getGeneratingAction(dwpFile);
    String dwpToolPath = toolchain.getToolPathFragment(Tool.DWP, ruleContext).getPathString();
    assertThat(dwpAction.getMnemonic()).isEqualTo("CcGenerateDwp");
    assertThat(dwpToolPath).isEqualTo(dwpAction.getCommandFilename());
    List<String> commandArgs = dwpAction.getArguments();
    // The first argument should be the command being executed.
    assertThat(dwpToolPath).isEqualTo(commandArgs.get(0));
    // The final two arguments should be "-o dwpOutputFile".
    assertThat(commandArgs.subList(commandArgs.size() - 2, commandArgs.size()))
        .containsExactly("-o", dwpFile.getExecPathString())
        .inOrder();
    // The remaining arguments should be the set of .dwo inputs (in any order).
    assertThat(commandArgs.subList(1, commandArgs.size() - 2))
        .containsExactlyElementsIn(expectedInputs);
  }

  @Test
  public void testLinkstaticCcLibraryOnTestFission() throws Exception {
    createTestFiles("", "linkstatic = 1,");

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.THIN_LTO,
                    CppRuleClasses.SUPPORTS_PIC,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    CppRuleClasses.THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES,
                    CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration(
        "--fission=yes", "--features=thin_lto_linkstatic_tests_use_shared_nonlto_backends");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin_test");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath = pkgArtifact.getRoot().getExecPathString();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);

    // The cc_test source should still get LTO in this case
    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "bin_test.lto/" + rootExecPath + "/pkg/_objs/bin_test/bin_test.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(artifactsToStrings(backendAction.getOutputs()))
        .containsExactly(
            "bin pkg/bin_test.lto/" + rootExecPath + "/pkg/_objs/bin_test/bin_test.pic.o",
            "bin pkg/bin_test.lto/" + rootExecPath + "/pkg/_objs/bin_test/bin_test.pic.dwo");

    assertThat(backendAction.getArguments()).contains("per_object_debug_info_option");

    // The linkstatic cc_library source should get shared non-LTO
    backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "shared.nonlto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments()).contains("-fPIC");
    assertThat(artifactsToStrings(backendAction.getOutputs()))
        .containsExactly(
            "bin shared.nonlto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.o",
            "bin shared.nonlto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.dwo");

    assertThat(backendAction.getArguments()).contains("per_object_debug_info_option");

    // Now check the dwp action.
    Artifact dwpFile = getFileConfiguredTarget(pkg.getLabel() + ".dwp").getArtifact();
    PathFragment rootPrefix = dwpRootPrefix(dwpFile);
    RuleContext ruleContext = getRuleContext(pkg);
    CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
    validateDwp(
        ruleContext,
        dwpFile,
        toolchain,
        ImmutableList.of(
            rootPrefix + "/shared.nonlto/" + rootExecPath + "/pkg/_objs/lib/libfile.pic.dwo",
            rootPrefix
                + "/pkg/bin_test.lto/"
                + rootExecPath
                + "/pkg/_objs/bin_test/bin_test.pic.dwo"));
  }

  @Test
  public void testLinkstaticCcLibraryOnTest() throws Exception {
    createTestFiles("", "linkstatic = 1,");

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.THIN_LTO,
                    CppRuleClasses.SUPPORTS_START_END_LIB,
                    CppRuleClasses.THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS,
                    MockCcSupport.HOST_AND_NONHOST_CONFIGURATION_FEATURES,
                    CppRuleClasses.SUPPORTS_PIC,
                    CppRuleClasses.PER_OBJECT_DEBUG_INFO));
    useConfiguration("--features=thin_lto_linkstatic_tests_use_shared_nonlto_backends");

    ConfiguredTarget pkg = getConfiguredTarget("//pkg:bin_test");
    Artifact pkgArtifact = getFilesToBuild(pkg).getSingleton();
    String rootExecPath1 = pkgArtifact.getRoot().getExecPathString();
    CppLinkAction linkAction = (CppLinkAction) getGeneratingAction(pkgArtifact);

    ConfiguredTarget pkg2 = getConfiguredTarget("//pkg:bin_test2");
    Artifact pkgArtifact2 = getFilesToBuild(pkg2).getSingleton();
    String rootExecPath2 = pkgArtifact2.getRoot().getExecPathString();
    CppLinkAction linkAction2 = (CppLinkAction) getGeneratingAction(pkgArtifact2);

    // The cc_test source should still get LTO in this case
    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "bin_test.lto/" + rootExecPath1 + "/pkg/_objs/bin_test/bin_test.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    // The linkstatic cc_library sources should get shared non-LTO

    backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction, "shared.nonlto/" + rootExecPath1 + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");
    assertThat(backendAction.getArguments()).contains("-fPIC");

    LtoBackendAction backendAction2 =
        (LtoBackendAction)
            getPredecessorByInputName(
                linkAction2, "shared.nonlto/" + rootExecPath2 + "/pkg/_objs/lib/libfile.pic.o");
    assertThat(backendAction2.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    assertThat(backendAction).isEqualTo(backendAction2);
  }

  private Action getPredecessorByInputName(Action action, String str) {
    for (Artifact a : action.getInputs().toList()) {
      if (a.getExecPathString().contains(str)) {
        return getGeneratingAction(a);
      }
    }
    return null;
  }
}
