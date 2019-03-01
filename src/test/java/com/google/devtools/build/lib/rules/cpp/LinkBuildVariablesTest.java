// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import java.io.IOException;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@code CppLinkAction} is populated with the correct build variables. */
@RunWith(JUnit4.class)
public class LinkBuildVariablesTest extends LinkBuildVariablesTestCase {

  @Before
  public void createFooFooCcLibraryForRuleContext() throws IOException {
    scratch.file("foo/BUILD", "cc_library(name = 'foo')");
  }

  private RuleContext getRuleContext() throws Exception {
    return getRuleContext(getConfiguredTarget("//foo:foo"));
  }

  @Test
  public void testIsUsingFissionIsIdenticalForCompileAndLink() {
    assertThat(LinkBuildVariables.IS_USING_FISSION.getVariableName())
        .isEqualTo(CompileBuildVariables.IS_USING_FISSION.getVariableName());
  }

  @Test
  public void testForcePicBuildVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.SUPPORTS_PIC_FEATURE);
    useConfiguration("--force_pic");
    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    CcToolchainVariables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    String variableValue =
        getVariableValue(
            getRuleContext(), variables, LinkBuildVariables.FORCE_PIC.getVariableName());
    assertThat(variableValue).contains("");
  }

  @Test
  public void testLibrariesToLinkAreExported() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.SUPPORTS_DYNAMIC_LINKER_FEATURE);
    useConfiguration();

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CcToolchainVariables variables =
        getLinkBuildVariables(target, LinkTargetType.NODEPS_DYNAMIC_LIBRARY);
    VariableValue librariesToLinkSequence =
        variables.getVariable(LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName());
    assertThat(librariesToLinkSequence).isNotNull();
    Iterable<? extends VariableValue> librariesToLink =
        librariesToLinkSequence.getSequenceValue(
            LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName());
    assertThat(librariesToLink).hasSize(1);
    VariableValue nameValue =
        librariesToLink
            .iterator()
            .next()
            .getFieldValue(
                LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(),
                LibraryToLinkValue.NAME_FIELD_NAME);
    assertThat(nameValue).isNotNull();
    String name = nameValue.getStringValue(LibraryToLinkValue.NAME_FIELD_NAME);
    assertThat(name).matches(".*a\\..*o");
  }

  @Test
  public void testLibrarySearchDirectoriesAreExported() throws Exception {
    AnalysisMock.get().ccSupport().setupCrosstool(mockToolsConfig);
    useConfiguration();

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['some-dir/bar.so'])");
    scratch.file("x/some-dir/bar.so");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    CcToolchainVariables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getSequenceVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.LIBRARY_SEARCH_DIRECTORIES.getVariableName());
    assertThat(Iterables.getOnlyElement(variableValue)).contains("some-dir");
  }

  @Test
  public void testLinkerParamFileIsExported() throws Exception {
    AnalysisMock.get().ccSupport().setupCrosstool(mockToolsConfig);
    useConfiguration();

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['some-dir/bar.so'])");
    scratch.file("x/some-dir/bar.so");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    CcToolchainVariables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    String variableValue =
        getVariableValue(
            getRuleContext(), variables, LinkBuildVariables.LINKER_PARAM_FILE.getVariableName());
    assertThat(variableValue).matches(".*bin/x/bin" + "-2.params$");
  }

  @Test
  public void testInterfaceLibraryBuildingVariablesWhenLegacyGenerationPossible() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE,
            MockCcSupport.SUPPORTS_DYNAMIC_LINKER_FEATURE);
    useConfiguration();

    verifyIfsoVariables();
  }

  @Test
  public void testInterfaceLibraryBuildingVariablesWhenGenerationPossible() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.SUPPORTS_DYNAMIC_LINKER_FEATURE,
            MockCcSupport.SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE);
    useConfiguration();

    verifyIfsoVariables();
  }

  private void verifyIfsoVariables() throws Exception {
    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CcToolchainVariables variables =
        getLinkBuildVariables(target, LinkTargetType.NODEPS_DYNAMIC_LIBRARY);

    String interfaceLibraryBuilder =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_BUILDER.getVariableName());
    String interfaceLibraryInput =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_INPUT.getVariableName());
    String interfaceLibraryOutput =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_OUTPUT.getVariableName());
    String generateInterfaceLibrary =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.GENERATE_INTERFACE_LIBRARY.getVariableName());

    assertThat(generateInterfaceLibrary).isEqualTo("yes");
    assertThat(interfaceLibraryInput).endsWith("libfoo.so");
    assertThat(interfaceLibraryOutput).endsWith("libfoo.ifso");
    assertThat(interfaceLibraryBuilder).endsWith("build_interface_so");
  }

  @Test
  public void testNoIfsoBuildingWhenWhenThinLtoIndexing() throws Exception {
    // Make sure the interface shared object generation is enabled in the configuration
    // (which it is not by default for some windows toolchains)
    invalidatePackages(true);
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.THIN_LTO_CONFIGURATION,
            MockCcSupport.SUPPORTS_PIC_FEATURE,
            MockCcSupport.HOST_AND_NONHOST_CONFIGURATION,
            MockCcSupport.SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE,
            MockCcSupport.SUPPORTS_DYNAMIC_LINKER_FEATURE,
            MockCcSupport.SUPPORTS_START_END_LIB_FEATURE);
    useConfiguration("--features=thin_lto");

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CppLinkAction linkAction = getCppLinkAction(target, LinkTargetType.NODEPS_DYNAMIC_LIBRARY);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(linkAction, "x/libfoo.so.lto/x/_objs/foo/a.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    CppLinkAction indexAction =
        (CppLinkAction)
            getPredecessorByInputName(
                backendAction, "x/libfoo.so.lto/x/_objs/foo/a.pic.o.thinlto.bc");
    CcToolchainVariables variables = indexAction.getLinkCommandLine().getBuildVariables();

    String interfaceLibraryBuilder =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_BUILDER.getVariableName());
    String interfaceLibraryInput =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_INPUT.getVariableName());
    String interfaceLibraryOutput =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_OUTPUT.getVariableName());
    String generateInterfaceLibrary =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.GENERATE_INTERFACE_LIBRARY.getVariableName());

    assertThat(generateInterfaceLibrary).isEqualTo("no");
    assertThat(interfaceLibraryInput).endsWith("ignored");
    assertThat(interfaceLibraryOutput).endsWith("ignored");
    assertThat(interfaceLibraryBuilder).endsWith("ignored");
  }

  @Test
  public void testInterfaceLibraryBuildingVariablesWhenGenerationNotAllowed() throws Exception {
    // Make sure the interface shared object generation is enabled in the configuration
    // (which it is not by default for some windows toolchains)
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE);
    useConfiguration();

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CcToolchainVariables variables = getLinkBuildVariables(target, LinkTargetType.STATIC_LIBRARY);

    String interfaceLibraryBuilder =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_BUILDER.getVariableName());
    String interfaceLibraryInput =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_INPUT.getVariableName());
    String interfaceLibraryOutput =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.INTERFACE_LIBRARY_OUTPUT.getVariableName());
    String generateInterfaceLibrary =
        getVariableValue(
            getRuleContext(),
            variables,
            LinkBuildVariables.GENERATE_INTERFACE_LIBRARY.getVariableName());

    assertThat(generateInterfaceLibrary).isEqualTo("no");
    assertThat(interfaceLibraryInput).endsWith("ignored");
    assertThat(interfaceLibraryOutput).endsWith("ignored");
    assertThat(interfaceLibraryBuilder).endsWith("ignored");
  }

  @Test
  public void testOutputExecpath() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.SUPPORTS_DYNAMIC_LINKER_FEATURE);
    // Make sure the interface shared object generation is enabled in the configuration
    // (which it is not by default for some windows toolchains)
    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CcToolchainVariables variables =
        getLinkBuildVariables(target, LinkTargetType.NODEPS_DYNAMIC_LIBRARY);

    assertThat(
            getVariableValue(
                getRuleContext(), variables, LinkBuildVariables.OUTPUT_EXECPATH.getVariableName()))
        .endsWith("x/libfoo.so");
  }

  @Test
  public void testOutputExecpathIsNotExposedWhenThinLtoIndexing() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.THIN_LTO_CONFIGURATION,
            MockCcSupport.HOST_AND_NONHOST_CONFIGURATION,
            MockCcSupport.SUPPORTS_DYNAMIC_LINKER_FEATURE,
            MockCcSupport.SUPPORTS_PIC_FEATURE,
            MockCcSupport.SUPPORTS_INTERFACE_SHARED_LIBRARIES_FEATURE,
            MockCcSupport.SUPPORTS_START_END_LIB_FEATURE);
    useConfiguration("--features=thin_lto");

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CppLinkAction linkAction = getCppLinkAction(target, LinkTargetType.NODEPS_DYNAMIC_LIBRARY);

    LtoBackendAction backendAction =
        (LtoBackendAction)
            getPredecessorByInputName(linkAction, "x/libfoo.so.lto/x/_objs/foo/a.pic.o");
    assertThat(backendAction.getMnemonic()).isEqualTo("CcLtoBackendCompile");

    CppLinkAction indexAction =
        (CppLinkAction)
            getPredecessorByInputName(
                backendAction, "x/libfoo.so.lto/x/_objs/foo/a.pic.o.thinlto.bc");
    CcToolchainVariables variables = indexAction.getLinkCommandLine().getBuildVariables();

    assertThat(variables.isAvailable(LinkBuildVariables.OUTPUT_EXECPATH.getVariableName()))
        .isFalse();
  }

  @Test
  public void testIsCcTestLinkActionBuildVariable() throws Exception {
    scratch.file("x/BUILD",
        "cc_test(name = 'foo_test', srcs = ['a.cc'])",
        "cc_binary(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget testTarget = getConfiguredTarget("//x:foo_test");
    CcToolchainVariables testVariables =
        getLinkBuildVariables(testTarget, LinkTargetType.EXECUTABLE);

    assertThat(
            testVariables.getVariable(LinkBuildVariables.IS_CC_TEST.getVariableName()).isTruthy())
        .isTrue();

    ConfiguredTarget binaryTarget = getConfiguredTarget("//x:foo");
    CcToolchainVariables binaryVariables =
        getLinkBuildVariables(binaryTarget, LinkTargetType.EXECUTABLE);

    assertThat(
            binaryVariables.getVariable(LinkBuildVariables.IS_CC_TEST.getVariableName()).isTruthy())
        .isFalse();
  }

  @Test
  public void testStripBinariesIsEnabledWhenStripModeIsAlwaysNoMatterWhat() throws Exception {
    scratch.file("x/BUILD", "cc_binary(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    assertStripBinaryVariableIsPresent("always", "opt", true);
    assertStripBinaryVariableIsPresent("always", "fastbuild", true);
    assertStripBinaryVariableIsPresent("always", "dbg", true);
    assertStripBinaryVariableIsPresent("sometimes", "opt", false);
    assertStripBinaryVariableIsPresent("sometimes", "fastbuild", true);
    assertStripBinaryVariableIsPresent("sometimes", "dbg", false);
    assertStripBinaryVariableIsPresent("never", "opt", false);
    assertStripBinaryVariableIsPresent("never", "fastbuild", false);
    assertStripBinaryVariableIsPresent("never", "dbg", false);
  }

  private void assertStripBinaryVariableIsPresent(
      String stripMode, String compilationMode, boolean isEnabled) throws Exception {
    useConfiguration("--strip=" + stripMode, "--compilation_mode=" + compilationMode);
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CcToolchainVariables variables = getLinkBuildVariables(target, LinkTargetType.EXECUTABLE);
    assertThat(variables.isAvailable(LinkBuildVariables.STRIP_DEBUG_SYMBOLS.getVariableName()))
        .isEqualTo(isEnabled);
  }

  @Test
  public void testIsUsingFissionVariableUsingLegacyFields() throws Exception {
    scratch.file("x/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    scratch.file("x/foo.cc");

    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PER_OBJECT_DEBUG_INFO_CONFIGURATION);

    useConfiguration("--fission=no");
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CcToolchainVariables variables = getLinkBuildVariables(target, LinkTargetType.EXECUTABLE);
    assertThat(variables.isAvailable(LinkBuildVariables.IS_USING_FISSION.getVariableName()))
        .isFalse();

    useConfiguration("--fission=yes");
    ConfiguredTarget fissionTarget = getConfiguredTarget("//x:foo");
    CcToolchainVariables fissionVariables =
        getLinkBuildVariables(fissionTarget, LinkTargetType.EXECUTABLE);
    assertThat(fissionVariables.isAvailable(LinkBuildVariables.IS_USING_FISSION.getVariableName()))
        .isTrue();
  }

  @Test
  public void testIsUsingFissionVariable() throws Exception {
    scratch.file("x/BUILD", "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    scratch.file("x/foo.cc");

    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, MockCcSupport.PER_OBJECT_DEBUG_INFO_CONFIGURATION);

    useConfiguration("--fission=no");
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    CcToolchainVariables variables = getLinkBuildVariables(target, LinkTargetType.EXECUTABLE);
    assertThat(variables.isAvailable(LinkBuildVariables.IS_USING_FISSION.getVariableName()))
        .isFalse();

    useConfiguration("--fission=yes");
    ConfiguredTarget fissionTarget = getConfiguredTarget("//x:foo");
    CcToolchainVariables fissionVariables =
        getLinkBuildVariables(fissionTarget, LinkTargetType.EXECUTABLE);
    assertThat(fissionVariables.isAvailable(LinkBuildVariables.IS_USING_FISSION.getVariableName()))
        .isTrue();
  }

  @Test
  public void testSysrootVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "builtin_sysroot: '/usr/local/custom-sysroot'");
    useConfiguration();

    scratch.file("x/BUILD", "cc_binary(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget testTarget = getConfiguredTarget("//x:foo");
    CcToolchainVariables testVariables =
        getLinkBuildVariables(testTarget, LinkTargetType.EXECUTABLE);

    assertThat(testVariables.isAvailable(CcCommon.SYSROOT_VARIABLE_NAME)).isTrue();
  }

  private Action getPredecessorByInputName(Action action, String str) {
    for (Artifact a : action.getInputs()) {
      if (a.getExecPathString().contains(str)) {
        return getGeneratingAction(a);
      }
    }
    return null;
  }

  @Test
  public void testUserLinkFlagsWithLinkoptOption() throws Exception {
    useConfiguration("--linkopt=-bar");

    scratch.file("x/BUILD", "cc_binary(name = 'foo', srcs = ['a.cc'], linkopts = ['-foo'])");
    scratch.file("x/a.cc");

    ConfiguredTarget testTarget = getConfiguredTarget("//x:foo");
    CcToolchainVariables testVariables =
        getLinkBuildVariables(testTarget, LinkTargetType.EXECUTABLE);

    ImmutableList<String> userLinkFlags =
        CcToolchainVariables.toStringList(
            testVariables, LinkBuildVariables.USER_LINK_FLAGS.getVariableName());
    assertThat(userLinkFlags).containsAllOf("-foo", "-bar").inOrder();

    ImmutableList<String> legacyLinkFlags =
        CcToolchainVariables.toStringList(
            testVariables, LinkBuildVariables.LEGACY_LINK_FLAGS.getVariableName());
    assertThat(legacyLinkFlags).doesNotContain("-foo");
    assertThat(legacyLinkFlags).doesNotContain("-bar");
  }
}
