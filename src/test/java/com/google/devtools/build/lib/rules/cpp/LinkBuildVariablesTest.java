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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.util.OsUtils;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@code CppLinkAction} is populated with the correct build variables. */
@RunWith(JUnit4.class)
public class LinkBuildVariablesTest extends BuildViewTestCase {

  private CppLinkAction getCppLinkAction(ConfiguredTarget target, Link.LinkTargetType type) {
    Artifact linkerOutput = null;
    switch (type) {
      case STATIC_LIBRARY:
      case ALWAYS_LINK_STATIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + ".a", target);
        break;
      case PIC_STATIC_LIBRARY:
      case ALWAYS_LINK_PIC_STATIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + "pic.a", target);
        break;
      case DYNAMIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + ".so", target);
        break;
      case EXECUTABLE:
        linkerOutput = getExecutable(target);
        break;
      default:
        throw new IllegalArgumentException(
            String.format("Cannot get CppLinkAction for link type %s", type));
    }
    return (CppLinkAction) getGeneratingAction(linkerOutput);
  }

  /** Returns active build variables for a link action of given type for given target. */
  protected Variables getLinkBuildVariables(ConfiguredTarget target, Link.LinkTargetType type) {
    return getCppLinkAction(target, type).getLinkCommandLine().getBuildVariables();
  }

  /** Returns the value of a given variable in context of the given Variables instance. */
  protected List<String> getVariableValue(Variables variables, String variable) throws Exception {
    FeatureConfiguration mockFeatureConfiguration =
        CcToolchainFeaturesTest.buildFeatures(
                "feature {",
                "   name: 'a'",
                "   flag_set {",
                "   action: 'foo'",
                "      flag_group {",
                "         flag: '%{" + variable + "}'",
                "      }",
                "   }",
                "}")
            .getFeatureConfiguration("a");
    return mockFeatureConfiguration.getCommandLine("foo", variables);
  }

  @Test
  public void testLinkstampBuildVariable() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(",
        "   name = 'bin',",
        "   srcs = ['a.cc'],",
        "   deps = [':lib'],",
        ")",
        "cc_library(",
        "   name = 'lib',",
        "   srcs = ['b.cc'],",
        "   linkstamp = 'c.cc',",
        ")");
    scratch.file("x/a.cc");
    scratch.file("x/b.cc");
    scratch.file("x/c.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    Variables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getVariableValue(variables, CppLinkActionBuilder.LINKSTAMP_PATHS_VARIABLE);
    assertThat(Iterables.getOnlyElement(variableValue)).contains("c.o");
  }

  @Test
  public void testForcePicBuildVariable() throws Exception {
    useConfiguration("--force_pic");
    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    Variables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getVariableValue(variables, CppLinkActionBuilder.FORCE_PIC_VARIABLE);
    assertThat(variableValue).contains("");
  }

  @Test
  public void testLibrariesToLinkAreExported() throws Exception {
    AnalysisMock.get().ccSupport().setupCrosstool(mockToolsConfig);
    useConfiguration();

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Variables variables = getLinkBuildVariables(target, LinkTargetType.DYNAMIC_LIBRARY);
    VariableValue librariesToLinkSequence =
        variables.getVariable(CppLinkActionBuilder.LIBRARIES_TO_LINK_VARIABLE);
    assertThat(librariesToLinkSequence).isNotNull();
    Iterable<? extends VariableValue> librariesToLink =
        librariesToLinkSequence.getSequenceValue(CppLinkActionBuilder.LIBRARIES_TO_LINK_VARIABLE);
    assertThat(librariesToLink).hasSize(1);
    VariableValue nameValue =
        librariesToLink
            .iterator()
            .next()
            .getFieldValue(
                CppLinkActionBuilder.LIBRARIES_TO_LINK_VARIABLE,
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
    Variables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getVariableValue(variables, CppLinkActionBuilder.LIBRARY_SEARCH_DIRECTORIES_VARIABLE);
    assertThat(Iterables.getOnlyElement(variableValue)).contains("some-dir");
  }

  @Test
  public void testLinkerParamFileIsExported() throws Exception {
    AnalysisMock.get().ccSupport().setupCrosstool(mockToolsConfig);
    useConfiguration();

    scratch.file("x/BUILD", "cc_binary(name = 'bin', srcs = ['some-dir/bar.so'])");
    scratch.file("x/some-dir/bar.so");

    ConfiguredTarget target = getConfiguredTarget("//x:bin");
    Variables variables = getLinkBuildVariables(target, Link.LinkTargetType.EXECUTABLE);
    List<String> variableValue =
        getVariableValue(variables, CppLinkActionBuilder.LINKER_PARAM_FILE_VARIABLE);
    assertThat(Iterables.getOnlyElement(variableValue)).matches(".*bin/x/bin"
        + OsUtils.executableExtension() + "-2.params$");
  }

  @Test
  public void testInterfaceLibraryBuildingVariablesWhenGenerationPossible() throws Exception {
    // Make sure the interface shared object generation is enabled in the configuration
    // (which it is not by default for some windows toolchains)
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "supports_interface_shared_objects: true");
    useConfiguration();

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Variables variables = getLinkBuildVariables(target, LinkTargetType.DYNAMIC_LIBRARY);

    String interfaceLibraryBuilder =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.INTERFACE_LIBRARY_BUILDER_VARIABLE));
    String interfaceLibraryInput =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.INTERFACE_LIBRARY_INPUT_VARIABLE));
    String interfaceLibraryOutput =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.INTERFACE_LIBRARY_OUTPUT_VARIABLE));
    String generateInterfaceLibrary =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.GENERATE_INTERFACE_LIBRARY_VARIABLE));

    assertThat(generateInterfaceLibrary).isEqualTo("yes");
    assertThat(interfaceLibraryInput).endsWith("libfoo.so");
    assertThat(interfaceLibraryOutput).endsWith("libfoo.ifso");
    assertThat(interfaceLibraryBuilder).endsWith("build_interface_so");
  }

  @Test
  public void testInterfaceLibraryBuildingVariablesWhenGenerationNotAllowed() throws Exception {
    // Make sure the interface shared object generation is enabled in the configuration
    // (which it is not by default for some windows toolchains)
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "supports_interface_shared_objects: true");
    useConfiguration();

    scratch.file("x/BUILD", "cc_library(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Variables variables = getLinkBuildVariables(target, LinkTargetType.STATIC_LIBRARY);

    String interfaceLibraryBuilder =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.INTERFACE_LIBRARY_BUILDER_VARIABLE));
    String interfaceLibraryInput =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.INTERFACE_LIBRARY_INPUT_VARIABLE));
    String interfaceLibraryOutput =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.INTERFACE_LIBRARY_OUTPUT_VARIABLE));
    String generateInterfaceLibrary =
        Iterables.getOnlyElement(
            getVariableValue(variables, CppLinkActionBuilder.GENERATE_INTERFACE_LIBRARY_VARIABLE));

    assertThat(generateInterfaceLibrary).isEqualTo("no");
    assertThat(interfaceLibraryInput).endsWith("ignored");
    assertThat(interfaceLibraryOutput).endsWith("ignored");
    assertThat(interfaceLibraryBuilder).endsWith("ignored");
  }

  @Test
  public void testIsCcTestLinkActionBuildVariable() throws Exception {
    scratch.file("x/BUILD",
        "cc_test(name = 'foo_test', srcs = ['a.cc'])",
        "cc_binary(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget testTarget = getConfiguredTarget("//x:foo_test");
    Variables testVariables = getLinkBuildVariables(testTarget, LinkTargetType.EXECUTABLE);

    assertThat(testVariables.isAvailable(CppLinkActionBuilder.IS_CC_TEST_LINK_ACTION_VARIABLE))
        .isTrue();

    ConfiguredTarget binaryTarget = getConfiguredTarget("//x:foo");
    Variables binaryVariables = getLinkBuildVariables(binaryTarget, LinkTargetType.EXECUTABLE);

    assertThat(binaryVariables.isAvailable(CppLinkActionBuilder.IS_CC_TEST_LINK_ACTION_VARIABLE))
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
    Variables variables = getLinkBuildVariables(target, LinkTargetType.EXECUTABLE);
    assertThat(variables.isAvailable(CppLinkActionBuilder.STRIP_DEBUG_SYMBOLS_VARIABLE))
        .isEqualTo(isEnabled);
  }

  @Test
  public void testIsUsingFissionVariable() throws Exception {
    scratch.file("x/BUILD",
        "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    scratch.file("x/foo.cc");

    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "supports_fission: true");

    useConfiguration("--fission=no");
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Variables variables = getLinkBuildVariables(target, LinkTargetType.EXECUTABLE);
    assertThat(variables.isAvailable(CppLinkActionBuilder.IS_USING_FISSION_VARIABLE)).isFalse();

    useConfiguration("--fission=yes");
    ConfiguredTarget fissionTarget = getConfiguredTarget("//x:foo");
    Variables fissionVariables = getLinkBuildVariables(fissionTarget, LinkTargetType.EXECUTABLE);
    assertThat(fissionVariables.isAvailable(CppLinkActionBuilder.IS_USING_FISSION_VARIABLE)).isTrue();
  }
}
