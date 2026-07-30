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
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import java.io.IOException;
import java.util.Iterator;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that C++ linking action is populated with the correct build variables. */
@RunWith(JUnit4.class)
public class LinkBuildVariablesTest extends LinkBuildVariablesTestCase {

  @Before
  public void createFooFooCcLibraryForRuleContext() throws IOException {
    scratch.file(
        "foo/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name = 'foo')");
  }

  @Test
  public void testIsUsingFissionIsIdenticalForCompileAndLink() {
    assertThat(LinkBuildVariables.IS_USING_FISSION.getVariableName())
        .isEqualTo(CompileBuildVariables.IS_USING_FISSION.getVariableName());
  }


  

  @Test
  public void testIsUsingFissionVariableUsingLegacyFields() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    scratch.file("x/foo.cc");

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));

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
    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'foo', srcs = ['foo.cc'])");
    scratch.file("x/foo.cc");

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PER_OBJECT_DEBUG_INFO));

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
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withSysroot("/usr/local/custom-sysroot"));
    useConfiguration();

    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'foo', srcs = ['a.cc'])");
    scratch.file("x/a.cc");

    ConfiguredTarget testTarget = getConfiguredTarget("//x:foo");
    CcToolchainVariables testVariables =
        getLinkBuildVariables(testTarget, LinkTargetType.EXECUTABLE);

    assertThat(testVariables.isAvailable(SYSROOT_VARIABLE_NAME)).isTrue();
  }

  @Test
  public void testUserLinkFlagsWithLinkoptOption() throws Exception {
    useConfiguration("--linkopt=-bar");

    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "cc_binary(name = 'foo', srcs = ['a.cc'], linkopts = ['-foo'])");
    scratch.file("x/a.cc");

    ConfiguredTarget testTarget = getConfiguredTarget("//x:foo");
    CcToolchainVariables testVariables =
        getLinkBuildVariables(testTarget, LinkTargetType.EXECUTABLE);

    ImmutableList<String> userLinkFlags =
        CcToolchainVariables.toStringList(
            testVariables, LinkBuildVariables.USER_LINK_FLAGS.getVariableName(), PathMapper.NOOP);
    assertThat(userLinkFlags).containsAtLeast("-foo", "-bar").inOrder();
  }

  @Test
  public void testLinkerInputsOverrideWholeArchive() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures("disable_whole_archive_for_static_lib_configuration"));

    scratch.file(
        "x/BUILD",
        "load('@rules_cc//cc:cc_binary.bzl', 'cc_binary')",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name='a', hdrs=['a.h'], srcs = ['a.cc'], "
            + " features=['disable_whole_archive_for_static_lib'])",
        "cc_library(name='b', hdrs=['b.h'], srcs = ['b.cc'], alwayslink=1)",
        "cc_binary(name = 'c.so', linkstatic=1, linkshared=1, deps=[':a', ':b'])");

    ConfiguredTarget testTarget = getConfiguredTarget("//x:c.so");
    CcToolchainVariables testVariables =
        getLinkBuildVariables(testTarget, LinkTargetType.DYNAMIC_LIBRARY);

    VariableValue librariesToLinkSequence =
        testVariables.getVariable(
            LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(), PathMapper.NOOP);
    assertThat(librariesToLinkSequence).isNotNull();
    Iterable<? extends VariableValue> librariesToLink =
        CcToolchainVariables.getSequenceValue(
            LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(), librariesToLinkSequence);
    assertThat(Iterables.size(librariesToLink)).isAnyOf(2, 3);

    Iterator<? extends VariableValue> librariesToLinkIterator = librariesToLink.iterator();
    // :a should not be whole archive
    VariableValue aWholeArchiveValue =
        librariesToLinkIterator
            .next()
            .getFieldValue(
                LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(), "is_whole_archive");
    assertThat(aWholeArchiveValue).isNotNull();
    assertThat(aWholeArchiveValue.isTruthy()).isFalse();

    // :b should be whole archive
    VariableValue bWholeArchiveValue =
        librariesToLinkIterator
            .next()
            .getFieldValue(
                LinkBuildVariables.LIBRARIES_TO_LINK.getVariableName(), "is_whole_archive");
    assertThat(bWholeArchiveValue).isNotNull();
    assertThat(bWholeArchiveValue.isTruthy()).isTrue();
  }
}
