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
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that {@link CppLinkstampCompileHelper} creates sane compile actions for linkstamps */
@RunWith(JUnit4.class)
public class CppLinkstampCompileHelperTest extends BuildViewTestCase {

  /** Tests that linkstamp compilation applies expected command line options. */
  @Test
  public void testLinkstampCompileOptionsForExecutable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withSysroot("/usr/local/custom-sysroot"));
    useConfiguration();
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            deps = ["a"],
        )

        cc_library(
            name = "a",
            srcs = ["a.cc"],
            linkstamp = "ls.cc",
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);

    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);

    CcToolchainProvider ccToolchainProvider =
        getConfiguredTarget(
                ruleClassProvider.getToolsRepository() + "//tools/cpp:current_cc_toolchain")
            .get(CcToolchainProvider.PROVIDER);

    List<String> arguments = linkstampCompileAction.getArguments();
    assertThatArgumentsAreValid(
        arguments,
        ccToolchainProvider.getToolchainIdentifier(),
        target.getLabel().getCanonicalForm(),
        executable.getFilename());
  }

  private void assertThatArgumentsAreValid(
      List<String> arguments, String platform, String targetName, String buildTargetNameSuffix) {
    assertThat(arguments).contains("--sysroot=/usr/local/custom-sysroot");
    assertThat(arguments).contains("-include");
    assertThat(arguments).contains("-DG3_TARGET_NAME=\"" + targetName + "\"");
    assertThat(arguments).contains("-DGPLATFORM=\"" + platform + "\"");
    assertThat(arguments).contains("-I.");
    String correctG3BuildTargetPattern = "-DG3_BUILD_TARGET=\".*" + buildTargetNameSuffix + "\"";
    assertWithMessage("in " + arguments + " flag matching " + correctG3BuildTargetPattern)
        .that(Iterables.tryFind(arguments, (arg) -> arg.matches(correctG3BuildTargetPattern)))
        .isPresent();
    String fdoStampPattern = "-D" + CppConfiguration.FDO_STAMP_MACRO + "=\".*\"";
    assertWithMessage("in " + arguments + " flag matching " + fdoStampPattern)
        .that(Iterables.tryFind(arguments, (arg) -> arg.matches(fdoStampPattern)))
        .isAbsent();
  }

  /** Tests that linkstamp compilation applies expected command line options. */
  @Test
  public void testLinkstampCompileOptionsForSharedLibrary() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withSysroot("/usr/local/custom-sysroot"));
    useConfiguration();
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "libfoo.so",
            linkshared = 1,
            deps = ["a"],
        )

        cc_library(
            name = "a",
            srcs = ["a.cc"],
            linkstamp = "ls.cc",
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//x:libfoo.so");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);
    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    assertThat(generatingAction.getInputs().toList()).contains(compiledLinkstamp);

    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);
    CcToolchainProvider ccToolchainProvider =
        getConfiguredTarget(
                ruleClassProvider.getToolsRepository() + "//tools/cpp:current_cc_toolchain")
            .get(CcToolchainProvider.PROVIDER);

    List<String> arguments = linkstampCompileAction.getArguments();
    assertThatArgumentsAreValid(
        arguments,
        ccToolchainProvider.getToolchainIdentifier(),
        target.getLabel().getCanonicalForm(),
        executable.getFilename());
  }

  @Test
  public void testLinkstampRespectsPicnessFromConfiguration() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.SUPPORTS_PIC, CppRuleClasses.PIC));

    useConfiguration("--force_pic");
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            deps = ["a"],
        )

        cc_library(
            name = "a",
            srcs = ["a.cc"],
            linkstamp = "ls.cc",
        )
        """);
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);
    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    assertThat(generatingAction.getInputs().toList()).contains(compiledLinkstamp);

    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);
    assertThat(linkstampCompileAction.getArguments()).contains("-fPIC");
  }

  @Test
  public void testLinkstampRespectsFdoFromConfiguration() throws Exception {
    useConfiguration("--fdo_instrument=foo");
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            deps = ["a"],
        )

        cc_library(
            name = "a",
            srcs = ["a.cc"],
            linkstamp = "ls.cc",
        )
        """);
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);
    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    assertThat(generatingAction.getInputs().toList()).contains(compiledLinkstamp);

    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);
    assertThat(linkstampCompileAction.getArguments())
        .contains("-D" + CppConfiguration.FDO_STAMP_MACRO + "=\"FDO\"");
  }

  /**
   * Regression test for b/73447914: Linkstamps were not re-built when only volatile data changed,
   * i.e. when we modified cc_binary source, linkstamp was not recompiled so we got old timestamps.
   * The proper behavior is to recompile linkstamp whenever any input to cc_binary action changes.
   * And the current implementation solves this by adding all linking inputs as
   * inputsForInvalidation to linkstamp compile action.
   */
  @Test
  public void testLinkstampCompileDependsOnAllCcBinaryLinkingInputs() throws Exception {
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            srcs = ["main.cc"],
            deps = ["bar"],
        )

        cc_library(
            name = "bar",
            srcs = ["bar.cc"],
            linkstamp = "ls.cc",
        )
        """);
    useConfiguration();

    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CcToolchainProvider toolchain = CppHelper.getToolchain(getRuleContext(target));
    CppConfiguration cppConfiguration = getRuleContext(target).getFragment(CppConfiguration.class);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrThrowEvalException(
            /* requestedFeatures= */ ImmutableSet.of(),
            /* unsupportedFeatures= */ ImmutableSet.of(),
            Language.CPP,
            toolchain,
            cppConfiguration);
    boolean usePic = CppHelper.usePicForBinaries(cppConfiguration, featureConfiguration);

    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);

    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);

    Artifact mainObject =
        ActionsTestUtil.getFirstArtifactEndingWith(
            generatingAction.getInputs(), usePic ? "main.pic.o" : "main.o");
    Artifact bar =
        generatingAction.getInputs().toList().stream()
            .filter(a -> a.getExecPath().getBaseName().contains("bar"))
            .findFirst()
            .get();
    ImmutableList<Artifact> linkstampInputs = linkstampCompileAction.getInputs().toList();
    assertThat(linkstampInputs).containsAtLeast(mainObject, bar);
  }

  @Test
  public void testLinkstampGetsCoptsFromOptions() throws Exception {
    useConfiguration("--copt=-foo_copt_from_option");
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            copts = ["-bar_copt_from_attribute"],
            deps = ["a"],
        )

        cc_library(
            name = "a",
            srcs = ["a.cc"],
            linkstamp = "ls.cc",
        )
        """);
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);
    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    assertThat(generatingAction.getInputs().toList()).contains(compiledLinkstamp);

    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);
    assertThat(linkstampCompileAction.getArguments()).contains("-foo_copt_from_option");
  }

  @Test
  public void testLinkstampDoesNotGetCoptsFromAttribute() throws Exception {
    useConfiguration("--copt=-foo_copt_from_option");
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            copts = ["-bar_copt_from_attribute"],
            deps = ["a"],
        )

        cc_library(
            name = "a",
            srcs = ["a.cc"],
            copts = ["-baz_copt_from_attribute"],
            linkstamp = "ls.cc",
        )
        """);
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);
    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    assertThat(generatingAction.getInputs().toList()).contains(compiledLinkstamp);

    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);
    assertThat(linkstampCompileAction.getArguments()).doesNotContain("-bar_copt_from_attribute");
    assertThat(linkstampCompileAction.getArguments()).doesNotContain("-baz_copt_from_attribute");
  }

  @Test
  public void testLinkstampCompileIsUsingMemProf() throws Exception {
    useConfiguration(
        "--compilation_mode=opt", "--features=memprof_optimize", "--fdo_profile=//x:prof");
    scratch.file(
        "x/BUILD",
        """
        cc_binary(
            name = "foo",
            deps = ["a"],
        )

        cc_library(
            name = "a",
            srcs = ["a.cc"],
            linkstamp = "ls.cc",
        )

        fdo_profile(
          name = "prof",
          profile = "out.afdo",
          memprof_profile = "memprof.zip",
        )
        """);
    ConfiguredTarget target = getConfiguredTarget("//x:foo");
    Artifact executable = getExecutable(target);
    CppLinkAction generatingAction = (CppLinkAction) getGeneratingAction(executable);
    Artifact compiledLinkstamp =
        ActionsTestUtil.getFirstArtifactEndingWith(generatingAction.getInputs(), "ls.o");
    assertThat(generatingAction.getInputs().toList()).contains(compiledLinkstamp);

    CppCompileAction linkstampCompileAction =
        (CppCompileAction) getGeneratingAction(compiledLinkstamp);
    CompileCommandLine cmdline = linkstampCompileAction.getCompileCommandLine();
    CcToolchainVariables variables = cmdline.getVariables();
    assertThat(variables.isAvailable("is_using_memprof")).isTrue();
  }
}
