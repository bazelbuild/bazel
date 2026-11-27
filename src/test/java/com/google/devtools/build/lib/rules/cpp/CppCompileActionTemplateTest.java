// Copyright 2007 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link CppCompileActionTemplate} and its integration with C++ rule common logic. */
@RunWith(JUnit4.class)
public final class CppCompileActionTemplateTest extends BuildViewTestCase {

  @Before
  public void compileActionTemplateLibraryBzl() throws Exception {
    scratch.file("test/BUILD");
    scratch.file(
        "test/cc_template_library.bzl",
"""
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def _cc_template_library_impl(ctx):
    headers = ctx.actions.declare_directory("dummy/main/cc_headers")
    sources = ctx.actions.declare_directory("dummy/main/cc_sources")
    additional_sources = ctx.actions.declare_directory("dummy/additional/cc_sources")
    ctx.actions.run_shell(
        outputs = [headers, sources, additional_sources],
        command = "touch %s && touch %s && touch %s" % (headers.path, sources.path, additional_sources.path),
    )

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = ctx.toolchains["__TOOLS_REPOSITORY__//tools/cpp:toolchain_type"].cc,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    _, compilation_outputs = cc_common.compile(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = ctx.toolchains["__TOOLS_REPOSITORY__//tools/cpp:toolchain_type"].cc,
        srcs = [sources, additional_sources],
        public_hdrs = [headers],
        includes = ["DUMMY_INCLUDE"],
        defines = ["DUMMY_DEFINE"],
        name = ctx.label.name,
    )

    pic_mode = ctx.attr.pic_mode
    object_files = []
    if pic_mode == "on" or pic_mode == "both":
        object_files.extend(compilation_outputs.pic_objects)
    if pic_mode == "off" or pic_mode == "both":
        object_files.extend(compilation_outputs.objects)
    files_to_build = depset(object_files + compilation_outputs._header_tokens)
    return [DefaultInfo(files = files_to_build)]

cc_template_library = rule(
    implementation = _cc_template_library_impl,
    attrs = {
        "pic_mode": attr.string(default = "off"),
    },
    toolchains = ["__TOOLS_REPOSITORY__//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)
"""
            .replace("__TOOLS_REPOSITORY__", TestConstants.TOOLS_REPOSITORY.toString()));
  }

  private void validateObjectFileArtifact(Artifact objectFiles, String binDir, boolean usesPic)
      throws Exception {
    assertThat(objectFiles.isTreeArtifact()).isTrue();

    ActionAnalysisMetadata metadata = getActionGraph().getGeneratingAction(objectFiles);
    CppCompileActionTemplate compileActionTemplate = (CppCompileActionTemplate) metadata;

    assertThat(Artifact.toRootRelativePaths(compileActionTemplate.getInputs()))
        .containsAtLeast(
            "main/dummy/main/cc_sources",
            TestConstants.MOCK_CC_CROSSTOOL_PATH + "/crosstool.cppmap");
    assertThat(compileActionTemplate.getPrimaryInput().getRootRelativePathString())
        .isEqualTo("main/dummy/main/cc_sources");

    Artifact sources = getFirstArtifactEndingWith(compileActionTemplate.getInputs(), "cc_sources");
    TreeFileArtifact sourceTreeFileArtifact =
        TreeFileArtifact.createTreeOutput((TreeArtifact) sources, "children/child0.cc");
    TreeFileArtifact headerTreeFileArtifact =
        TreeFileArtifact.createTreeOutput((TreeArtifact) sources, "children/child0.h");
    ImmutableList<CppCompileAction> compileActions =
        compileActionTemplate.generateActionsForInputArtifacts(
            ImmutableList.of(sourceTreeFileArtifact, headerTreeFileArtifact),
            ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER,
            NullEventHandler.INSTANCE);
    CppCompileAction compileAction = Iterables.getOnlyElement(compileActions);

    assertThat(Artifact.toRootRelativePaths(compileAction.getInputs()))
        .containsAtLeast(
            "main/dummy/main/cc_sources/children/child0.cc",
            TestConstants.MOCK_CC_CROSSTOOL_PATH + "/crosstool.cppmap");
    assertThat(compileAction.getIncludeDirs()).contains(PathFragment.create("DUMMY_INCLUDE"));
    assertThat(compileAction.getDefines()).contains("DUMMY_DEFINE");

    if (usesPic) {
      assertThat(objectFiles.getExecPathString())
          .isEqualTo(binDir + "/main/_pic_objs/main/0/cc_sources");
      assertThat(Artifact.toRootRelativePaths(compileActionTemplate.getOutputs()))
          .contains("main/_pic_objs/main/0/cc_sources");
      assertThat(compileActionTemplate.getPrimaryOutput().getRootRelativePathString())
          .isEqualTo("main/_pic_objs/main/0/cc_sources");
      assertThat(Artifact.toRootRelativePaths(compileAction.getOutputs()))
          .contains("main/_pic_objs/main/0/cc_sources/children/child0.pic.o");
    } else {
      assertThat(objectFiles.getExecPathString())
          .isEqualTo(binDir + "/main/_objs/main/0/cc_sources");
      assertThat(Artifact.toRootRelativePaths(compileActionTemplate.getOutputs()))
          .contains("main/_objs/main/0/cc_sources");
      assertThat(compileActionTemplate.getPrimaryOutput().getRootRelativePathString())
          .isEqualTo("main/_objs/main/0/cc_sources");
      assertThat(Artifact.toRootRelativePaths(compileAction.getOutputs()))
          .contains("main/_objs/main/0/cc_sources/children/child0.o");
    }
  }

  @Test
  public void testSourceCppCompileActionTemplate() throws Exception {
    // We need to set opt mode or else non-PIC actions aren't created.
    useConfiguration("--compilation_mode=opt");
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    scratch.file(
        "main/BUILD",
        "load('//test:cc_template_library.bzl', 'cc_template_library')",
        "cc_template_library(name = 'main', pic_mode = 'off')");
    ConfiguredTarget target = getConfiguredTarget("//main:main");
    FileProvider fileProvider = Preconditions.checkNotNull(target).getProvider(FileProvider.class);
    Artifact objectFiles = fileProvider.getFilesToBuild().toList().get(0);
    String binDir = getRuleContext(target).getBinDirectory().getExecPathString();

    validateObjectFileArtifact(objectFiles, binDir, false);
  }

  @Test
  public void testForcePicSourceCppCompileActionTemplate() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    scratch.file(
        "main/BUILD",
        "load('//test:cc_template_library.bzl', 'cc_template_library')",
        "cc_template_library(name = 'main', pic_mode = 'on')");
    ConfiguredTarget target = getConfiguredTarget("//main:main");
    Artifact objectFiles = target.getProvider(FileProvider.class).getFilesToBuild().toList().get(0);
    String binDir = getRuleContext(target).getBinDirectory().getExecPathString();
    validateObjectFileArtifact(objectFiles, binDir, true);
  }

  @Test
  public void testPicSourceCppCompileActionTemplateNotCreatedInFastbuild() throws Exception {
    // When compiling in fastbuild mode, only PIC actions are created.
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    scratch.file(
        "main/BUILD",
        "load('//test:cc_template_library.bzl', 'cc_template_library')",
        "cc_template_library(name = 'main', pic_mode = 'both')");
    ConfiguredTarget target = getConfiguredTarget("//main:main");
    ImmutableList<Artifact> filesToBuild =
        target.getProvider(FileProvider.class).getFilesToBuild().toList();
    assertThat(filesToBuild).hasSize(2);
    Artifact picObjectFiles = filesToBuild.get(0);
    String binDir = getRuleContext(target).getBinDirectory().getExecPathString();

    validateObjectFileArtifact(picObjectFiles, binDir, true);
  }

  @Test
  public void testPicSourceCppCompileActionTemplate() throws Exception {
    // When compiling in opt mode, PIC and non-PIC actions are created.
    useConfiguration("--compilation_mode=opt");
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    scratch.file(
        "main/BUILD",
        "load('//test:cc_template_library.bzl', 'cc_template_library')",
        "cc_template_library(name = 'main', pic_mode = 'both')");
    ConfiguredTarget target = getConfiguredTarget("//main:main");
    Artifact picObjectFiles =
        target.getProvider(FileProvider.class).getFilesToBuild().toList().get(0);
    Artifact nonPicObjectFiles =
        target.getProvider(FileProvider.class).getFilesToBuild().toList().get(2);
    String binDir = getRuleContext(target).getBinDirectory().getExecPathString();

    validateObjectFileArtifact(picObjectFiles, binDir, true);
    validateObjectFileArtifact(nonPicObjectFiles, binDir, false);
  }

  @Test
  public void testHeaderCppCompileActionTemplate() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withFeatures(CppRuleClasses.PARSE_HEADERS));
    useConfiguration("--features=parse_headers", "--features=cc_include_scanning");
    scratch.file(
        "main/BUILD",
        "load('//test:cc_template_library.bzl', 'cc_template_library')",
        "cc_template_library(name = 'main')");
    ConfiguredTarget target = getConfiguredTarget("//main:main");
    Artifact headerTokenFiles =
        target.getProvider(FileProvider.class).getFilesToBuild().toList().get(2);
    String binDir = getRuleContext(target).getBinDirectory().getExecPathString();

    assertThat(headerTokenFiles.isTreeArtifact()).isTrue();
    assertThat(headerTokenFiles.getExecPathString())
        .isEqualTo(binDir + "/main/_objs/main/cc_headers");

    ActionAnalysisMetadata metadata = getActionGraph().getGeneratingAction(headerTokenFiles);
    CppCompileActionTemplate compileActionTemplate = (CppCompileActionTemplate) metadata;

    Artifact headers = getFirstArtifactEndingWith(compileActionTemplate.getInputs(), "cc_headers");
    TreeFileArtifact headerTreeFileArtifact =
        TreeFileArtifact.createTreeOutput((TreeArtifact) headers, "children/child0.h");
    ImmutableList<CppCompileAction> headerCompileActions =
        compileActionTemplate.generateActionsForInputArtifacts(
            ImmutableList.of(headerTreeFileArtifact),
            ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER,
            NullEventHandler.INSTANCE);
    CppCompileAction headerCompileAction = Iterables.getOnlyElement(headerCompileActions);

    assertThat(Artifact.toRootRelativePaths(headerCompileAction.getInputs()))
        .containsAtLeast(
            "main/dummy/main/cc_headers/children/child0.h",
            TestConstants.MOCK_CC_CROSSTOOL_PATH + "/crosstool.cppmap");
    assertThat(Artifact.toRootRelativePaths(headerCompileAction.getOutputs()))
        .contains("main/_objs/main/cc_headers/children/child0.h.processed");
    assertThat(headerCompileAction.getIncludeDirs()).contains(PathFragment.create("DUMMY_INCLUDE"));
    assertThat(headerCompileAction.getDefines()).contains("DUMMY_DEFINE");
  }

  @Before
  public void setUp() throws Exception {
    useConfiguration("--features=cc_include_scanning");
  }

  private void writeTreeArtifactCcBzlAndHeader() throws Exception {
    scratch.file(
        "main/tree_artifact_cc.bzl",
        """
        def _tree_artifact_cc_impl(ctx):
            directory = ctx.actions.declare_directory(ctx.attr.name + "_artifact")
            ctx.actions.run_shell(
                outputs = [directory],
                use_default_shell_env = True,
                command = 'echo \\'#include "main/lib.h"\\' > %s/main.cc' % directory.path,
            )
            return [DefaultInfo(files = depset([directory]))]

        tree_artifact_cc = rule(implementation = _tree_artifact_cc_impl)
        """);
    scratch.file("main/lib.h", "const int CONSTANT = 1;");
  }

  @Test
  public void getKey_same() throws Exception {
    ActionKeyContext keyContext = new ActionKeyContext();
    writeTreeArtifactCcBzlAndHeader();
    scratch.file(
        "main/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load(":tree_artifact_cc.bzl", "tree_artifact_cc")

        tree_artifact_cc(name = "tree")

        cc_library(
            name = "lib",
            srcs = [":tree"],
            hdrs = ["lib.h"],
        )
        """);

    CppCompileActionTemplate libTemplate = findTemplate("//main:lib");
    createBuildView();
    CppCompileActionTemplate otherLibTemplate = findTemplate("//main:lib");

    assertThat(libTemplate).isNotSameInstanceAs(otherLibTemplate);
    assertThat(libTemplate.getKey(keyContext, /* inputMetadataProvider= */ null))
        .isEqualTo(otherLibTemplate.getKey(keyContext, /* inputMetadataProvider= */ null));
  }

  @Test
  public void testUsesCorrectModulesForNonPicCompiles() throws Exception {
    useConfiguration("--compilation_mode=opt", "--features=cc_include_scanning");
    CppCompileAction compileAction = createTestCaseUsingModules();
    assertThat(compileAction.getPrimaryOutput().getFilename()).isEqualTo("child0.o");
    assertThat(artifactsToStrings(compileAction.getAllowedDerivedInputs()))
        .contains("bin main/_objs/a/a.pcm");
  }

  @Test
  public void testUsesCorrectModulesForPicCompiles() throws Exception {
    CppCompileAction compileAction = createTestCaseUsingModules();
    assertThat(compileAction.getPrimaryOutput().getFilename()).isEqualTo("child0.pic.o");
    assertThat(artifactsToStrings(compileAction.getAllowedDerivedInputs()))
        .contains("bin main/_objs/a/a.pic.pcm");
  }

  private CppCompileAction createTestCaseUsingModules() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(MockCcSupport.HEADER_MODULES_FEATURES, CppRuleClasses.SUPPORTS_PIC));
    writeTreeArtifactCcBzlAndHeader();
    scratch.file(
        "main/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load(":tree_artifact_cc.bzl", "tree_artifact_cc")

        package(features = ["use_header_modules"])

        tree_artifact_cc(name = "tree")

        cc_library(
            name = "a",
            hdrs = ["a.h"],
            features = ["header_modules"],
        )

        cc_library(
            name = "lib",
            srcs = [":tree"],
            deps = [":a"],
        )
        """);
    CppCompileActionTemplate template = findTemplate("//main:lib");
    Artifact sources = getFirstArtifactEndingWith(template.getInputs(), "tree_artifact");
    TreeFileArtifact sourceTreeFileArtifact =
        TreeFileArtifact.createTreeOutput((TreeArtifact) sources, "children/child0.cc");
    ImmutableList<CppCompileAction> compileActions =
        template.generateActionsForInputArtifacts(
            ImmutableList.of(sourceTreeFileArtifact),
            ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER,
            NullEventHandler.INSTANCE);
    return Iterables.getOnlyElement(compileActions);
  }

  @Test
  public void testUsesCorrectModulesForNonPicHeaderCompiles() throws Exception {
    useConfiguration("--compilation_mode=opt", "--features=cc_include_scanning");
    CppCompileAction compileAction = createTestCaseUsingModulesForParseHeaders();
    // Header compile actions will always prefer PIC if available.
    assertThat(compileAction.getPrimaryOutput().getFilename()).isEqualTo("child0.h.processed");
    assertThat(artifactsToStrings(compileAction.getAllowedDerivedInputs()))
        .contains("bin main/_objs/a/a.pic.pcm");
  }

  @Test
  public void testUsesCorrectModulesForPicHeaderCompiles() throws Exception {
    CppCompileAction compileAction = createTestCaseUsingModulesForParseHeaders();
    assertThat(compileAction.getPrimaryOutput().getFilename()).isEqualTo("child0.h.processed");
    assertThat(artifactsToStrings(compileAction.getAllowedDerivedInputs()))
        .contains("bin main/_objs/a/a.pic.pcm");
  }

  private CppCompileAction createTestCaseUsingModulesForParseHeaders() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    MockCcSupport.HEADER_MODULES_FEATURES,
                    CppRuleClasses.PARSE_HEADERS,
                    CppRuleClasses.SUPPORTS_PIC));
    writeTreeArtifactCcBzlAndHeader();
    scratch.file(
        "main/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load(":tree_artifact_cc.bzl", "tree_artifact_cc")

        package(features = [
            "use_header_modules",
            "parse_headers",
        ])

        tree_artifact_cc(name = "tree")

        cc_library(
            name = "a",
            hdrs = ["a.h"],
            features = ["header_modules"],
        )

        cc_library(
            name = "lib",
            hdrs = [":tree"],
            deps = [":a"],
        )
        """);
    CppCompileActionTemplate template = findTemplate("//main:lib");
    Artifact sources = getFirstArtifactEndingWith(template.getInputs(), "tree_artifact");
    TreeFileArtifact headerTreeFileArtifact =
        TreeFileArtifact.createTreeOutput((TreeArtifact) sources, "children/child0.h");
    ImmutableList<CppCompileAction> compileActions =
        template.generateActionsForInputArtifacts(
            ImmutableList.of(headerTreeFileArtifact),
            ActionsTestUtil.NULL_TEMPLATE_EXPANSION_ARTIFACT_OWNER,
            NullEventHandler.INSTANCE);
    return Iterables.getOnlyElement(compileActions);
  }

  @Test
  public void getKey_differs() throws Exception {
    ActionKeyContext keyContext = new ActionKeyContext();
    writeTreeArtifactCcBzlAndHeader();
    scratch.file(
        "main/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")
        load(":tree_artifact_cc.bzl", "tree_artifact_cc")

        tree_artifact_cc(name = "tree")

        cc_library(
            name = "lib1",
            srcs = [":tree"],
            hdrs = ["lib.h"],
        )

        cc_library(
            name = "lib2",
            srcs = [":tree"],
            hdrs = ["lib.h"],
            copts = ["-DFOO"],
        )
        """);

    CppCompileActionTemplate lib1Template = findTemplate("//main:lib1");
    CppCompileActionTemplate lib2Template = findTemplate("//main:lib2");

    assertThat(lib1Template.getKey(keyContext, /* inputMetadataProvider= */ null))
        .isNotEqualTo(lib2Template.getKey(keyContext, /* inputMetadataProvider= */ null));
  }

  private CppCompileActionTemplate findTemplate(String target) throws Exception {
    RuleConfiguredTarget configuredTarget = (RuleConfiguredTarget) getConfiguredTarget(target);
    for (ActionAnalysisMetadata metadata : configuredTarget.getActions()) {
      if (metadata instanceof CppCompileActionTemplate cppCompileActionTemplate) {
        return cppCompileActionTemplate;
      }
    }
    throw new AssertionError("No CppCompileActionTemplate in " + configuredTarget.getActions());
  }

}
