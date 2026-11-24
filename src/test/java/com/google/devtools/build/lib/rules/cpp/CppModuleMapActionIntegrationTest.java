// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.ByteArrayOutputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CppModuleMapActionIntegrationTest extends BuildViewTestCase {

  @Before
  public final void setupCrosstoolMock() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.MODULE_MAPS));
    useConfiguration();
  }

  private Action getModuleMapAction(String targetName, String file, String... content)
      throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(MockCcSupport.HEADER_MODULES_FEATURES, CppRuleClasses.PARSE_HEADERS));
    useConfiguration();
    scratch.overwriteFile(file, content);

    invalidatePackages();
    ConfiguredTarget target = getConfiguredTarget(targetName);
    Artifact moduleMap =
        target.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    return getGeneratingAction(moduleMap);
  }

  @Test
  public void testCacheKeyDifferenceOnChangeOfExecRoot() throws Exception {
    ActionTester.runTest(
        KeyAttributes.class,
        new ActionCombinationFactory<KeyAttributes>() {
          @Override
          public Action generate(ImmutableSet<KeyAttributes> attributesToFlip) throws Exception {
            return getModuleMapAction(
                "//foo:a",
                "foo/BUILD",
                "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
                attributesToFlip.contains(KeyAttributes.GENRULE_FOR_A)
                    ? "genrule(name = 'gena', cmd = '', outs = ['a.h'])"
                    : "",
                attributesToFlip.contains(KeyAttributes.GENRULE_FOR_S)
                    ? "genrule(name = 'gens', cmd = '', outs = ['s.h'])"
                    : "",
                "cc_library(",
                "    name = 'a',",
                "    hdrs = ['a.h'],",
                "    srcs = ['s.h'],",
                ")");
          }
        },
        actionKeyContext);
  }

  @Test
  public void testCacheKeyDifference() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(MockCcSupport.HEADER_MODULES_FEATURES, CppRuleClasses.PARSE_HEADERS));
    useConfiguration();
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        package(features = ["modules"])

        cc_library(
            name = "a",
            srcs = ["a.h"],
        )
        """);
    ConfiguredTarget lib = getConfiguredTarget("//foo:a");
    Artifact moduleMap =
        lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    String privateHeadersKey =
        getGeneratingAction(moduleMap).getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    scratch.overwriteFile(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        package(features = ["modules"])

        cc_library(
            name = "a",
            hdrs = ["a.h"],
        )
        """);

    invalidatePackages();
    lib = getConfiguredTarget("//foo:a");
    moduleMap = lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    String publicHeadersKey =
        getGeneratingAction(moduleMap).getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    assertThat(publicHeadersKey).isNotEqualTo(privateHeadersKey);

    scratch.overwriteFile(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        package(features = ["header_modules"])

        cc_library(
            name = "a",
            hdrs = ["a.h"],
        )
        """);

    invalidatePackages();
    lib = getConfiguredTarget("//foo:a");
    moduleMap = lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    String headerModulesKey =
        getGeneratingAction(moduleMap).getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    assertThat(headerModulesKey).isNotEqualTo(publicHeadersKey);

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    MockCcSupport.HEADER_MODULES_FEATURES,
                    CppRuleClasses.PARSE_HEADERS,
                    CppRuleClasses.MODULE_MAP_HOME_CWD));
    useConfiguration();
    scratch.overwriteFile(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        package(features = ["modules"])

        cc_library(
            name = "a",
            hdrs = ["a.h"],
        )
        """);

    invalidatePackages();
    lib = getConfiguredTarget("//foo:a");
    moduleMap = lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    String homeIsCwdKey =
        getGeneratingAction(moduleMap).getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    assertThat(homeIsCwdKey).isNotEqualTo(publicHeadersKey);

    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    MockCcSupport.HEADER_MODULES_FEATURES,
                    CppRuleClasses.PARSE_HEADERS,
                    CppRuleClasses.GENERATE_SUBMODULES));
    useConfiguration();
    scratch.overwriteFile(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        package(features = [
            "modules",
            "generate_submodules",
        ])

        cc_library(
            name = "a",
            hdrs = ["a.h"],
        )
        """);

    invalidatePackages();
    lib = getConfiguredTarget("//foo:a");
    moduleMap = lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    String generateSubmodulesKey =
        getGeneratingAction(moduleMap).getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    assertThat(generateSubmodulesKey).isNotEqualTo(publicHeadersKey);
  }

  @Test
  public void testCacheKeyDifferenceOnAdditionalExportedHeaders() throws Exception {
    scratch.file("bazel_internal/test_rules/cc/BUILD", "exports_files(['rule.bzl'])");
    scratch.file(
        "bazel_internal/test_rules/cc/rule.bzl",
        """
        load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
        load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

        def _impl(ctx):
            cc_toolchain = ctx.toolchains["%s//tools/cpp:toolchain_type"].cc
            feature_configuration = cc_common.configure_features(
                ctx = ctx,
                cc_toolchain = cc_toolchain,
                requested_features = ctx.features,
                unsupported_features = ctx.disabled_features,
            )
            compilation_context, _ = cc_common.compile(
                actions = ctx.actions,
                feature_configuration = feature_configuration,
                cc_toolchain = cc_toolchain,
                name = ctx.label.name,
                additional_exported_hdrs = ctx.attr.additional_exported_hdrs,
            )
            return [CcInfo(compilation_context = compilation_context)]

        cc_additional_exported_hdrs_lib = rule(
            implementation = _impl,
            attrs = {"additional_exported_hdrs": attr.string_list()},
            toolchains = ["%s//tools/cpp:toolchain_type"],
            fragments = ["cpp"],
        )
        """
            .formatted(TestConstants.TOOLS_REPOSITORY, TestConstants.TOOLS_REPOSITORY));
    scratch.file(
        "foo/BUILD",
        """
        load("//bazel_internal/test_rules/cc:rule.bzl", "cc_additional_exported_hdrs_lib")

        cc_additional_exported_hdrs_lib(
            name = "ac",
            additional_exported_hdrs = ["foo/b.pb.h"],
        )
        """);
    ConfiguredTarget lib = getConfiguredTarget("//foo:ac");
    Artifact moduleMap =
        lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    String additionalExportedHeadersKey =
        getGeneratingAction(moduleMap).getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    scratch.overwriteFile(
        "foo/BUILD",
        """
        load("//bazel_internal/test_rules/cc:rule.bzl", "cc_additional_exported_hdrs_lib")

        cc_additional_exported_hdrs_lib(
            name = "ac",
            additional_exported_hdrs = ["foo/b2.pb.h"],
        )
        """);
    invalidatePackages();
    lib = getConfiguredTarget("//foo:ac");
    moduleMap = lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    String differentExportedHeadersKey =
        getGeneratingAction(moduleMap).getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    assertThat(differentExportedHeadersKey).isNotEqualTo(additionalExportedHeadersKey);
  }

  @Test
  public void testCacheKeyDifferenceOnRenamedDependencyWithSameExecPath() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        cc_library(
            name = "bar/baz",
            hdrs = ["bar/baz.h"],
        )

        cc_library(
            name = "foo",
            deps = [":bar/baz"],
        )
        """);
    // The module name is foo:bar/baz
    String keyWithOldName =
        getCppMapGeneratingAction("//foo")
            .getKey(actionKeyContext, /* inputMetadataProvider= */ null);
    scratch.overwriteFile(
        "foo/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name='foo', deps=['//foo/bar:baz'])");
    scratch.file(
        "foo/bar/BUILD",
        "load('@rules_cc//cc:cc_library.bzl', 'cc_library')",
        "cc_library(name='baz', hdrs=['baz.h'])");
    invalidatePackages();
    // New module name is foo/bar:baz
    Action action = getCppMapGeneratingAction("//foo");

    String keyWithNewName = action.getKey(actionKeyContext, /* inputMetadataProvider= */ null);

    assertThat(keyWithNewName).isNotEqualTo(keyWithOldName);
  }

  private Action getCppMapGeneratingAction(String target) throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget(target);
    Artifact moduleMap =
        configuredTarget
            .get(CcInfo.PROVIDER)
            .getCcCompilationContext()
            .getCppModuleMap()
            .getArtifact();
    return getGeneratingAction(moduleMap);
  }

  private String getModuleMapWithFeatures(String... args) throws Exception {
    invalidatePackages();
    useConfiguration(args);
    scratch.file(
        "foo/BUILD",
        """
        load("@rules_cc//cc:cc_library.bzl", "cc_library")

        cc_library(
            name = "a",
            srcs = ["a.h"],
            textual_hdrs = ["t.h"],
        )
        """);
    ConfiguredTarget lib = getConfiguredTarget("//foo:a");
    Artifact moduleMap =
        lib.get(CcInfo.PROVIDER).getCcCompilationContext().getCppModuleMap().getArtifact();
    AbstractFileWriteAction action = (AbstractFileWriteAction) getGeneratingAction(moduleMap);
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    FakeActionInputFileCache fakeInputMetadata = new FakeActionInputFileCache();
    ActionExecutionContext actionContext =
        ActionsTestUtil.createContext(
            new DummyExecutor(),
            reporter,
            new ActionKeyContext(),
            null,
            fakeInputMetadata,
            null,
            ImmutableMap.of());
    action.newDeterministicWriter(actionContext).writeTo(output);
    return output.toString(UTF_8);
  }

  @Test
  public void testModuleMapHomeCwd() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.MODULE_MAPS, CppRuleClasses.MODULE_MAP_HOME_CWD));
    assertThat(getModuleMapWithFeatures()).contains("private textual header \"foo/a.h\"");
  }

  @Test
  public void testModuleMapHomeIsModuleMapPath() throws Exception {
    assertThat(getModuleMapWithFeatures())
        .contains("private textual header \"../../../../foo/a.h\"");
  }

  @Test
  public void testTextualHeaders() throws Exception {
    assertThat(getModuleMapWithFeatures()).contains("textual header \"../../../../foo/t.h\"");
  }

  @Test
  public void testExternModule() throws Exception {
    assertThat(getModuleMapWithFeatures()).contains("extern module \"crosstool\"");
  }

  @Test
  public void testNoExternModule() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.MODULE_MAPS, CppRuleClasses.MODULE_MAP_WITHOUT_EXTERN_MODULE));
    assertThat(getModuleMapWithFeatures("--features=module_map_without_extern_module"))
        .doesNotContain("extern module \"crosstool\"");
  }

  @Test
  public void testGeneratesSubmodules() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(CppRuleClasses.MODULE_MAPS, CppRuleClasses.GENERATE_SUBMODULES));
    assertThat(getModuleMapWithFeatures("--features=generate_submodules"))
        .contains("module \"foo/a.h\"");
  }

  private enum KeyAttributes {
    GENRULE_FOR_A,
    GENRULE_FOR_S
  }
}

