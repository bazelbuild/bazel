// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.SkylarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ArtifactNamePattern;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvEntry;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagGroup;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Tool;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.VariableWithValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.WithFeatureSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.StringValueParser;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.MakeVariable;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@code cc_common} Skylark module.
 */
@RunWith(JUnit4.class)
public class SkylarkCcCommonTest extends BuildViewTestCase {

  @Before
  public void setSkylarkSemanticsOptions() throws Exception {
    setSkylarkSemanticsOptions(SkylarkCcCommonTestHelper.CC_SKYLARK_WHITELIST_FLAG);
    invalidatePackages();
  }

  @Test
  public void testGetToolForAction() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  return struct(",
        "    action_tool_path = cc_common.get_tool_for_action(",
        "        feature_configuration = feature_configuration,",
        "        action_name = 'c++-compile'))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    String actionToolPath = (String) r.get("action_tool_path");
    RuleContext ruleContext = getRuleContext(r);
    CcToolchainProvider toolchain =
        CppHelper.getToolchain(
            ruleContext, ruleContext.getPrerequisite("$cc_toolchain", Mode.TARGET));
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrThrowEvalException(
            ImmutableSet.of(), ImmutableSet.of(), toolchain);
    assertThat(actionToolPath)
        .isEqualTo(featureConfiguration.getToolPathForAction(CppActionNames.CPP_COMPILE));
  }

  @Test
  public void testFeatureConfigurationWithAdditionalEnabledFeature() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "feature { name: 'foo_feature' }");
    useConfiguration();
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(",
        "      cc_toolchain = toolchain,",
        "      requested_features = ['foo_feature'])",
        "  return struct(",
        "    foo_feature_enabled = cc_common.is_enabled(",
        "        feature_configuration = feature_configuration,",
        "        feature_name = 'foo_feature'))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    boolean fooFeatureEnabled = (boolean) r.get("foo_feature_enabled");
    assertThat(fooFeatureEnabled).isTrue();
  }

  @Test
  public void testFeatureConfigurationWithAdditionalUnsupportedFeature() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "feature { name: 'foo_feature' }");
    useConfiguration("--features=foo_feature");
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(",
        "      cc_toolchain = toolchain,",
        "      unsupported_features = ['foo_feature'])",
        "  return struct(",
        "    foo_feature_enabled = cc_common.is_enabled(",
        "        feature_configuration = feature_configuration,",
        "        feature_name = 'foo_feature'))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    boolean fooFeatureEnabled = (boolean) r.get("foo_feature_enabled");
    assertThat(fooFeatureEnabled).isFalse();
  }

  @Test
  public void testGetCommandLine() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  return struct(",
        "    command_line = cc_common.get_memory_inefficient_command_line(",
        "        feature_configuration = feature_configuration,",
        "        action_name = 'c++-link-executable',",
        "        variables = cc_common.empty_variables()))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    SkylarkList<String> commandLine = (SkylarkList<String>) r.get("command_line");
    RuleContext ruleContext = getRuleContext(r);
    CcToolchainProvider toolchain =
        CppHelper.getToolchain(
            ruleContext, ruleContext.getPrerequisite("$cc_toolchain", Mode.TARGET));
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrThrowEvalException(
            ImmutableSet.of(), ImmutableSet.of(), toolchain);
    assertThat(commandLine)
        .containsExactlyElementsIn(
            featureConfiguration.getCommandLine("c++-link-executable", CcToolchainVariables.EMPTY));
  }

  @Test
  public void testGetEnvironment() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  return struct(",
        "    environment_variables = cc_common.get_environment_variables(",
        "        feature_configuration = feature_configuration,",
        "        action_name = 'c++-compile',",
        "        variables = cc_common.empty_variables()))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    SkylarkDict<String, String> environmentVariables =
        (SkylarkDict<String, String>) r.get("environment_variables");
    RuleContext ruleContext = getRuleContext(r);
    CcToolchainProvider toolchain =
        CppHelper.getToolchain(
            ruleContext, ruleContext.getPrerequisite("$cc_toolchain", Mode.TARGET));
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrThrowEvalException(
            ImmutableSet.of(), ImmutableSet.of(), toolchain);
    assertThat(environmentVariables)
        .containsExactlyEntriesIn(
            featureConfiguration.getEnvironmentVariables(
                CppActionNames.CPP_COMPILE, CcToolchainVariables.EMPTY));
  }

  @Test
  public void testActionIsEnabled() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  return struct(",
        "    enabled_action = cc_common.action_is_enabled(",
        "        feature_configuration = feature_configuration,",
        "        action_name = 'c-compile'),",
        "    disabled_action = cc_common.action_is_enabled(",
        "        feature_configuration = feature_configuration,",
        "        action_name = 'wololoo'))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    boolean enabledActionIsEnabled = (boolean) r.get("enabled_action");
    @SuppressWarnings("unchecked")
    boolean disabledActionIsDisabled = (boolean) r.get("disabled_action");
    assertThat(enabledActionIsEnabled).isTrue();
    assertThat(disabledActionIsDisabled).isFalse();
  }

  @Test
  public void testIsEnabled() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  return struct(",
        "    enabled_feature = cc_common.is_enabled(",
        "        feature_configuration = feature_configuration,",
        "        feature_name = 'libraries_to_link'),",
        "    disabled_feature = cc_common.is_enabled(",
        "        feature_configuration = feature_configuration,",
        "        feature_name = 'wololoo'))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    boolean enabledFeatureIsEnabled = (boolean) r.get("enabled_feature");
    @SuppressWarnings("unchecked")
    boolean disabledFeatureIsDisabled = (boolean) r.get("disabled_feature");
    assertThat(enabledFeatureIsEnabled).isTrue();
    assertThat(disabledFeatureIsDisabled).isFalse();
  }

  @Test
  public void testActionNames() throws Exception {
    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");
    scratch.file("tools/build_defs/cc/BUILD");
    scratch.file(
        "tools/build_defs/cc/action_names.bzl",
        ResourceLoader.readFromResources(
            TestConstants.BAZEL_REPO_PATH + "tools/build_defs/cc/action_names.bzl"));

    scratch.file(
        "a/rule.bzl",
        "load('//tools/build_defs/cc:action_names.bzl',",
        "    'C_COMPILE_ACTION_NAME',",
        "    'CPP_COMPILE_ACTION_NAME',",
        "    'LINKSTAMP_COMPILE_ACTION_NAME',",
        "    'CC_FLAGS_MAKE_VARIABLE_ACTION_NAME',",
        "    'CPP_MODULE_CODEGEN_ACTION_NAME',",
        "    'CPP_HEADER_PARSING_ACTION_NAME',",
        "    'CPP_MODULE_COMPILE_ACTION_NAME',",
        "    'ASSEMBLE_ACTION_NAME',",
        "    'PREPROCESS_ASSEMBLE_ACTION_NAME',",
        "    'LTO_INDEXING_ACTION_NAME',",
        "    'LTO_BACKEND_ACTION_NAME',",
        "    'CPP_LINK_EXECUTABLE_ACTION_NAME',",
        "    'CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME',",
        "    'CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME',",
        "    'CPP_LINK_STATIC_LIBRARY_ACTION_NAME',",
        "    'STRIP_ACTION_NAME')",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  return struct(",
        "      c_compile_action_name=C_COMPILE_ACTION_NAME,",
        "      cpp_compile_action_name=CPP_COMPILE_ACTION_NAME,",
        "      linkstamp_compile_action_name=LINKSTAMP_COMPILE_ACTION_NAME,",
        "      cc_flags_make_variable_action_name_action_name=CC_FLAGS_MAKE_VARIABLE_ACTION_NAME,",
        "      cpp_module_codegen_action_name=CPP_MODULE_CODEGEN_ACTION_NAME,",
        "      cpp_header_parsing_action_name=CPP_HEADER_PARSING_ACTION_NAME,",
        "      cpp_module_compile_action_name=CPP_MODULE_COMPILE_ACTION_NAME,",
        "      assemble_action_name=ASSEMBLE_ACTION_NAME,",
        "      preprocess_assemble_action_name=PREPROCESS_ASSEMBLE_ACTION_NAME,",
        "      lto_indexing_action_name=LTO_INDEXING_ACTION_NAME,",
        "      lto_backend_action_name=LTO_BACKEND_ACTION_NAME,",
        "      cpp_link_executable_action_name=CPP_LINK_EXECUTABLE_ACTION_NAME,",
        "      cpp_link_dynamic_library_action_name=CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME,",
        "      cpp_link_nodeps_dynamic_library_action_name=CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME,",
        "      cpp_link_static_library_action_name=CPP_LINK_STATIC_LIBRARY_ACTION_NAME,",
        "      strip_action_name=STRIP_ACTION_NAME)",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ")");

    assertThat(getTarget("//a:r")).isNotNull();
    ConfiguredTarget r = getConfiguredTarget("//a:r");
    assertThat(r.get("c_compile_action_name")).isEqualTo(CppActionNames.C_COMPILE);
    assertThat(r.get("cpp_compile_action_name")).isEqualTo(CppActionNames.CPP_COMPILE);
    assertThat(r.get("linkstamp_compile_action_name")).isEqualTo(CppActionNames.LINKSTAMP_COMPILE);
    assertThat(r.get("cc_flags_make_variable_action_name_action_name"))
        .isEqualTo(CppActionNames.CC_FLAGS_MAKE_VARIABLE);
    assertThat(r.get("cpp_module_codegen_action_name"))
        .isEqualTo(CppActionNames.CPP_MODULE_CODEGEN);
    assertThat(r.get("cpp_header_parsing_action_name"))
        .isEqualTo(CppActionNames.CPP_HEADER_PARSING);
    assertThat(r.get("cpp_module_compile_action_name"))
        .isEqualTo(CppActionNames.CPP_MODULE_COMPILE);
    assertThat(r.get("assemble_action_name")).isEqualTo(CppActionNames.ASSEMBLE);
    assertThat(r.get("preprocess_assemble_action_name"))
        .isEqualTo(CppActionNames.PREPROCESS_ASSEMBLE);
    assertThat(r.get("lto_indexing_action_name")).isEqualTo(CppActionNames.LTO_INDEXING);
    assertThat(r.get("lto_backend_action_name")).isEqualTo(CppActionNames.LTO_BACKEND);
    assertThat(r.get("cpp_link_executable_action_name"))
        .isEqualTo(CppActionNames.CPP_LINK_EXECUTABLE);
    assertThat(r.get("cpp_link_dynamic_library_action_name"))
        .isEqualTo(CppActionNames.CPP_LINK_DYNAMIC_LIBRARY);
    assertThat(r.get("cpp_link_nodeps_dynamic_library_action_name"))
        .isEqualTo(CppActionNames.CPP_LINK_NODEPS_DYNAMIC_LIBRARY);
    assertThat(r.get("cpp_link_static_library_action_name"))
        .isEqualTo(CppActionNames.CPP_LINK_STATIC_LIBRARY);
    assertThat(r.get("strip_action_name")).isEqualTo(CppActionNames.STRIP);
  }

  @Test
  public void testEmptyCompileBuildVariables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "compiler_flag: '-foo'", "cxx_flag: '-foo_for_cxx_only'");
    useConfiguration();
    SkylarkList<String> commandLine =
        commandLineForVariables(
            CppActionNames.CPP_COMPILE,
            "cc_common.create_compile_variables(",
            "feature_configuration = feature_configuration,",
            "cc_toolchain = toolchain,",
            ")");
    assertThat(commandLine).contains("-foo");
    assertThat(commandLine).doesNotContain("-foo_for_cxx_only");
  }

  @Test
  public void testEmptyCompileBuildVariablesForCxx() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "compiler_flag: '-foo'", "cxx_flag: '-foo_for_cxx_only'");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "add_legacy_cxx_options = True",
                ")"))
        .containsAllOf("-foo", "-foo_for_cxx_only")
        .inOrder();
  }

  @Test
  public void testCompileBuildVariablesWithSourceFile() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "source_file = 'foo/bar/hello'",
                ")"))
        .containsAllOf("-c", "foo/bar/hello")
        .inOrder();
  }

  @Test
  public void testCompileBuildVariablesWithOutputFile() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "output_file = 'foo/bar/hello.o'",
                ")"))
        .containsAllOf("-o", "foo/bar/hello.o")
        .inOrder();
  }

  @Test
  public void testCompileBuildVariablesForIncludes() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "include_directories = depset(['foo/bar/include'])",
                ")"))
        .contains("-Ifoo/bar/include");
  }

  @Test
  public void testCompileBuildVariablesForDefines() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "preprocessor_defines = depset(['DEBUG_FOO'])",
                ")"))
        .contains("-DDEBUG_FOO");
  }

  @Test
  public void testCompileBuildVariablesForPic() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "use_pic = True",
                ")"))
        .contains("-fPIC");
  }

  @Test
  public void testUserCompileFlags() throws Exception {
    useConfiguration("--noincompatible_disable_depset_in_cc_user_flags");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "user_compile_flags = depset(['-foo'])",
                ")"))
        .contains("-foo");
  }

  @Test
  public void testUserCompileFlagsAsList() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "user_compile_flags = ['-foo']",
                ")"))
        .contains("-foo");
  }

  @Test
  public void testUserCompileFlagsAsDepsetWhenDisabled() throws Exception {
    useConfiguration("--incompatible_disable_depset_in_cc_user_flags");
    reporter.removeHandler(failFastHandler);
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_COMPILE,
                "cc_common.create_compile_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "user_compile_flags = depset(['-foo'])",
                ")"))
        .isNull();
    assertContainsEvent("Passing depset into user flags is deprecated");
  }

  @Test
  public void testEmptyLinkVariables() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "user_link_flags = [ '-foo' ],",
                ")"))
        .contains("-foo");
  }

  @Test
  public void testEmptyLinkVariablesContainSysroot() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "builtin_sysroot: '/foo/bar/sysroot'",
            "feature {",
            "  name: 'sysroot'",
            "  enabled: true",
            "  flag_set {",
            "    action: 'c++-link-executable'",
            "    flag_group {",
            "      expand_if_all_available: 'sysroot'",
            "      flag: '--yolo_sysroot_flag=%{sysroot}'",
            "    }",
            "  }",
            "}");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                ")"))
        .contains("--yolo_sysroot_flag=/foo/bar/sysroot");
  }

  @Test
  public void testLibrarySearchDirectoriesLinkVariables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "feature {",
            "  name: 'library_search_directories'",
            "  enabled: true",
            "  flag_set {",
            "    action: 'c++-link-executable'",
            "    flag_group {",
            "      expand_if_all_available: 'library_search_directories'",
            "      iterate_over: 'library_search_directories'",
            "      flag: '--library=%{library_search_directories}'",
            "    }",
            "  }",
            "}");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "library_search_directories = depset([ 'a', 'b', 'c' ]),",
                ")"))
        .containsAllOf("--library=a", "--library=b", "--library=c")
        .inOrder();
  }

  @Test
  public void testRuntimeLibrarySearchDirectoriesLinkVariables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "feature {",
            "  name: 'runtime_library_search_directories'",
            "  enabled: true",
            "  flag_set {",
            "    action: 'c++-link-executable'",
            "    flag_group {",
            "      expand_if_all_available: 'runtime_library_search_directories'",
            "      iterate_over: 'runtime_library_search_directories'",
            "      flag: '--runtime_library=%{runtime_library_search_directories}'",
            "    }",
            "  }",
            "}");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "runtime_library_search_directories = depset([ 'a', 'b', 'c' ]),",
                ")"))
        .containsAllOf("--runtime_library=a", "--runtime_library=b", "--runtime_library=c")
        .inOrder();
  }

  @Test
  public void testUserLinkFlagsLinkVariables() throws Exception {
    useConfiguration("--noincompatible_disable_depset_in_cc_user_flags");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "user_link_flags = depset([ '-avocado' ]),",
                ")"))
        .contains("-avocado");
  }

  @Test
  public void testUserLinkFlagsLinkVariablesAsList() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "user_link_flags = [ '-avocado' ],",
                ")"))
        .contains("-avocado");
  }

  @Test
  public void testUserLinkFlagsLinkVariablesAsDepsetWhenDisabled() throws Exception {
    useConfiguration("--incompatible_disable_depset_in_cc_user_flags");
    reporter.removeHandler(failFastHandler);
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "user_link_flags = depset([ '-avocado' ]),",
                ")"))
        .isNull();
    assertContainsEvent("Passing depset into user flags is deprecated");
  }

  @Test
  public void testIfsoRelatedVariablesAreNotExposed() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "feature {",
            "  name: 'uses_ifso_variables'",
            "  enabled: true",
            "  flag_set {",
            "    action: 'c++-link-dynamic-library'",
            "    flag_group {",
            "      expand_if_all_available: 'generate_interface_library'",
            "      flag: '--generate_interface_library_was_available'",
            "    }",
            "  }",
            "}");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_DYNAMIC_LIBRARY,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                ")"))
        .doesNotContain("--generate_interface_library_was_available");
  }

  @Test
  public void testOutputFileLinkVariables() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "output_file = 'foo/bar/executable',",
                ")"))
        .contains("foo/bar/executable");
  }

  @Test
  public void testParamFileLinkVariables() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "param_file = 'foo/bar/params',",
                ")"))
        .contains("-Wl,@foo/bar/params");
  }

  @Test
  public void testDefFileLinkVariables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "feature {",
            "  name: 'def'",
            "  enabled: true",
            "  flag_set {",
            "    action: 'c++-link-executable'",
            "    flag_group {",
            "      expand_if_all_available: 'def_file_path'",
            "      flag: '-qux_%{def_file_path}'",
            "    }",
            "  }",
            "}");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "def_file = 'foo/bar/def',",
                ")"))
        .contains("-qux_foo/bar/def");
  }

  @Test
  public void testMustKeepDebugLinkVariables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "feature {",
            "  name: 'strip_debug_symbols'",
            "  enabled: true",
            "  flag_set {",
            "    action: 'c++-link-executable'",
            "    flag_group {",
            "      expand_if_all_available: 'strip_debug_symbols'",
            "      flag: '-strip_stuff'",
            "    }",
            "  }",
            "}");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                0,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "must_keep_debug = False,",
                ")"))
        .contains("-strip_stuff");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                1,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "must_keep_debug = True,",
                ")"))
        .doesNotContain("-strip_stuff");
  }

  @Test
  public void testIsLinkingDynamicLibraryLinkVariables() throws Exception {
    useConfiguration("--linkopt=-pie");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                0,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "is_linking_dynamic_library = True,",
                "user_link_flags = [ '-pie' ],",
                ")"))
        .doesNotContain("-pie");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                1,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "is_linking_dynamic_library = False,",
                "user_link_flags = [ '-pie' ],",
                ")"))
        .contains("-pie");
  }

  @Test
  public void testIsUsingLinkerLinkVariables() throws Exception {
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                0,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "is_using_linker = True,",
                "user_link_flags = [ '-i_dont_want_to_see_this_on_archiver_command_line' ],",
                ")"))
        .contains("-i_dont_want_to_see_this_on_archiver_command_line");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                1,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "is_using_linker = False,",
                "user_link_flags = [ '-i_dont_want_to_see_this_on_archiver_command_line' ],",
                ")"))
        .doesNotContain("-i_dont_want_to_see_this_on_archiver_command_line");
  }

  @Test
  public void testUseTestOnlyFlagsLinkVariables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(mockToolsConfig, "test_only_linker_flag: '-im_only_testing_flag'");
    useConfiguration();
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                0,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "use_test_only_flags = False,",
                ")"))
        .doesNotContain("-im_only_testing_flag");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                1,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "use_test_only_flags = True,",
                ")"))
        .contains("-im_only_testing_flag");
  }

  @Test
  public void testIsStaticLinkingModeLinkVariables() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "linking_mode_flags {",
            "  mode: MOSTLY_STATIC",
            "  linker_flag: '-static_linking_mode_flag'",
            "}",
            "linking_mode_flags {",
            "  mode: DYNAMIC",
            "  linker_flag: '-dynamic_linking_mode_flag'",
            "}");
    useConfiguration();
    SkylarkList<String> staticLinkingModeFlags =
        commandLineForVariables(
            CppActionNames.CPP_LINK_EXECUTABLE,
            0,
            "cc_common.create_link_variables(",
            "feature_configuration = feature_configuration,",
            "cc_toolchain = toolchain,",
            "is_static_linking_mode = True,",
            ")");
    assertThat(staticLinkingModeFlags).contains("-static_linking_mode_flag");
    assertThat(staticLinkingModeFlags).doesNotContain("-dynamic_linking_mode_flag");

    SkylarkList<String> dynamicLinkingModeFlags =
        commandLineForVariables(
            CppActionNames.CPP_LINK_EXECUTABLE,
            1,
            "cc_common.create_link_variables(",
            "feature_configuration = feature_configuration,",
            "cc_toolchain = toolchain,",
            "is_static_linking_mode = False,",
            ")");
    assertThat(dynamicLinkingModeFlags).doesNotContain("-static_linking_mode_flag");
    assertThat(dynamicLinkingModeFlags).contains("-dynamic_linking_mode_flag");
  }

  private SkylarkList<String> commandLineForVariables(String actionName, String... variables)
      throws Exception {
    return commandLineForVariables(actionName, 0, variables);
  }

  // This method is only there to change the package to fix multiple runs of this method in a single
  // test.
  // TODO(b/109917616): Remove pkgSuffix argument when bzl files are not cached within single test
  private SkylarkList<String> commandLineForVariables(
      String actionName, int pkgSuffix, String... variables) throws Exception {
    scratch.file(
        "a" + pkgSuffix + "/BUILD",
        "load(':rule.bzl', 'crule')",
        "cc_toolchain_alias(name='alias')",
        "crule(name='r')");

    scratch.file(
        "a" + pkgSuffix + "/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  variables = " + Joiner.on("\n").join(variables),
        "  return struct(",
        "    command_line = cc_common.get_memory_inefficient_command_line(",
        "        feature_configuration = feature_configuration,",
        "        action_name = '" + actionName + "',",
        "        variables = variables))",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a" + pkgSuffix + ":alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    /** Calling {@link #getTarget} to get loading errors */
    getTarget("//a" + pkgSuffix + ":r");
    ConfiguredTarget r = getConfiguredTarget("//a" + pkgSuffix + ":r");
    if (r == null) {
      return null;
    }
    @SuppressWarnings("unchecked")
    SkylarkList<String> result = (SkylarkList<String>) r.get("command_line");
    return result;
  }

  @Test
  public void testCcCompilationProvider() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//tools/build_defs/cc:rule.bzl', 'crule')",
        "licenses(['notice'])",
        "cc_library(",
        "    name='lib',",
        "    hdrs = ['lib.h'],",
        "    srcs = ['lib.cc'],",
        "    deps = ['r']",
        ")",
        "cc_library(",
        "    name = 'dep1',",
        "    srcs = ['dep1.cc'],",
        "    hdrs = ['dep1.h'],",
        "    includes = ['dep1/baz'],",
        /*"    include_prefix = 'dep1/include_prefix',",*/
        "    defines = ['DEP1'],",
        ")",
        "cc_library(",
        "    name = 'dep2',",
        "    srcs = ['dep2.cc'],",
        "    hdrs = ['dep2.h'],",
        "    includes = ['dep2/qux'],",
        /*"    include_prefix = 'dep2/include_prefix',",*/
        "    defines = ['DEP2'],",
        ")",
        "crule(name='r')");
    scratch.file("tools/build_defs/cc/BUILD", "");
    scratch.file(
        "tools/build_defs/cc/rule.bzl",
        "def _impl(ctx):",
        "  compilation_context = cc_common.create_compilation_context(",
        "    headers=depset([ctx.file._header]),",
        "    system_includes=depset([ctx.attr._system_include]),",
        "    includes=depset([ctx.attr._include]),",
        "    quote_includes=depset([ctx.attr._quote_include]),",
        "    defines=depset([ctx.attr._define]))",
        "  cc_infos = [CcInfo(compilation_context=compilation_context)]",
        "  for dep in ctx.attr._deps:",
        "      cc_infos.append(dep[CcInfo])",
        "  merged_cc_info=cc_common.merge_cc_infos(cc_infos=cc_infos)",
        "  return struct(",
        "    providers=[merged_cc_info],",
        "    merged_headers=merged_cc_info.compilation_context.headers,",
        "    merged_system_includes=merged_cc_info.compilation_context.system_includes,",
        "    merged_includes=merged_cc_info.compilation_context.includes,",
        "    merged_quote_includes=merged_cc_info.compilation_context.quote_includes,",
        "    merged_defines=merged_cc_info.compilation_context.defines",
        "  )",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_header': attr.label(allow_single_file=True,",
        "        default=Label('//a:header.h')),",
        "    '_system_include': attr.string(default='foo/bar'),",
        "    '_include': attr.string(default='baz/qux'),",
        "    '_quote_include': attr.string(default='quux/abc'),",
        "    '_define': attr.string(default='MYDEFINE'),",
        "    '_deps': attr.label_list(default=['//a:dep1', '//a:dep2'])",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget lib = getConfiguredTarget("//a:lib");
    @SuppressWarnings("unchecked")
    CcCompilationContext ccCompilationContext = lib.get(CcInfo.PROVIDER).getCcCompilationContext();
    assertThat(
            ccCompilationContext.getDeclaredIncludeSrcs().toCollection().stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .containsExactly("lib.h", "header.h", "dep1.h", "dep2.h");

    ConfiguredTarget r = getConfiguredTarget("//a:r");

    List<Artifact> mergedHeaders =
        ((SkylarkNestedSet) r.get("merged_headers")).getSet(Artifact.class).toList();
    assertThat(
            mergedHeaders.stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("header.h", "dep1.h", "dep2.h");

    List<String> mergedDefines =
        ((SkylarkNestedSet) r.get("merged_defines")).getSet(String.class).toList();
    assertThat(mergedDefines).containsAllOf("MYDEFINE", "DEP1", "DEP2");

    List<String> mergedSystemIncludes =
        ((SkylarkNestedSet) r.get("merged_system_includes")).getSet(String.class).toList();
    assertThat(mergedSystemIncludes).containsAllOf("foo/bar", "a/dep1/baz", "a/dep2/qux");

    List<String> mergedIncludes =
        ((SkylarkNestedSet) r.get("merged_includes")).getSet(String.class).toList();
    assertThat(mergedIncludes).contains("baz/qux");

    List<String> mergedQuoteIncludes =
        ((SkylarkNestedSet) r.get("merged_quote_includes")).getSet(String.class).toList();
    assertThat(mergedQuoteIncludes).contains("quux/abc");
  }

  @Test
  public void testCcCompilationProviderDefaultValues() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//tools/build_defs/cc:rule.bzl', 'crule')",
        "licenses(['notice'])",
        "crule(name='r')");
    scratch.file("tools/build_defs/cc/BUILD", "");
    scratch.file(
        "tools/build_defs/cc/rule.bzl",
        "def _impl(ctx):",
        "  compilation_context = cc_common.create_compilation_context()",
        "  return struct()",
        "crule = rule(",
        "  _impl,",
        "  fragments = ['cpp'],",
        ");");

    assertThat(getConfiguredTarget("//a:r")).isNotNull();
  }

  @Test
  public void testCcCompilationProviderInvalidValues() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "a/BUILD",
        "load('//tools/build_defs/cc:rule.bzl', 'crule')",
        "licenses(['notice'])",
        "crule(name='r')");
    scratch.file("tools/build_defs/cc/BUILD", "");
    scratch.file(
        "tools/build_defs/cc/rule.bzl",
        "def _impl(ctx):",
        "  compilation_context = cc_common.create_compilation_context(headers=[])",
        "  return struct()",
        "crule = rule(",
        "  _impl,",
        "  fragments = ['cpp'],",
        ");");

    getConfiguredTarget("//a:r");
    assertContainsEvent("'headers' argument must be a depset");
  }

  @Test
  public void testFlagWhitelist() throws Exception {
    setSkylarkSemanticsOptions("--experimental_cc_skylark_api_enabled_packages=\"\"");
    SkylarkCcCommonTestHelper.createFiles(scratch, "foo/bar");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:bin");
    assertContainsEvent(
        "You can try it out by passing "
            + "--experimental_cc_skylark_api_enabled_packages=<list of packages>. Beware that we "
            + "will be making breaking changes to this API without prior warning.");
  }

  @Test
  public void testCcLinkingContextOnWindows() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.COPY_DYNAMIC_LIBRARIES_TO_BINARY_CONFIGURATION,
            MockCcSupport.TARGETS_WINDOWS_CONFIGURATION,
            "supports_interface_shared_objects: false",
            "needsPic: false");
    doTestCcLinkingContext(
        ImmutableList.of("a.a", "libdep2.a", "b.a", "c.a", "d.a", "libdep1.a"),
        ImmutableList.of("a.pic.a", "b.pic.a", "c.pic.a", "e.pic.a"),
        ImmutableList.of("a.so", "libdep2.so", "b.so", "e.so", "libdep1.so"));
  }

  @Test
  public void testCcLinkingContext() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            MockCcSupport.PIC_FEATURE,
            "supports_interface_shared_objects: false",
            "needsPic: true");
    doTestCcLinkingContext(
        ImmutableList.of("a.a", "b.a", "c.a", "d.a"),
        ImmutableList.of("a.pic.a", "libdep2.a", "b.pic.a", "c.pic.a", "e.pic.a", "libdep1.a"),
        ImmutableList.of("a.so", "liba_Slibdep2.so", "b.so", "e.so", "liba_Slibdep1.so"));
  }

  private void doTestCcLinkingContext(
      List<String> staticLibraryList,
      List<String> picStaticLibraryList,
      List<String> dynamicLibraryList)
      throws Exception {
    useConfiguration();
    setUpCcLinkingContextTest();
    ConfiguredTarget a = getConfiguredTarget("//a:a");

    StructImpl info = ((StructImpl) a.get("info"));
    @SuppressWarnings("unchecked")
    SkylarkList<String> userLinkFlags =
        (SkylarkList<String>) info.getValue("user_link_flags", SkylarkList.class);
    assertThat(userLinkFlags.getImmutableList())
        .containsExactly("-la", "-lc2", "-DEP2_LINKOPT", "-lc1", "-lc2", "-DEP1_LINKOPT");
    @SuppressWarnings("unchecked")
    SkylarkList<LibraryToLinkWrapper> librariesToLink =
        (SkylarkList<LibraryToLinkWrapper>) info.getValue("libraries_to_link", SkylarkList.class);
    assertThat(
            librariesToLink.stream()
                .filter(x -> x.getStaticLibrary() != null)
                .map(x -> x.getStaticLibrary().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsExactlyElementsIn(staticLibraryList);
    assertThat(
            librariesToLink.stream()
                .filter(x -> x.getPicStaticLibrary() != null)
                .map(x -> x.getPicStaticLibrary().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsExactlyElementsIn(picStaticLibraryList);
    assertThat(
            librariesToLink.stream()
                .filter(x -> x.getDynamicLibrary() != null)
                .map(x -> x.getDynamicLibrary().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsExactlyElementsIn(dynamicLibraryList);
    assertThat(
            librariesToLink.stream()
                .filter(x -> x.getInterfaceLibrary() != null)
                .map(x -> x.getInterfaceLibrary().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsExactly("a.ifso");
    Artifact staticLibrary = info.getValue("static_library", Artifact.class);
    assertThat(staticLibrary.getFilename()).isEqualTo("a.a");
    Artifact picStaticLibrary = info.getValue("pic_static_library", Artifact.class);
    assertThat(picStaticLibrary.getFilename()).isEqualTo("a.pic.a");
    Artifact dynamicLibrary = info.getValue("dynamic_library", Artifact.class);
    assertThat(dynamicLibrary.getFilename()).isEqualTo("a.so");
    Artifact interfaceLibrary = info.getValue("interface_library", Artifact.class);
    assertThat(interfaceLibrary.getFilename()).isEqualTo("a.ifso");
    boolean alwayslink = info.getValue("alwayslink", Boolean.class);
    assertThat(alwayslink).isTrue();

    ConfiguredTarget bin = getConfiguredTarget("//a:bin");
    assertThat(bin).isNotNull();
  }

  private void setUpCcLinkingContextTest() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//tools/build_defs/cc:rule.bzl', 'crule')",
        "cc_binary(name='bin',",
        "   deps = [':a'],",
        ")",
        "crule(name='a',",
        "   static_library = 'a.a',",
        "   pic_static_library = 'a.pic.a',",
        "   dynamic_library = 'a.so',",
        "   interface_library = 'a.ifso',",
        "   user_link_flags = ['-la', '-lc2'],",
        "   alwayslink = True,",
        "   deps = [':c', ':dep2', ':b'],",
        ")",
        "crule(name='b',",
        "   static_library = 'b.a',",
        "   pic_static_library = 'b.pic.a',",
        "   dynamic_library = 'b.so',",
        "   deps = [':c', ':d'],",
        ")",
        "crule(name='c',",
        "   static_library = 'c.a',",
        "   pic_static_library = 'c.pic.a',",
        "   user_link_flags = ['-lc1', '-lc2'],",
        ")",
        "crule(name='d',",
        "   static_library = 'd.a',",
        "   alwayslink = True,",
        "   deps = [':e'],",
        ")",
        "crule(name='e',",
        "   pic_static_library = 'e.pic.a',",
        "   dynamic_library = 'e.so',",
        "   deps = [':dep1'],",
        ")",
        "cc_toolchain_alias(name='alias')",
        "cc_library(",
        "    name = 'dep1',",
        "    srcs = ['dep1.cc'],",
        "    hdrs = ['dep1.h'],",
        "    linkopts = ['-DEP1_LINKOPT'],",
        ")",
        "cc_library(",
        "    name = 'dep2',",
        "    srcs = ['dep2.cc'],",
        "    hdrs = ['dep2.h'],",
        "    linkopts = ['-DEP2_LINKOPT'],",
        ")");
    scratch.file("a/lib.a", "");
    scratch.file("a/lib.so", "");
    scratch.file("tools/build_defs/cc/BUILD", "");
    scratch.file(
        "tools/build_defs/cc/rule.bzl",
        "top_linking_context_smoke = cc_common.create_linking_context(libraries_to_link=[],",
        "   user_link_flags=['-first_flag', '-second_flag'])",
        "def _create(ctx, feature_configuration, static_library, pic_static_library,",
        "  dynamic_library, interface_library, alwayslink):",
        "  return cc_common.create_library_to_link(",
        "    actions=ctx.actions, feature_configuration=feature_configuration, ",
        "    cc_toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo], ",
        "    static_library=static_library, pic_static_library=pic_static_library,",
        "    dynamic_library=dynamic_library, interface_library=interface_library,",
        "    alwayslink=alwayslink)",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  feature_configuration = cc_common.configure_features(cc_toolchain = toolchain)",
        "  library_to_link = _create(ctx, feature_configuration, ctx.file.static_library, ",
        "     ctx.file.pic_static_library, ctx.file.dynamic_library, ctx.file.interface_library,",
        "     ctx.attr.alwayslink)",
        "  linking_context = cc_common.create_linking_context(libraries_to_link=[library_to_link],",
        "     user_link_flags=ctx.attr.user_link_flags)",
        "  cc_infos = [CcInfo(linking_context=linking_context)]",
        "  for dep in ctx.attr.deps:",
        "      cc_infos.append(dep[CcInfo])",
        "  merged_cc_info = cc_common.merge_cc_infos(cc_infos=cc_infos)",
        "  return struct(",
        "     info = struct(",
        "       cc_info = merged_cc_info,",
        "       user_link_flags = merged_cc_info.linking_context.user_link_flags,",
        "       libraries_to_link = merged_cc_info.linking_context.libraries_to_link,",
        "       static_library = library_to_link.static_library,",
        "       pic_static_library = library_to_link.pic_static_library,",
        "       dynamic_library = library_to_link.dynamic_library,",
        "       interface_library = library_to_link.interface_library,",
        "       alwayslink = library_to_link.alwayslink),",
        "     providers = [merged_cc_info]",
        "  )",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    'user_link_flags' : attr.string_list(),",
        "    'static_library': attr.label(allow_single_file=True),",
        "    'pic_static_library': attr.label(allow_single_file=True),",
        "    'dynamic_library': attr.label(allow_single_file=True),",
        "    'interface_library': attr.label(allow_single_file=True),",
        "    'alwayslink': attr.bool(),",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias')),",
        "    'deps': attr.label_list(),",
        "  },",
        "  fragments = ['cpp'],",
        ");");
  }

  @Test
  public void testCcNativeRuleDependingOnSkylarkDefinedRule() throws Exception {
    SkylarkCcCommonTestHelper.createFiles(scratch, "tools/build_defs/cc");
    assertThat(getConfiguredTarget("//foo:bin")).isNotNull();
  }

  @Test
  public void testCopts() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingCompilation(
        scratch, "tools/build_defs/foo", "copts=depset(['-COMPILATION_OPTION'])");
    assertThat(getConfiguredTarget("//foo:bin")).isNotNull();
    ConfiguredTarget target = getConfiguredTarget("//foo:skylark_lib");
    CppCompileAction action =
        (CppCompileAction) getGeneratingAction(artifactByPath(getFilesToBuild(target), ".o"));
    assertThat(action.getArguments()).contains("-COMPILATION_OPTION");
  }

  @Test
  public void testIncludeDirs() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingCompilation(
        scratch, "tools/build_defs/foo", "includes=depset(['foo/bar', 'baz/qux'])");
    ConfiguredTarget target = getConfiguredTarget("//foo:skylark_lib");
    assertThat(target).isNotNull();
    CppCompileAction action =
        (CppCompileAction) getGeneratingAction(artifactByPath(getFilesToBuild(target), ".o"));
    assertThat(action.getArguments()).containsAllOf("-Ifoo/bar", "-Ibaz/qux");
  }

  @Test
  public void testLinkingOutputs() throws Exception {
    SkylarkCcCommonTestHelper.createFiles(scratch, "tools/build_defs/foo");
    ConfiguredTarget target = getConfiguredTarget("//foo:skylark_lib");
    assertThat(target).isNotNull();
    @SuppressWarnings("unchecked")
    SkylarkList<LibraryToLinkWrapper> libraries =
        (SkylarkList<LibraryToLinkWrapper>) target.get("libraries");
    assertThat(
            libraries.stream()
                .map(x -> x.getResolvedSymlinkDynamicLibrary().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .contains("libskylark_lib.so");
  }

  @Test
  public void testLinkopts() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingLinking(
        scratch, "tools/build_defs/foo", "linkopts=depset(['-LINKING_OPTION'])");
    ConfiguredTarget target = getConfiguredTarget("//foo:skylark_lib");
    assertThat(target).isNotNull();
    assertThat(target.get(CcInfo.PROVIDER).getCcLinkingContext().getFlattenedUserLinkFlags())
        .contains("-LINKING_OPTION");
  }

  @Test
  public void testSettingDynamicLibraryArtifact() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingLinking(
        scratch,
        "tools/build_defs/foo",
        "dynamic_library=ctx.actions.declare_file('dynamic_lib_artifact.so')");
    assertThat(getConfiguredTarget("//foo:skylark_lib")).isNotNull();
    ConfiguredTarget target = getConfiguredTarget("//foo:skylark_lib");
    @SuppressWarnings("unchecked")
    SkylarkList<LibraryToLinkWrapper> libraries =
        (SkylarkList<LibraryToLinkWrapper>) target.get("libraries");
    assertThat(
            libraries.stream()
                .map(x -> x.getResolvedSymlinkDynamicLibrary().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .contains("dynamic_lib_artifact.so");
  }

  @Test
  public void testCcLinkingContexts() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingLinking(
        scratch, "tools/build_defs/foo", "linking_contexts=dep_linking_contexts");
    assertThat(getConfiguredTarget("//foo:bin")).isNotNull();
    ConfiguredTarget target = getConfiguredTarget("//foo:bin");
    CppLinkAction action =
        (CppLinkAction) getGeneratingAction(artifactByPath(getFilesToBuild(target), "bin"));
    assertThat(action.getArguments()).containsAllOf("-DEP1_LINKOPT", "-DEP2_LINKOPT");
  }

  @Test
  public void testNeverlinkTrue() throws Exception {
    assertThat(setUpNeverlinkTest("True").getArguments()).doesNotContain("-NEVERLINK_OPTION");
  }

  @Test
  public void testNeverlinkFalse() throws Exception {
    assertThat(setUpNeverlinkTest("False").getArguments()).contains("-NEVERLINK_OPTION");
  }

  private CppLinkAction setUpNeverlinkTest(String value) throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingLinking(
        scratch,
        "tools/build_defs/foo",
        String.join(",", "linkopts=depset(['-NEVERLINK_OPTION'])", "neverlink=" + value));
    assertThat(getConfiguredTarget("//foo:bin")).isNotNull();
    ConfiguredTarget target = getConfiguredTarget("//foo:bin");
    return (CppLinkAction) getGeneratingAction(artifactByPath(getFilesToBuild(target), "bin"));
  }

  private void loadCcToolchainConfigLib() throws IOException {
    scratch.appendFile("tools/cpp/BUILD", "");
    scratch.file(
        "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.BAZEL_REPO_PATH + "tools/cpp/cc_toolchain_config_lib.bzl"));
  }

  @Test
  public void testVariableWithValue() throws Exception {
    loadCcToolchainConfigLib();
    createVariableWithValueRule("one", /* name= */ "None", /* value= */ "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("name parameter of variable_with_value should be a string, found NoneType");

    createVariableWithValueRule("two", /* name= */ "'abc'", /* value= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("value parameter of variable_with_value should be a string, found NoneType");

    createVariableWithValueRule("three", /* name= */ "''", /* value= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("name parameter of variable_with_value must be a nonempty string");

    createVariableWithValueRule("four", /* name= */ "'abc'", /* value= */ "''");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//four:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("value parameter of variable_with_value must be a nonempty string");

    createVariableWithValueRule("five", /* name= */ "'abc'", /* value= */ "'def'");

    ConfiguredTarget t = getConfiguredTarget("//five:a");
    SkylarkInfo variable = (SkylarkInfo) t.get("variable");
    assertThat(variable).isNotNull();
    VariableWithValue v = CcModule.variableWithValueFromSkylark(variable);
    assertThat(v).isNotNull();
    assertThat(v.variable).isEqualTo("abc");
    assertThat(v.value).isEqualTo("def");

    createEnvEntryRule("six", /* key= */ "'abc'", /* value= */ "'def'");
    t = getConfiguredTarget("//six:a");
    SkylarkInfo envEntry = (SkylarkInfo) t.get("entry");
    try {
      CcModule.variableWithValueFromSkylark(envEntry);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'variable_with_value', received 'env_entry");
    }
  }

  private void createVariableWithValueRule(String pkg, String name, String value)
      throws IOException {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'variable_with_value')",
        "def _impl(ctx):",
        "   return struct(variable = variable_with_value(",
        "       name = " + name + ",",
        "       value = " + value + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomVariableWithValue() throws Exception {
    loadCcToolchainConfigLib();
    createCustomVariableWithValueRule("one", /* name= */ "None", /* value= */ "None");
    ConfiguredTarget t = getConfiguredTarget("//one:a");
    SkylarkInfo variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.variableWithValueFromSkylark(variable);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'name' parameter of variable_with_value must be a nonempty string.");
    }

    createCustomVariableWithValueRule("two", /* name= */ "'abc'", /* value= */ "None");

    t = getConfiguredTarget("//two:a");
    variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.variableWithValueFromSkylark(variable);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'value' parameter of variable_with_value must be a nonempty string.");
    }

    createCustomVariableWithValueRule("three", /* name= */ "'abc'", /* value= */ "struct()");

    t = getConfiguredTarget("//three:a");
    variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.variableWithValueFromSkylark(variable);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'value' is not of 'java.lang.String' type.");
    }

    createCustomVariableWithValueRule("four", /* name= */ "True", /* value= */ "'abc'");

    t = getConfiguredTarget("//four:a");
    variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.variableWithValueFromSkylark(variable);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'name' is not of 'java.lang.String' type.");
    }
  }

  private void createCustomVariableWithValueRule(String pkg, String name, String value)
      throws IOException {
    scratch.file(
        pkg + "/foo.bzl",
        "def _impl(ctx):",
        "   return struct(variable = struct(",
        "       name = " + name + ",",
        "       value = " + value + ",",
        "       type_name = 'variable_with_value'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testEnvEntry() throws Exception {
    loadCcToolchainConfigLib();
    createEnvEntryRule("one", "None", /* value= */ "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("key parameter of env_entry should be a string, found NoneType");

    createEnvEntryRule("two", "'abc'", /* value= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("value parameter of env_entry should be a string, found NoneType");

    createEnvEntryRule("three", "''", /* value= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e).hasMessageThat().contains("key parameter of env_entry must be a nonempty string");

    createEnvEntryRule("four", "'abc'", /* value= */ "''");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//four:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("value parameter of env_entry must be a nonempty string");

    createEnvEntryRule("five", "'abc'", /* value= */ "'def'");

    ConfiguredTarget t = getConfiguredTarget("//five:a");
    SkylarkInfo entryProvider = (SkylarkInfo) t.get("entry");
    assertThat(entryProvider).isNotNull();
    EnvEntry entry = CcModule.envEntryFromSkylark(entryProvider);
    assertThat(entry).isNotNull();
    StringValueParser parser = new StringValueParser("def");
    assertThat(entry).isEqualTo(new EnvEntry("abc", parser.getChunks()));

    createVariableWithValueRule("six", /* name= */ "'abc'", /* value= */ "'def'");
    t = getConfiguredTarget("//six:a");
    SkylarkInfo variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.envEntryFromSkylark(variable);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'env_entry', received 'variable_with_value");
    }
  }

  private void createEnvEntryRule(String pkg, String key, String value) throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'env_entry')",
        "def _impl(ctx):",
        "   return struct(entry = env_entry(",
        "       key = " + key + ",",
        "       value = " + value + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomEnvEntry() throws Exception {
    loadCcToolchainConfigLib();

    createCustomEnvEntryRule("one", /* key= */ "None", /* value= */ "None");

    ConfiguredTarget t = getConfiguredTarget("//one:a");
    SkylarkInfo entry = (SkylarkInfo) t.get("entry");
    try {
      CcModule.envEntryFromSkylark(entry);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'key' parameter of env_entry must be a nonempty string.");
    }

    createCustomEnvEntryRule("two", /* key= */ "'abc'", /* value= */ "None");

    t = getConfiguredTarget("//two:a");
    entry = (SkylarkInfo) t.get("entry");
    try {
      CcModule.envEntryFromSkylark(entry);
      fail("Should have failed because of empty string.");
      ;
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'value' parameter of env_entry must be a nonempty string.");
    }

    createCustomEnvEntryRule("three", /* key= */ "'abc'", /* value= */ "struct()");

    t = getConfiguredTarget("//three:a");
    entry = (SkylarkInfo) t.get("entry");
    try {
      CcModule.envEntryFromSkylark(entry);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'value' is not of 'java.lang.String' type.");
    }

    createCustomEnvEntryRule("four", /* key= */ "True", /* value= */ "'abc'");

    t = getConfiguredTarget("//four:a");
    entry = (SkylarkInfo) t.get("entry");
    try {
      CcModule.envEntryFromSkylark(entry);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'key' is not of 'java.lang.String' type.");
    }
  }

  private void createCustomEnvEntryRule(String pkg, String key, String value) throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "def _impl(ctx):",
        "   return struct(entry = struct(",
        "       key = " + key + ",",
        "       value = " + value + ",",
        "       type_name = 'env_entry'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testToolPath() throws Exception {
    loadCcToolchainConfigLib();
    createToolPathRule("one", /* name= */ "None", "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("name parameter of tool_path should be a string, found NoneType");

    createToolPathRule("two", /* name= */ "'abc'", "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("path parameter of tool_path should be a string, found NoneType");

    createToolPathRule("three", /* name= */ "''", "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("name parameter of tool_path must be a nonempty string");

    createToolPathRule("four", /* name= */ "'abc'", "''");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//four:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("path parameter of tool_path must be a nonempty string");

    createToolPathRule("five", /* name= */ "'abc'", "'/d/e/f'");

    ConfiguredTarget t = getConfiguredTarget("//five:a");
    SkylarkInfo toolPathProvider = (SkylarkInfo) t.get("toolpath");
    assertThat(toolPathProvider).isNotNull();
    Pair<String, String> toolPath = CcModule.toolPathFromSkylark(toolPathProvider);
    assertThat(toolPath).isNotNull();
    assertThat(toolPath.first).isEqualTo("abc");
    assertThat(toolPath.second).isEqualTo("/d/e/f");

    createVariableWithValueRule("six", /* name= */ "'abc'", /* value= */ "'def'");
    t = getConfiguredTarget("//six:a");
    SkylarkInfo variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.toolPathFromSkylark(variable);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'tool_path', received 'variable_with_value");
    }
  }

  private void createToolPathRule(String pkg, String name, String path) throws IOException {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'tool_path')",
        "def _impl(ctx):",
        "   return struct(toolpath = tool_path(",
        "       name = " + name + ",",
        "       path = " + path + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomToolPath() throws Exception {
    loadCcToolchainConfigLib();

    createCustomToolPathRule("one", /* name= */ "None", /* path= */ "None");

    ConfiguredTarget t = getConfiguredTarget("//one:a");
    SkylarkInfo toolPath = (SkylarkInfo) t.get("toolpath");
    try {
      CcModule.toolPathFromSkylark(toolPath);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'name' parameter of tool_path must be a nonempty string.");
    }

    createCustomToolPathRule("two", /* name= */ "'abc'", /* path= */ "None");

    t = getConfiguredTarget("//two:a");
    toolPath = (SkylarkInfo) t.get("toolpath");
    try {
      CcModule.toolPathFromSkylark(toolPath);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'path' parameter of tool_path must be a nonempty string.");
    }

    createCustomToolPathRule("three", /* name= */ "'abc'", /* path= */ "struct()");

    t = getConfiguredTarget("//three:a");
    toolPath = (SkylarkInfo) t.get("toolpath");
    try {
      CcModule.toolPathFromSkylark(toolPath);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'path' is not of 'java.lang.String' type.");
    }

    createCustomToolPathRule("four", /* name= */ "True", /* path= */ "'abc'");

    t = getConfiguredTarget("//four:a");
    toolPath = (SkylarkInfo) t.get("toolpath");
    try {
      CcModule.toolPathFromSkylark(toolPath);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'name' is not of 'java.lang.String' type.");
    }
  }

  private void createCustomToolPathRule(String pkg, String name, String path) throws IOException {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'tool_path')",
        "def _impl(ctx):",
        "   return struct(toolpath = struct(",
        "       name = " + name + ",",
        "       path = " + path + ",",
        "       type_name = 'tool_path'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testMakeVariable() throws Exception {
    loadCcToolchainConfigLib();
    createMakeVariablerule("one", /* name= */ "None", /* value= */ "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("name parameter of make_variable should be a string, found NoneType");

    createMakeVariablerule("two", /* name= */ "'abc'", /* value= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("value parameter of make_variable should be a string, found NoneType");

    createMakeVariablerule("three", /* name= */ "''", /* value= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("name parameter of make_variable must be a nonempty string");

    createMakeVariablerule("four", /* name= */ "'abc'", /* value= */ "''");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//four:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("value parameter of make_variable must be a nonempty string");

    createMakeVariablerule("five", /* name= */ "'abc'", /* value= */ "'val'");

    ConfiguredTarget t = getConfiguredTarget("//five:a");
    SkylarkInfo makeVariableProvider = (SkylarkInfo) t.get("variable");
    assertThat(makeVariableProvider).isNotNull();
    Pair<String, String> makeVariable = CcModule.makeVariableFromSkylark(makeVariableProvider);
    assertThat(makeVariable).isNotNull();
    assertThat(makeVariable.first).isEqualTo("abc");
    assertThat(makeVariable.second).isEqualTo("val");

    createVariableWithValueRule("six", /* name= */ "'abc'", /* value= */ "'def'");
    t = getConfiguredTarget("//six:a");
    SkylarkInfo variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.makeVariableFromSkylark(variable);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'make_variable', received 'variable_with_value");
    }
  }

  private void createMakeVariablerule(String pkg, String name, String value) throws IOException {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'make_variable')",
        "def _impl(ctx):",
        "   return struct(variable = make_variable(",
        "       name = " + name + ",",
        "       value = " + value + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomMakeVariable() throws Exception {
    createCustomMakeVariablerule("one", /* name= */ "None", /* value= */ "None");

    try {
      ConfiguredTarget t = getConfiguredTarget("//one:a");
      SkylarkInfo makeVariableProvider = (SkylarkInfo) t.get("variable");
      CcModule.makeVariableFromSkylark(makeVariableProvider);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'name' parameter of make_variable must be a nonempty string.");
    }

    createCustomMakeVariablerule("two", /* name= */ "'abc'", /* value= */ "None");

    try {
      ConfiguredTarget t = getConfiguredTarget("//two:a");
      SkylarkInfo makeVariableProvider = (SkylarkInfo) t.get("variable");
      CcModule.makeVariableFromSkylark(makeVariableProvider);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'value' parameter of make_variable must be a nonempty string.");
    }

    createCustomMakeVariablerule("three", /* name= */ "[]", /* value= */ "None");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo makeVariableProvider = (SkylarkInfo) t.get("variable");
      CcModule.makeVariableFromSkylark(makeVariableProvider);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'name' is not of 'java.lang.String' type.");
    }

    createCustomMakeVariablerule("four", /* name= */ "'abc'", /* value= */ "True");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo makeVariableProvider = (SkylarkInfo) t.get("variable");
      CcModule.makeVariableFromSkylark(makeVariableProvider);
      fail("Should have failed because of empty string.");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains("Field 'value' is not of 'java.lang.String' type.");
    }
  }

  private void createCustomMakeVariablerule(String pkg, String name, String value)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "def _impl(ctx):",
        "   return struct(variable = struct(",
        "       name = " + name + ",",
        "       value = " + value + ",",
        "       type_name = 'make_variable'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testWithFeatureSet() throws Exception {
    loadCcToolchainConfigLib();
    createWithFeatureSetRule("one", /* features= */ "None", /* notFeatures= */ "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("features parameter of with_feature_set should be a list, found NoneType");

    createWithFeatureSetRule("two", /* features= */ "['abc']", /* notFeatures= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("not_features parameter of with_feature_set should be a list, found NoneType");

    createWithFeatureSetRule("three", /* features= */ "'asdf'", /* notFeatures= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("features parameter of with_feature_set should be a list, found string");

    createWithFeatureSetRule("four", /* features= */ "['abc']", /* notFeatures= */ "'def'");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//four:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("not_features parameter of with_feature_set should be a list, found string");

    createWithFeatureSetRule(
        "five", /* features= */ "['f1', 'f2']", /* notFeatures= */ "['nf1', 'nf2']");

    ConfiguredTarget t = getConfiguredTarget("//five:a");
    SkylarkInfo withFeatureSetProvider = (SkylarkInfo) t.get("wfs");
    assertThat(withFeatureSetProvider).isNotNull();
    WithFeatureSet withFeatureSet = CcModule.withFeatureSetFromSkylark(withFeatureSetProvider);
    assertThat(withFeatureSet).isNotNull();
    assertThat(withFeatureSet.getFeatures()).containsExactly("f1", "f2");
    assertThat(withFeatureSet.getNotFeatures()).containsExactly("nf1", "nf2");

    createVariableWithValueRule("six", /* name= */ "'abc'", /* value= */ "'def'");
    t = getConfiguredTarget("//six:a");
    SkylarkInfo variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.withFeatureSetFromSkylark(variable);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'with_feature_set', received 'variable_with_value");
    }
  }

  private void createWithFeatureSetRule(String pkg, String features, String notFeatures)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'with_feature_set')",
        "def _impl(ctx):",
        "   return struct(wfs = with_feature_set(",
        "       features = " + features + ",",
        "       not_features = " + notFeatures + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomWithFeatureSet() throws Exception {
    createCustomWithFeatureSetRule("one", /* features= */ "struct()", /* notFeatures= */ "None");

    try {
      ConfiguredTarget t = getConfiguredTarget("//one:a");
      SkylarkInfo withFeatureSetProvider = (SkylarkInfo) t.get("wfs");
      assertThat(withFeatureSetProvider).isNotNull();
      CcModule.withFeatureSetFromSkylark(withFeatureSetProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Illegal argument: 'features' is not of expected type list or NoneType");
    }

    createCustomWithFeatureSetRule("two", /* features= */ "['abc']", /* notFeatures= */ "struct()");

    try {
      ConfiguredTarget t = getConfiguredTarget("//two:a");
      SkylarkInfo withFeatureSetProvider = (SkylarkInfo) t.get("wfs");
      assertThat(withFeatureSetProvider).isNotNull();
      CcModule.withFeatureSetFromSkylark(withFeatureSetProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Illegal argument: 'not_features' is not of expected type list or NoneType");
    }

    createCustomWithFeatureSetRule("three", /* features= */ "[struct()]", /* notFeatures= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo withFeatureSetProvider = (SkylarkInfo) t.get("wfs");
      assertThat(withFeatureSetProvider).isNotNull();
      CcModule.withFeatureSetFromSkylark(withFeatureSetProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("expected type 'string' for 'features' element but got type 'struct' instead");
    }

    createCustomWithFeatureSetRule("four", /* features= */ "[]", /* notFeatures= */ "[struct()]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo withFeatureSetProvider = (SkylarkInfo) t.get("wfs");
      assertThat(withFeatureSetProvider).isNotNull();
      CcModule.withFeatureSetFromSkylark(withFeatureSetProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "expected type 'string' for 'not_features' element but got type 'struct' instead");
    }
  }

  private void createCustomWithFeatureSetRule(String pkg, String features, String notFeatures)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "def _impl(ctx):",
        "   return struct(wfs = struct(",
        "       features = " + features + ",",
        "       not_features = " + notFeatures + ",",
        "       type_name = 'with_feature_set'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testEnvSet() throws Exception {
    loadCcToolchainConfigLib();
    createEnvSetRule(
        "one", /* actions= */ "['a1']", /* envEntries= */ "None", /* withFeatures= */ "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("env_entries parameter of env_set should be a list, found NoneType");

    createEnvSetRule(
        "two", /* actions= */ "['a1']", /* envEntries= */ "['abc']", /* withFeatures= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("with_features parameter of env_set should be a list, found NoneType");

    createEnvSetRule(
        "three", /* actions= */ "['a1']", /* envEntries= */ "'asdf'", /* withFeatures= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("env_entries parameter of env_set should be a list, found string");

    createEnvSetRule(
        "four", /* actions= */ "['a1']", /* envEntries= */ "['abc']", /* withFeatures= */ "'def'");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//four:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("with_features parameter of env_set should be a list, found string");

    createEnvSetRule(
        "five",
        /* actions= */ "['a1']",
        /* envEntries= */ "[env_entry(key = 'a', value = 'b'),"
            + "variable_with_value(name = 'a', value = 'b')]",
        /* withFeatures= */ "[]");

    ConfiguredTarget t = getConfiguredTarget("//five:a");
    SkylarkInfo envSetProvider = (SkylarkInfo) t.get("envset");
    assertThat(envSetProvider).isNotNull();
    try {
      CcModule.envSetFromSkylark(envSetProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'env_entry', received 'variable_with_value'");
    }

    createEnvSetRule("six", /* actions= */ "[]", /* envEntries= */ "[]", /* withFeatures= */ "[]");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//six:a"));
    assertThat(e).hasMessageThat().contains("actions parameter of env_set must be a nonempty list");

    createEnvSetRule(
        "seven",
        /* actions= */ "['a1']",
        /* envEntries= */ "[env_entry(key = 'a', value = 'b')]",
        /* withFeatures= */ "[with_feature_set(features = ['a'])]");

    t = getConfiguredTarget("//seven:a");
    envSetProvider = (SkylarkInfo) t.get("envset");
    assertThat(envSetProvider).isNotNull();
    EnvSet envSet = CcModule.envSetFromSkylark(envSetProvider);
    assertThat(envSet).isNotNull();

    createVariableWithValueRule("eight", /* name= */ "'abc'", /* value= */ "'def'");
    t = getConfiguredTarget("//eight:a");
    SkylarkInfo variable = (SkylarkInfo) t.get("variable");
    try {
      CcModule.envSetFromSkylark(variable);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'env_set', received 'variable_with_value");
    }
  }

  private void createEnvSetRule(String pkg, String actions, String envEntries, String withFeatures)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "   'env_set', 'env_entry', 'with_feature_set', 'variable_with_value')",
        "def _impl(ctx):",
        "   return struct(envset = env_set(",
        "       actions = " + actions + ",",
        "       env_entries = " + envEntries + ",",
        "       with_features = " + withFeatures + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomEnvSet() throws Exception {
    loadCcToolchainConfigLib();
    createCustomEnvSetRule(
        "one", /* actions= */ "[]", /* envEntries= */ "None", /* withFeatures= */ "None");
    ConfiguredTarget t = getConfiguredTarget("//one:a");
    SkylarkInfo envSetProvider = (SkylarkInfo) t.get("envset");
    assertThat(envSetProvider).isNotNull();
    try {
      CcModule.envSetFromSkylark(envSetProvider);
      fail("Should have failed because of empty action list.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("actions parameter of env_set must be a nonempty list");
    }

    createCustomEnvSetRule(
        "two", /* actions= */ "['a1']", /* envEntries= */ "struct()", /* withFeatures= */ "None");
    t = getConfiguredTarget("//two:a");
    envSetProvider = (SkylarkInfo) t.get("envset");
    assertThat(envSetProvider).isNotNull();
    try {
      CcModule.envSetFromSkylark(envSetProvider);
      fail("Should have failed because of wrong envEntries type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'env_entries' is not of expected type list or NoneType");
    }

    createCustomEnvSetRule(
        "three",
        /* actions= */ "['a1']",
        /* envEntries= */ "[struct()]",
        /* withFeatures= */ "None");
    t = getConfiguredTarget("//three:a");
    envSetProvider = (SkylarkInfo) t.get("envset");
    assertThat(envSetProvider).isNotNull();
    try {
      CcModule.envSetFromSkylark(envSetProvider);
      fail("Should have failed because of wrong envEntry type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Expected object of type 'env_entry', received 'struct'");
    }

    createCustomEnvSetRule(
        "four",
        /* actions= */ "['a1']",
        /* envEntries= */ "[env_entry(key = 'a', value = 'b')]",
        /* withFeatures= */ "'a'");
    t = getConfiguredTarget("//four:a");
    envSetProvider = (SkylarkInfo) t.get("envset");
    assertThat(envSetProvider).isNotNull();
    try {
      CcModule.envSetFromSkylark(envSetProvider);
      fail("Should have failed because of wrong withFeatures type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("'with_features' is not of expected type list or NoneType");
    }

    createCustomEnvSetRule(
        "five",
        /* actions= */ "['a1']",
        /* envEntries= */ "[env_entry(key = 'a', value = 'b')]",
        /* withFeatures= */ "[env_entry(key = 'a', value = 'b')]");
    t = getConfiguredTarget("//five:a");
    envSetProvider = (SkylarkInfo) t.get("envset");
    assertThat(envSetProvider).isNotNull();
    try {
      CcModule.envSetFromSkylark(envSetProvider);
      fail("Should have failed because of wrong withFeatures type.");
    } catch (EvalException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Expected object of type 'with_feature_set', received 'env_entry'.");
    }
  }

  private void createCustomEnvSetRule(
      String pkg, String actions, String envEntries, String withFeatures) throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "   'env_entry', 'with_feature_set', 'variable_with_value')",
        "def _impl(ctx):",
        "   return struct(envset = struct(",
        "       actions = " + actions + ",",
        "       env_entries = " + envEntries + ",",
        "       with_features = " + withFeatures + ",",
        "       type_name = 'env_set'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testFlagGroup() throws Exception {
    loadCcToolchainConfigLib();
    createFlagGroupRule(
        "one",
        /* flags= */ "[]",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "None",
        /* expandIfTrue= */ "None",
        /* expandIfFalse= */ "None",
        /* expandIfAvailable= */ "None",
        /* expandIfNotAvailable= */ "None",
        /* expandIfEqual= */ "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("flag_group must contain either a list of flags or a list of flag_groups");

    createFlagGroupRule(
        "two",
        /* flags= */ "['a']",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "struct(val = 'a')",
        /* expandIfTrue= */ "None",
        /* expandIfFalse= */ "None",
        /* expandIfAvailable= */ "None",
        /* expandIfNotAvailable= */ "None",
        /* expandIfEqual= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("iterate_over parameter of flag_group should be a string, found struct");

    createFlagGroupRule(
        "three",
        /* flags= */ "['a']",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "None",
        /* expandIfTrue= */ "struct(val = 'a')",
        /* expandIfFalse= */ "None",
        /* expandIfAvailable= */ "None",
        /* expandIfNotAvailable= */ "None",
        /* expandIfEqual= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("expand_if_true parameter of flag_group should be a string, found struct");

    createFlagGroupRule(
        "four",
        /* flags= */ "['a']",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "None",
        /* expandIfTrue= */ "None",
        /* expandIfFalse= */ "struct(val = 'a')",
        /* expandIfAvailable= */ "None",
        /* expandIfNotAvailable= */ "None",
        /* expandIfEqual= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//four:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("expand_if_false parameter of flag_group should be a string, found struct");

    createFlagGroupRule(
        "five",
        /* flags= */ "['a']",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "None",
        /* expandIfTrue= */ "None",
        /* expandIfFalse= */ "None",
        /* expandIfAvailable= */ "struct(val = 'a')",
        /* expandIfNotAvailable= */ "None",
        /* expandIfEqual= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//five:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("expand_if_available parameter of flag_group should be a string, found struct");

    createFlagGroupRule(
        "six",
        /* flags= */ "['a']",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "None",
        /* expandIfTrue= */ "None",
        /* expandIfFalse= */ "None",
        /* expandIfAvailable= */ "None",
        /* expandIfNotAvailable= */ "struct(val = 'a')",
        /* expandIfEqual= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//six:a"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "expand_if_not_available parameter of flag_group should be a string, found struct");

    createFlagGroupRule(
        "seven",
        /* flags= */ "['a']",
        /* flagGroups= */ "['b']",
        /* iterateOver= */ "None",
        /* expandIfTrue= */ "None",
        /* expandIfFalse= */ "None",
        /* expandIfAvailable= */ "None",
        /* expandIfNotAvailable= */ "struct(val = 'a')",
        /* expandIfEqual= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//seven:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("flag_group must not contain both a flag and another flag_group");

    createFlagGroupRule(
        "eight",
        /* flags= */ "['a']",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "'a'",
        /* expandIfTrue= */ "'b'",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "'a'");

    ConfiguredTarget t = getConfiguredTarget("//eight:a");
    SkylarkInfo flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "Field 'expand_if_equal' is not of "
                  + "'com.google.devtools.build.lib.packages.SkylarkInfo' type.");
    }

    createFlagGroupRule(
        "nine",
        /* flags= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a']), flag_group(flags = ['b'])]",
        /* iterateOver= */ "''",
        /* expandIfTrue= */ "''",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "variable_with_value(name = 'a', value = 'b')");

    t = getConfiguredTarget("//nine:a");
    flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    FlagGroup f = CcModule.flagGroupFromSkylark(flagGroupProvider);
    assertThat(f).isNotNull();

    createFlagGroupRule(
        "ten",
        /* flags= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a']), struct(value = 'a')]",
        /* iterateOver= */ "''",
        /* expandIfTrue= */ "''",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "variable_with_value(name = 'a', value = 'b')");

    t = getConfiguredTarget("//ten:a");
    flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'flag_group', received 'struct'");
    }
  }

  private void createFlagGroupRule(
      String pkg,
      String flags,
      String flagGroups,
      String iterateOver,
      String expandIfTrue,
      String expandIfFalse,
      String expandIfAvailable,
      String expandIfNotAvailable,
      String expandIfEqual)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "   'env_set', 'env_entry', 'with_feature_set', 'variable_with_value', 'flag_group')",
        "def _impl(ctx):",
        "   return struct(flaggroup = flag_group(",
        "       flags = " + flags + ",",
        "       flag_groups = " + flagGroups + ",",
        "       expand_if_true = " + expandIfTrue + ",",
        "       expand_if_false = " + expandIfFalse + ",",
        "       expand_if_available = " + expandIfAvailable + ",",
        "       expand_if_not_available = " + expandIfNotAvailable + ",",
        "       expand_if_equal = " + expandIfEqual + ",",
        "       iterate_over = " + iterateOver + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomFlagGroup() throws Exception {
    loadCcToolchainConfigLib();

    createCustomFlagGroupRule(
        "one",
        /* flags= */ "['a']",
        /* flagGroups= */ "[]",
        /* iterateOver= */ "struct()",
        /* expandIfTrue= */ "'b'",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "None");

    ConfiguredTarget t = getConfiguredTarget("//one:a");
    SkylarkInfo flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'iterate_over' is not of 'java.lang.String' type.");
    }

    createCustomFlagGroupRule(
        "two",
        /* flags= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a']), flag_group(flags = ['b'])]",
        /* iterateOver= */ "''",
        /* expandIfTrue= */ "struct()",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "variable_with_value(name = 'a', value = 'b')");

    t = getConfiguredTarget("//two:a");
    flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'expand_if_true' is not of 'java.lang.String' type.");
    }

    createCustomFlagGroupRule(
        "three",
        /* flags= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* iterateOver= */ "''",
        /* expandIfTrue= */ "''",
        /* expandIfFalse= */ "True",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "variable_with_value(name = 'a', value = 'b')");

    t = getConfiguredTarget("//three:a");
    flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'expand_if_false' is not of 'java.lang.String' type.");
    }

    createCustomFlagGroupRule(
        "four",
        /* flags= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* iterateOver= */ "''",
        /* expandIfTrue= */ "''",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "struct()",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "variable_with_value(name = 'a', value = 'b')");

    t = getConfiguredTarget("//four:a");
    flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'expand_if_available' is not of 'java.lang.String' type.");
    }

    createCustomFlagGroupRule(
        "five",
        /* flags= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* iterateOver= */ "''",
        /* expandIfTrue= */ "''",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "3",
        /* expandIfEqual= */ "variable_with_value(name = 'a', value = 'b')");

    t = getConfiguredTarget("//five:a");
    flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'expand_if_not_available' is not of 'java.lang.String' type.");
    }

    createCustomFlagGroupRule(
        "six",
        /* flags= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* iterateOver= */ "''",
        /* expandIfTrue= */ "''",
        /* expandIfFalse= */ "''",
        /* expandIfAvailable= */ "''",
        /* expandIfNotAvailable= */ "''",
        /* expandIfEqual= */ "struct(name = 'a', value = 'b')");

    t = getConfiguredTarget("//six:a");
    flagGroupProvider = (SkylarkInfo) t.get("flaggroup");
    assertThat(flagGroupProvider).isNotNull();
    try {
      CcModule.flagGroupFromSkylark(flagGroupProvider);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'variable_with_value', received 'struct'.");
    }
  }

  private void createCustomFlagGroupRule(
      String pkg,
      String flags,
      String flagGroups,
      String iterateOver,
      String expandIfTrue,
      String expandIfFalse,
      String expandIfAvailable,
      String expandIfNotAvailable,
      String expandIfEqual)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "   'env_set', 'env_entry', 'with_feature_set', 'variable_with_value', 'flag_group')",
        "def _impl(ctx):",
        "   return struct(flaggroup = struct(",
        "       flags = " + flags + ",",
        "       flag_groups = " + flagGroups + ",",
        "       expand_if_true = " + expandIfTrue + ",",
        "       expand_if_false = " + expandIfFalse + ",",
        "       expand_if_available = " + expandIfAvailable + ",",
        "       expand_if_not_available = " + expandIfNotAvailable + ",",
        "       expand_if_equal = " + expandIfEqual + ",",
        "       iterate_over = " + iterateOver + ",",
        "       type_name = 'flag_group'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testTool() throws Exception {
    loadCcToolchainConfigLib();
    createToolRule("one", /* path= */ "''", /* withFeatures= */ "[]", /* requirements= */ "[]");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e).hasMessageThat().contains("path parameter of tool must be a nonempty string");

    createToolRule("two", /* path= */ "'a'", /* withFeatures= */ "None", /* requirements= */ "[]");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("with_features parameter of tool should be a list, found NoneType");

    createToolRule(
        "three", /* path= */ "'a'", /* withFeatures= */ "[]", /* requirements= */ "None");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("execution_requirements parameter of tool should be a list, found NoneType");

    createToolRule(
        "four",
        /* path= */ "'a'",
        /* withFeatures= */ "[struct(val = 'a')]",
        /* requirements= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'with_feature_set', received 'struct'");
    }

    createToolRule(
        "five",
        /* path= */ "'a'",
        /* withFeatures= */ "[]",
        /* requirements= */ "[struct(val = 'a')]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//five:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "expected type 'string' for 'execution_requirements' "
                  + "element but got type 'struct' instead");
    }

    createToolRule(
        "six",
        /* path= */ "'/a/b/c'",
        /* withFeatures= */ "[with_feature_set(features = ['a'])]",
        /* requirements= */ "['a', 'b']");

    ConfiguredTarget t = getConfiguredTarget("//six:a");
    SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
    assertThat(toolStruct).isNotNull();
    Tool tool = CcModule.toolFromSkylark(toolStruct);
    assertThat(tool.getExecutionRequirements()).containsExactly("a", "b");
    assertThat(tool.getToolPathString(PathFragment.EMPTY_FRAGMENT)).isEqualTo("/a/b/c");
    assertThat(tool.getWithFeatureSetSets())
        .contains(new WithFeatureSet(ImmutableSet.of("a"), ImmutableSet.of()));
  }

  private void createToolRule(String pkg, String path, String withFeatures, String requirements)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'with_feature_set', 'tool')",
        "def _impl(ctx):",
        "   return struct(tool = tool(",
        "       path = " + path + ",",
        "       with_features = " + withFeatures + ",",
        "       execution_requirements = " + requirements + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomTool() throws Exception {
    loadCcToolchainConfigLib();
    createCustomToolRule(
        "one", /* path= */ "''", /* withFeatures= */ "[]", /* requirements= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//one:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("The 'path' field of tool must be a nonempty string.");
    }

    createCustomToolRule(
        "two", /* path= */ "struct()", /* withFeatures= */ "[]", /* requirements= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//two:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee).hasMessageThat().contains("Field 'path' is not of 'java.lang.String' type.");
    }

    createCustomToolRule(
        "three", /* path= */ "'a'", /* withFeatures= */ "struct()", /* requirements= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'with_features' is not of expected type list or NoneType");
    }

    createCustomToolRule(
        "four",
        /* path= */ "'a'",
        /* withFeatures= */ "[struct(val = 'a')]",
        /* requirements= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'with_feature_set', received 'struct'");
    }

    createCustomToolRule(
        "five", /* path= */ "'a'", /* withFeatures= */ "[]", /* requirements= */ "'a'");

    try {
      ConfiguredTarget t = getConfiguredTarget("//five:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "llegal argument: 'execution_requirements' is not of expected type list or NoneType");
    }

    createCustomToolRule(
        "six", /* path= */ "'a'", /* withFeatures= */ "[]", /* requirements= */ "[struct()]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//six:a");
      SkylarkInfo toolStruct = (SkylarkInfo) t.getValue("tool");
      assertThat(toolStruct).isNotNull();
      CcModule.toolFromSkylark(toolStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "expected type 'string' for 'execution_requirements' "
                  + "element but got type 'struct' instead");
    }
  }

  private void createCustomToolRule(
      String pkg, String path, String withFeatures, String requirements) throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'with_feature_set')",
        "def _impl(ctx):",
        "   return struct(tool = struct(",
        "       path = " + path + ",",
        "       with_features = " + withFeatures + ",",
        "       execution_requirements = " + requirements + ",",
        "       type_name = 'tool'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testFlagSet() throws Exception {
    loadCcToolchainConfigLib();

    createFlagSetRule(
        "two", /* actions= */ "['a']", /* flagGroups= */ "[]", /* withFeatures= */ "None");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("with_features parameter of flag_set should be a list, found NoneType");

    createFlagSetRule(
        "three", /* actions= */ "['a']", /* flagGroups= */ "None", /* withFeatures= */ "[]");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//three:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("flag_groups parameter of flag_set should be a list, found NoneType");

    createFlagSetRule(
        "four",
        /* actions= */ "['a', struct(val = 'a')]",
        /* flagGroups= */ "[]",
        /* withFeatures= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo flagSetStruct = (SkylarkInfo) t.getValue("flagset");
      assertThat(flagSetStruct).isNotNull();
      CcModule.flagSetFromSkylark(flagSetStruct, /* actionName= */ null);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("expected type 'string' for 'actions' element but got type 'struct' instead");
    }

    createFlagSetRule(
        "five",
        /* actions= */ "['a']",
        /* flagGroups= */ "[flag_group(flags = ['a']), struct(value = 'a')]",
        /* withFeatures= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//five:a");
      SkylarkInfo flagSetStruct = (SkylarkInfo) t.getValue("flagset");
      assertThat(flagSetStruct).isNotNull();
      CcModule.flagSetFromSkylark(flagSetStruct, /* actionName= */ null);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'flag_group', received 'struct'");
    }

    createFlagSetRule(
        "six",
        /* actions= */ "['a']",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* withFeatures= */ "[struct(val = 'a')]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//six:a");
      SkylarkInfo flagSetStruct = (SkylarkInfo) t.getValue("flagset");
      assertThat(flagSetStruct).isNotNull();
      CcModule.flagSetFromSkylark(flagSetStruct, /* actionName= */ null);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'with_feature_set', received 'struct'");
    }

    createFlagSetRule(
        "seven",
        /* actions= */ "['a']",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* withFeatures= */ "[with_feature_set(features = ['a'])]");
    ConfiguredTarget t = getConfiguredTarget("//seven:a");
    SkylarkInfo flagSetStruct = (SkylarkInfo) t.getValue("flagset");
    assertThat(flagSetStruct).isNotNull();
    FlagSet f = CcModule.flagSetFromSkylark(flagSetStruct, /* actionName= */ null);
    assertThat(f).isNotNull();

    createFlagSetRule(
        "eight",
        /* actions= */ "['a']",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* withFeatures= */ "[struct(val = 'a')]");

    try {
      t = getConfiguredTarget("//eight:a");
      flagSetStruct = (SkylarkInfo) t.getValue("flagset");
      assertThat(flagSetStruct).isNotNull();
      CcModule.flagSetFromSkylark(flagSetStruct, /* actionName= */ "action");
      fail("Should have failed because of nonempty actions field when created from action_config.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Thus, you must not specify action lists in an action_config's flag set.");
    }

    createFlagSetRule(
        "nine",
        /* actions= */ "[]",
        /* flagGroups= */ "[flag_group(flags = ['a'])]",
        /* withFeatures= */ "[with_feature_set(features = ['a'])]");
    t = getConfiguredTarget("//nine:a");
    flagSetStruct = (SkylarkInfo) t.getValue("flagset");
    assertThat(flagSetStruct).isNotNull();
    f = CcModule.flagSetFromSkylark(flagSetStruct, /* actionName= */ "action");
    assertThat(f).isNotNull();
    assertThat(f.getActions()).containsExactly("action");
  }

  private void createFlagSetRule(String pkg, String actions, String flagGroups, String withFeatures)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "   'env_set', 'env_entry', 'with_feature_set', 'variable_with_value', 'flag_group',",
        "   'flag_set')",
        "def _impl(ctx):",
        "   return struct(flagset = flag_set(",
        "       flag_groups = " + flagGroups + ",",
        "       actions = " + actions + ",",
        "       with_features = " + withFeatures + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomFlagSet() throws Exception {
    loadCcToolchainConfigLib();
    createCustomFlagSetRule(
        "one", /* actions= */ "[]", /* flagGroups= */ "[]", /* withFeatures= */ "[]");

    ConfiguredTarget target = getConfiguredTarget("//one:a");
    SkylarkInfo flagSet = (SkylarkInfo) target.getValue("flagset");
    assertThat(flagSet).isNotNull();
    FlagSet flagSetObject = CcModule.flagSetFromSkylark(flagSet, /* actionName */ null);
    assertThat(flagSetObject).isNotNull();

    createCustomFlagSetRule(
        "two", /* actions= */ "['a']", /* flagGroups= */ "struct()", /* withFeatures= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//two:a");
      SkylarkInfo flagSetStruct = (SkylarkInfo) t.getValue("flagset");
      assertThat(flagSetStruct).isNotNull();
      CcModule.flagSetFromSkylark(flagSetStruct, /* actionName */ null);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'flag_groups' is not of expected type list or NoneType");
    }

    createCustomFlagSetRule(
        "three", /* actions= */ "['a']", /* flagGroups= */ "[]", /* withFeatures= */ "struct()");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo flagSetStruct = (SkylarkInfo) t.getValue("flagset");
      assertThat(flagSetStruct).isNotNull();
      CcModule.flagSetFromSkylark(flagSetStruct, /* actionName */ null);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'with_features' is not of expected type list or NoneType");
    }

    createCustomFlagSetRule(
        "four", /* actions= */ "struct()", /* flagGroups= */ "[]", /* withFeatures= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo flagSetStruct = (SkylarkInfo) t.getValue("flagset");
      assertThat(flagSetStruct).isNotNull();
      CcModule.flagSetFromSkylark(flagSetStruct, /* actionName */ null);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'actions' is not of expected type list or NoneType");
    }
  }

  private void createCustomFlagSetRule(
      String pkg, String actions, String flagGroups, String withFeatures) throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "   'env_set', 'env_entry', 'with_feature_set', 'variable_with_value', 'flag_group',",
        "   'flag_set')",
        "def _impl(ctx):",
        "   return struct(flagset = struct(",
        "       flag_groups = " + flagGroups + ",",
        "       actions = " + actions + ",",
        "       with_features = " + withFeatures + ",",
        "       type_name = 'flag_set'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testActionConfig() throws Exception {
    loadCcToolchainConfigLib();
    createActionConfigRule(
        "one",
        /* actionName= */ "''",
        /* enabled= */ "True",
        /* tools= */ "[]",
        /* flagSets= */ "[]",
        /* implies= */ "[]");

    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//one:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("name parameter of action_config must be a nonempty string");

    createActionConfigRule(
        "two",
        /* actionName= */ "'actionname'",
        /* enabled= */ "['asd']",
        /* tools= */ "[]",
        /* flagSets= */ "[]",
        /* implies= */ "[]");
    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("enabled parameter of action_config should be a bool, found list");

    createActionConfigRule(
        "three",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "[with_feature_set(features = ['a'])]",
        /* flagSets= */ "[]",
        /* implies= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'tool', received 'with_feature_set'");
    }

    createActionConfigRule(
        "four",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[tool(path = 'a/b/c')]",
        /* implies= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'flag_set', received 'tool'");
    }

    createActionConfigRule(
        "five",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[]",
        /* implies= */ "flag_set(actions = ['a', 'b'])");

    e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//five:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("implies parameter of action_config should be a list, found struct");

    createActionConfigRule(
        "six",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[]",
        /* implies= */ "[flag_set(actions = ['a', 'b'])]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//six:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("expected type 'string' for 'implies' element but got type 'struct' instead");
    }

    createActionConfigRule(
        "seven",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[]",
        /* implies= */ "[flag_set(actions = ['a', 'b'])]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//seven:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("expected type 'string' for 'implies' element but got type 'struct' instead");
    }

    createActionConfigRule(
        "eight",
        /* actionName= */ "'actionname32._++-'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[flag_set(flag_groups=[flag_group(flags=['a'])])]",
        /* implies= */ "['a', 'b']");

    ConfiguredTarget t = getConfiguredTarget("//eight:a");
    SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
    assertThat(actionConfigStruct).isNotNull();
    ActionConfig a = CcModule.actionConfigFromSkylark(actionConfigStruct);
    assertThat(a).isNotNull();
    assertThat(a.getActionName()).isEqualTo("actionname32._++-");
    assertThat(a.getImplies()).containsExactly("a", "b").inOrder();
    assertThat(Iterables.getOnlyElement(a.getFlagSets()).getActions())
        .containsExactly("actionname32._++-");

    createActionConfigRule(
        "nine",
        /* actionName= */ "'Upper'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[]",
        /* implies= */ "[flag_set(actions = ['a', 'b'])]");

    try {
      t = getConfiguredTarget("//nine:a");
      actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "An action_config's name must consist solely "
                  + "of lowercase ASCII letters, digits, '.', '_', '+', and '-', got 'Upper'");
    }

    createActionConfigRule(
        "ten",
        /* actionName= */ "'white\tspace'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[]",
        /* implies= */ "[flag_set(actions = ['a', 'b'])]");

    try {
      t = getConfiguredTarget("//ten:a");
      actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "An action_config's name must consist solely "
                  + "of lowercase ASCII letters, digits, '.', '_', '+', and '-', "
                  + "got 'white\tspace'");
    }
  }

  private void createActionConfigRule(
      String pkg, String actionName, String enabled, String tools, String flagSets, String implies)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'with_feature_set',",
        "             'tool', 'flag_set', 'action_config', 'flag_group')",
        "def _impl(ctx):",
        "   return struct(config = action_config(",
        "       action_name = " + actionName + ",",
        "       enabled = " + enabled + ",",
        "       tools = " + tools + ",",
        "       flag_sets = " + flagSets + ",",
        "       implies = " + implies + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomActionConfig() throws Exception {
    loadCcToolchainConfigLib();
    createCustomActionConfigRule(
        "one",
        /* actionName= */ "struct()",
        /* enabled= */ "True",
        /* tools= */ "[]",
        /* flagSets= */ "[]",
        /* implies= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//one:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'action_name' is not of 'java.lang.String' type.");
    }

    createCustomActionConfigRule(
        "two",
        /* actionName= */ "'actionname'",
        /* enabled= */ "['asd']",
        /* tools= */ "[]",
        /* flagSets= */ "[]",
        /* implies= */ "[]");
    try {
      ConfiguredTarget t = getConfiguredTarget("//two:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'enabled' is not of 'java.lang.Boolean' type.");
    }

    createCustomActionConfigRule(
        "three",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "struct()",
        /* flagSets= */ "[]",
        /* implies= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'tools' is not of expected type list or NoneType");
    }

    createCustomActionConfigRule(
        "four",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "True",
        /* implies= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'flag_sets' is not of expected type list or NoneType");
    }

    createCustomActionConfigRule(
        "five",
        /* actionName= */ "'actionname'",
        /* enabled= */ "True",
        /* tools= */ "[tool(path = 'a/b/c')]",
        /* flagSets= */ "[]",
        /* implies= */ "flag_set(actions = ['a', 'b'])");

    try {
      ConfiguredTarget t = getConfiguredTarget("//five:a");
      SkylarkInfo actionConfigStruct = (SkylarkInfo) t.getValue("config");
      assertThat(actionConfigStruct).isNotNull();
      CcModule.actionConfigFromSkylark(actionConfigStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'implies' is not of expected type list or NoneType");
    }
  }

  private void createCustomActionConfigRule(
      String pkg, String actionName, String enabled, String tools, String flagSets, String implies)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'with_feature_set',",
        "             'tool', 'flag_set', 'action_config', )",
        "def _impl(ctx):",
        "   return struct(config = struct(",
        "       action_name = " + actionName + ",",
        "       enabled = " + enabled + ",",
        "       tools = " + tools + ",",
        "       flag_sets = " + flagSets + ",",
        "       implies = " + implies + ",",
        "       type_name = 'action_config'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testFeature() throws Exception {
    loadCcToolchainConfigLib();
    createFeatureRule(
        "one",
        /* name= */ "''",
        /* enabled= */ "False",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//one:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("A feature must either have a nonempty 'name' field or be enabled.");
    }

    createFeatureRule(
        "two",
        /* name= */ "'featurename'",
        /* enabled= */ "None",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//two:a"));
    assertThat(e)
        .hasMessageThat()
        .contains("enabled parameter of feature should be a bool, found NoneType");

    createFeatureRule(
        "three",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[struct()]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'flag_set', received 'struct'");
    }

    createFeatureRule(
        "four",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[flag_set(actions = ['a'], flag_groups = [flag_group(flags = ['a'])])]",
        /* envSets= */ "[tool(path = 'a/b/c')]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Expected object of type 'env_set', received 'tool'");
    }

    createFeatureRule(
        "five",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[flag_set(actions = ['a'], flag_groups = [flag_group(flags = ['a'])])]",
        /* envSets= */ "[env_set(actions = ['a1'], "
            + "env_entries = [env_entry(key = 'a', value = 'b')])]",
        /* requires= */ "[tool(path = 'a/b/c')]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//five:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee).hasMessageThat().contains("expected object of type 'feature_set'");
    }

    createFeatureRule(
        "six",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[flag_set(actions = ['a'], flag_groups = [flag_group(flags = ['a'])])]",
        /* envSets= */ "[env_set(actions = ['a1'], "
            + "env_entries = [env_entry(key = 'a', value = 'b')])]",
        /* requires= */ "[feature_set(features = ['f1', 'f2'])]",
        /* implies= */ "[tool(path = 'a/b/c')]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//six:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("expected type 'string' for 'implies' element but got type 'struct' instead");
    }

    createFeatureRule(
        "seven",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[flag_set(actions = ['a'], flag_groups = [flag_group(flags = ['a'])])]",
        /* envSets= */ "[env_set(actions = ['a1'], "
            + "env_entries = [env_entry(key = 'a', value = 'b')])]",
        /* requires= */ "[feature_set(features = ['f1', 'f2'])]",
        /* implies= */ "['a', 'b', 'c']",
        /* provides= */ "[struct()]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//seven:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("expected type 'string' for 'provides' element but got type 'struct' instead");
    }

    createFeatureRule(
        "eight",
        /* name= */ "'featurename32+.-_'",
        /* enabled= */ "True",
        /* flagSets= */ "[flag_set(actions = ['a'], flag_groups = [flag_group(flags = ['a'])])]",
        /* envSets= */ "[env_set(actions = ['a1'], "
            + "env_entries = [env_entry(key = 'a', value = 'b')])]",
        /* requires= */ "[feature_set(features = ['f1', 'f2'])]",
        /* implies= */ "['a', 'b', 'c']",
        /* provides= */ "['a', 'b', 'c']");

    ConfiguredTarget t = getConfiguredTarget("//eight:a");
    SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
    assertThat(featureStruct).isNotNull();
    Feature a = CcModule.featureFromSkylark(featureStruct);
    assertThat(a).isNotNull();

    createFeatureRule(
        "nine",
        /* name= */ "'UpperCase'",
        /* enabled= */ "False",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      t = getConfiguredTarget("//nine:a");
      featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "A feature's name must consist solely of lowercase ASCII letters, digits, "
                  + "'.', '_', '+', and '-', got 'UpperCase'");
    }

    createFeatureRule(
        "ten",
        /* name= */ "'white space'",
        /* enabled= */ "False",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      t = getConfiguredTarget("//ten:a");
      featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "A feature's name must consist solely of "
                  + "lowercase ASCII letters, digits, '.', '_', '+', and '-', got 'white space");
    }
  }

  private void createFeatureRule(
      String pkg,
      String name,
      String enabled,
      String flagSets,
      String envSets,
      String requires,
      String implies,
      String provides)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'with_feature_set', 'feature_set',",
        "             'flag_set', 'flag_group', 'tool', 'env_set', 'env_entry', 'feature')",
        "def _impl(ctx):",
        "   return struct(f = feature(",
        "       name = " + name + ",",
        "       enabled = " + enabled + ",",
        "       flag_sets = " + flagSets + ",",
        "       env_sets = " + envSets + ",",
        "       requires = " + requires + ",",
        "       implies = " + implies + ",",
        "       provides = " + provides + "))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomFeature() throws Exception {
    loadCcToolchainConfigLib();
    createCustomFeatureRule(
        "one",
        /* name= */ "struct()",
        /* enabled= */ "False",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//one:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee).hasMessageThat().contains("Field 'name' is not of 'java.lang.String' type.");
    }

    createCustomFeatureRule(
        "two",
        /* name= */ "'featurename'",
        /* enabled= */ "struct()",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");
    try {
      ConfiguredTarget t = getConfiguredTarget("//two:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'enabled' is not of 'java.lang.Boolean' type.");
    }

    createCustomFeatureRule(
        "three",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "struct()",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'flag_sets' is not of expected type list or NoneType");
    }

    createCustomFeatureRule(
        "four",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[]",
        /* envSets= */ "struct()",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'env_sets' is not of expected type list or NoneType");
    }

    createCustomFeatureRule(
        "five",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "struct()",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//five:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'requires' is not of expected type list or NoneType");
    }

    createCustomFeatureRule(
        "six",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "struct()",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//six:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'implies' is not of expected type list or NoneType");
    }

    createCustomFeatureRule(
        "seven",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "struct()");

    try {
      ConfiguredTarget t = getConfiguredTarget("//seven:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Illegal argument: 'provides' is not of expected type list or NoneType");
    }

    createCustomFeatureRule(
        "eight",
        /* name= */ "'featurename'",
        /* enabled= */ "True",
        /* flagSets= */ "[flag_set()]",
        /* envSets= */ "[]",
        /* requires= */ "[]",
        /* implies= */ "[]",
        /* provides= */ "[]");

    try {
      ConfiguredTarget t = getConfiguredTarget("//eight:a");
      SkylarkInfo featureStruct = (SkylarkInfo) t.getValue("f");
      assertThat(featureStruct).isNotNull();
      CcModule.featureFromSkylark(featureStruct);
      fail("Should have failed because of empty 'actions' parameter in flag_set.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("A flag_set that belongs to a feature must have nonempty 'actions' parameter.");
    }
  }

  private void createCustomFeatureRule(
      String pkg,
      String name,
      String enabled,
      String flagSets,
      String envSets,
      String requires,
      String implies,
      String provides)
      throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl', 'with_feature_set', 'feature_set',",
        "             'flag_set', 'flag_group', 'tool', 'env_set', 'env_entry', 'feature')",
        "def _impl(ctx):",
        "   return struct(f = struct(",
        "       name = " + name + ",",
        "       enabled = " + enabled + ",",
        "       flag_sets = " + flagSets + ",",
        "       env_sets = " + envSets + ",",
        "       requires = " + requires + ",",
        "       implies = " + implies + ",",
        "       provides = " + provides + ",",
        "       type_name = 'feature'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCustomArtifactNamePattern() throws Exception {
    loadCcToolchainConfigLib();
    createCustomArtifactNamePatternRule(
        "one", /* categoryName= */ "struct()", /* extension= */ "'a'", /* prefix= */ "'a'");

    try {
      ConfiguredTarget t = getConfiguredTarget("//one:a");
      SkylarkInfo artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
      assertThat(artifactNamePatternStruct).isNotNull();
      CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'category_name' is not of 'java.lang.String' type.");
    }

    createCustomArtifactNamePatternRule(
        "two",
        /* categoryName= */ "'static_library'",
        /* extension= */ "struct()",
        /* prefix= */ "'a'");

    try {
      ConfiguredTarget t = getConfiguredTarget("//two:a");
      SkylarkInfo artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
      assertThat(artifactNamePatternStruct).isNotNull();
      CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains("Field 'extension' is not of 'java.lang.String' type.");
    }

    createCustomArtifactNamePatternRule(
        "three",
        /* categoryName= */ "'static_library'",
        /* extension= */ "'.a'",
        /* prefix= */ "struct()");

    try {
      ConfiguredTarget t = getConfiguredTarget("//three:a");
      SkylarkInfo artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
      assertThat(artifactNamePatternStruct).isNotNull();
      CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
      fail("Should have failed because of wrong object type.");
    } catch (EvalException ee) {
      assertThat(ee).hasMessageThat().contains("Field 'prefix' is not of 'java.lang.String' type.");
    }

    createCustomArtifactNamePatternRule(
        "four", /* categoryName= */ "''", /* extension= */ "'.a'", /* prefix= */ "'a'");

    try {
      ConfiguredTarget t = getConfiguredTarget("//four:a");
      SkylarkInfo artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
      assertThat(artifactNamePatternStruct).isNotNull();
      CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
      fail("Should have failed because of empty string.");
    } catch (EvalException ee) {
      assertThat(ee)
          .hasMessageThat()
          .contains(
              "The 'category_name' field of artifact_name_pattern must be a nonempty string.");
    }

    createCustomArtifactNamePatternRule(
        "five", /* categoryName= */ "'executable'", /* extension= */ "''", /* prefix= */ "''");

    ConfiguredTarget t = getConfiguredTarget("//five:a");
    SkylarkInfo artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
    assertThat(artifactNamePatternStruct).isNotNull();
    ArtifactNamePattern artifactNamePattern =
        CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
    assertThat(artifactNamePattern).isNotNull();

    createCustomArtifactNamePatternRule(
        "six", /* categoryName= */ "'executable'", /* extension= */ "None", /* prefix= */ "None");

    t = getConfiguredTarget("//six:a");
    artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
    assertThat(artifactNamePatternStruct).isNotNull();
    artifactNamePattern = CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
    assertThat(artifactNamePattern).isNotNull();

    createCustomArtifactNamePatternRule(
        "seven", /* categoryName= */ "'unknown'", /* extension= */ "'.a'", /* prefix= */ "'a'");

    try {
      t = getConfiguredTarget("//seven:a");
      artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
      assertThat(artifactNamePatternStruct).isNotNull();
      CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
      fail("Should have failed because of unrecognized category.");
    } catch (EvalException ee) {
      assertThat(ee).hasMessageThat().contains("Artifact category unknown not recognized");
    }

    createCustomArtifactNamePatternRule(
        "eight",
        /* categoryName= */ "'static_library'",
        /* extension= */ "'a'",
        /* prefix= */ "'a'");

    try {
      t = getConfiguredTarget("//eight:a");
      artifactNamePatternStruct = (SkylarkInfo) t.getValue("namepattern");
      assertThat(artifactNamePatternStruct).isNotNull();
      CcModule.artifactNamePatternFromSkylark(artifactNamePatternStruct);
      fail("Should have failed because of unrecognized extension.");
    } catch (EvalException ee) {
      assertThat(ee).hasMessageThat().contains("Unrecognized file extension 'a'");
    }
  }

  private void createCustomArtifactNamePatternRule(
      String pkg, String categoryName, String extension, String prefix) throws Exception {
    scratch.file(
        pkg + "/foo.bzl",
        "def _impl(ctx):",
        "   return struct(namepattern = struct(",
        "       category_name = " + categoryName + ",",
        "       extension = " + extension + ",",
        "       prefix = " + prefix + ",",
        "       type_name = 'artifact_name_pattern'))",
        "crule = rule(implementation = _impl)");
    scratch.file(pkg + "/BUILD", "load(':foo.bzl', 'crule')", "crule(name = 'a')");
  }

  @Test
  public void testCcToolchainInfoFromSkylark() throws Exception {
    loadCcToolchainConfigLib();
    scratch.file(
        "foo/crosstool.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "        'feature',",
        "        'action_config',",
        "        'artifact_name_pattern',",
        "        'env_entry',",
        "        'variable_with_value',",
        "        'make_variable',",
        "        'feature_set',",
        "        'with_feature_set',",
        "        'env_set',",
        "        'flag_group',",
        "        'flag_set',",
        "        'tool_path',",
        "        'tool')",
        "",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "                ctx = ctx,",
        "                features = [feature(name = 'featureone')],",
        "                action_configs = [action_config(action_name = 'action', enabled=True)],",
        "                artifact_name_patterns = [artifact_name_pattern(",
        "                   category_name = 'static_library',",
        "                   prefix = 'prefix',",
        "                   extension = '.a')],",
        "                cxx_builtin_include_directories = ['dir1', 'dir2', 'dir3'],",
        "                toolchain_identifier = 'toolchain',",
        "                host_system_name = 'host',",
        "                target_system_name = 'target',",
        "                target_cpu = 'cpu',",
        "                target_libc = 'libc',",
        "                compiler = 'compiler',",
        "                abi_libc_version = 'abi_libc',",
        "                abi_version = 'abi',",
        "                tool_paths = [tool_path(name = 'name1', path = 'path1')],",
        "                cc_target_os = 'os',",
        "                builtin_sysroot = 'sysroot',",
        "                make_variables = [make_variable(name = 'acs', value = 'asd')])",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo], ",
        ")");

    scratch.file(
        "foo/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "cc_toolchain_alias(name='alias')",
        "cc_toolchain_config_rule(name='r')");
    useConfiguration("--experimental_enable_cc_toolchain_config_info");
    ConfiguredTarget target = getConfiguredTarget("//foo:r");
    assertThat(target).isNotNull();
    CcToolchainConfigInfo ccToolchainConfigInfo =
        (CcToolchainConfigInfo) target.get(CcToolchainConfigInfo.PROVIDER.getKey());
    assertThat(ccToolchainConfigInfo).isNotNull();

    useConfiguration("--experimental_enable_cc_toolchain_config_info=false");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("Creating a CcToolchainConfigInfo is not enabled.");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredToolchainIdentifier() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("toolchain_identifier");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e)
        .hasMessageThat()
        .contains("parameter 'toolchain_identifier' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredHostSystemName() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("host_system_name");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("parameter 'host_system_name' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredTargetSystemName() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("target_system_name");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("parameter 'target_system_name' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredTargetCpu() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("target_cpu");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("parameter 'target_cpu' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredTargetLibc() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("target_libc");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("parameter 'target_libc' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredCompiler() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("compiler");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("parameter 'compiler' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredAbiVersion() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("abi_version");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("parameter 'abi_version' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkRequiredAbiLibcVersion() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("abi_libc_version");
    AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:r"));
    assertThat(e).hasMessageThat().contains("parameter 'abi_libc_version' has no default value");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkAllRequiredStringsPresent() throws Exception {
    setupSkylarkRuleForStringFieldsTesting("");
    useConfiguration("--experimental_enable_cc_toolchain_config_info");
    ConfiguredTarget target = getConfiguredTarget("//foo:r");
    assertThat(target).isNotNull();
    CcToolchainConfigInfo ccToolchainConfigInfo =
        (CcToolchainConfigInfo) target.get(CcToolchainConfigInfo.PROVIDER.getKey());
    assertThat(ccToolchainConfigInfo).isNotNull();
  }

  private void setupSkylarkRuleForStringFieldsTesting(String fieldToExclude) throws Exception {
    ImmutableList<String> fields =
        ImmutableList.of(
            "toolchain_identifier = 'identifier'",
            "host_system_name = 'host_system_name'",
            "target_system_name = 'target_system_name'",
            "target_cpu = 'target_cpu'",
            "target_libc = 'target_libc'",
            "compiler = 'compiler'",
            "abi_version = 'abi'",
            "abi_libc_version = 'abi_libc'");

    scratch.file(
        "foo/crosstool.bzl",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "    ctx = ctx,",
        Joiner.on(",\n")
            .join(fields.stream().filter(el -> !el.startsWith(fieldToExclude + " =")).toArray()),
        ")",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo], ",
        ")");
    scratch.file(
        "foo/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "cc_toolchain_alias(name='alias')",
        "cc_toolchain_config_rule(name='r')");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkNoLegacyFeatures() throws Exception {
    loadCcToolchainConfigLib();
    scratch.file(
        "foo/crosstool.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "        'feature',",
        "        'action_config',",
        "        'artifact_name_pattern',",
        "        'env_entry',",
        "        'variable_with_value',",
        "        'make_variable',",
        "        'feature_set',",
        "        'with_feature_set',",
        "        'env_set',",
        "        'flag_group',",
        "        'flag_set',",
        "        'tool_path',",
        "        'tool')",
        "",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "                ctx = ctx,",
        "                features = [",
        "                    feature(name = 'no_legacy_features'),",
        "                    feature(name = 'custom_feature'),",
        "                ],",
        "                action_configs = [action_config(action_name = 'custom_action')],",
        "                artifact_name_patterns = [artifact_name_pattern(",
        "                   category_name = 'static_library',",
        "                   prefix = 'prefix',",
        "                   extension = '.a')],",
        "                toolchain_identifier = 'toolchain',",
        "                host_system_name = 'host',",
        "                target_system_name = 'target',",
        "                target_cpu = 'cpu',",
        "                target_libc = 'libc',",
        "                compiler = 'compiler',",
        "                abi_libc_version = 'abi_libc',",
        "                abi_version = 'abi')",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo], ",
        ")");

    scratch.file(
        "foo/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "cc_toolchain_alias(name='alias')",
        "cc_toolchain_config_rule(name='r')");
    useConfiguration("--experimental_enable_cc_toolchain_config_info");
    ConfiguredTarget target = getConfiguredTarget("//foo:r");
    assertThat(target).isNotNull();
    CcToolchainConfigInfo ccToolchainConfigInfo =
        (CcToolchainConfigInfo) target.get(CcToolchainConfigInfo.PROVIDER.getKey());
    ImmutableSet<String> featureNames =
        ccToolchainConfigInfo.getFeatures().stream()
            .map(feature -> feature.getName())
            .collect(ImmutableSet.toImmutableSet());
    ImmutableSet<String> actionConfigNames =
        ccToolchainConfigInfo.getActionConfigs().stream()
            .map(actionConfig -> actionConfig.getActionName())
            .collect(ImmutableSet.toImmutableSet());
    assertThat(featureNames).containsExactly("no_legacy_features", "custom_feature");
    assertThat(actionConfigNames).containsExactly("custom_action");
  }

  @Test
  public void testCcToolchainInfoFromSkylarkWithLegacyFeatures() throws Exception {
    loadCcToolchainConfigLib();
    scratch.file(
        "foo/crosstool.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "        'feature',",
        "        'action_config',",
        "        'artifact_name_pattern',",
        "        'env_entry',",
        "        'variable_with_value',",
        "        'make_variable',",
        "        'feature_set',",
        "        'with_feature_set',",
        "        'env_set',",
        "        'flag_group',",
        "        'flag_set',",
        "        'tool_path',",
        "        'tool')",
        "",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "                ctx = ctx,",
        "                features = [",
        "                    feature(name = 'custom_feature'),",
        "                    feature(name = 'legacy_compile_flags'),",
        "                    feature(name = 'fdo_optimize'),",
        "                    feature(name = 'default_compile_flags'),",
        "                ],",
        "                action_configs = [action_config(action_name = 'custom-action')],",
        "                artifact_name_patterns = [artifact_name_pattern(",
        "                   category_name = 'static_library',",
        "                   prefix = 'prefix',",
        "                   extension = '.a')],",
        "                toolchain_identifier = 'toolchain',",
        "                host_system_name = 'host',",
        "                target_system_name = 'target',",
        "                target_cpu = 'cpu',",
        "                target_libc = 'libc',",
        "                compiler = 'compiler',",
        "                abi_libc_version = 'abi_libc',",
        "                abi_version = 'abi')",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo], ",
        ")");

    scratch.file(
        "foo/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "cc_toolchain_alias(name='alias')",
        "cc_toolchain_config_rule(name='r')");
    useConfiguration("--experimental_enable_cc_toolchain_config_info");
    ConfiguredTarget target = getConfiguredTarget("//foo:r");
    assertThat(target).isNotNull();
    CcToolchainConfigInfo ccToolchainConfigInfo =
        (CcToolchainConfigInfo) target.get(CcToolchainConfigInfo.PROVIDER.getKey());
    ImmutableList<String> featureNames =
        ccToolchainConfigInfo.getFeatures().stream()
            .map(feature -> feature.getName())
            .collect(ImmutableList.toImmutableList());
    ImmutableSet<String> actionConfigNames =
        ccToolchainConfigInfo.getActionConfigs().stream()
            .map(actionConfig -> actionConfig.getActionName())
            .collect(ImmutableSet.toImmutableSet());
    // fdo_optimize should not be re-added to the list of features by legacy behavior
    assertThat(featureNames).containsNoDuplicates();
    // legacy_compile_flags should appear first in the list of features, followed by
    // default_compile_flags.
    assertThat(featureNames)
        .containsAllOf(
            "legacy_compile_flags", "default_compile_flags", "custom_feature", "fdo_optimize")
        .inOrder();
    // assemble is one of the action_configs added as a legacy behavior, therefore it needs to be
    // prepended to the action configs defined by the user.
    assertThat(actionConfigNames).containsAllOf("assemble", "custom-action").inOrder();
  }

  @Test
  public void testCcToolchainInfoToProto() throws Exception {
    loadCcToolchainConfigLib();
    scratch.file(
        "foo/crosstool.bzl",
        "load('//tools/cpp:cc_toolchain_config_lib.bzl',",
        "        'feature',",
        "        'action_config',",
        "        'artifact_name_pattern',",
        "        'env_entry',",
        "        'variable_with_value',",
        "        'make_variable',",
        "        'feature_set',",
        "        'with_feature_set',",
        "        'env_set',",
        "        'flag_group',",
        "        'flag_set',",
        "        'tool_path',",
        "        'tool')",
        "",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "                ctx = ctx,",
        "                features = [",
        "                    feature(name = 'featureone', enabled = True),",
        "                    feature(name = 'sysroot',",
        "                        flag_sets = [",
        "                            flag_set(",
        "                                actions = ['assemble', 'preprocess-assemble'],",
        "                                flag_groups = [",
        "                                    flag_group(",
        "                                        expand_if_available= 'available',",
        "                                        expand_if_not_available= 'notavailable',",
        "                                        expand_if_true = 'true',",
        "                                        expand_if_false = 'false',",
        "                                        expand_if_equal = variable_with_value(",
        "                                            name = 'variable', value = 'value'),",
        "                                        iterate_over = 'iterate_over',",
        "                                        flag_groups = [",
        "                                            flag_group(flags = ",
        "                                                ['foo%{libraries_to_link}bar',",
        "                                                 '-D__DATE__=\"redacted\"'])],",
        "                                    ),",
        "                                    flag_group(flags = ['a', 'b'])],",
        "                        )],",
        "                        implies = ['imply2', 'imply1'],",
        "                        provides = ['provides2', 'provides1'],",
        "                        requires = [feature_set(features = ['r1', 'r2']),",
        "                                    feature_set(features = ['r3'])],",
        "                        env_sets = [",
        "                             env_set(actions = ['a1', 'a2'],",
        "                                     env_entries = [",
        "                                         env_entry(key = 'k1', value = 'v1'),",
        "                                         env_entry(key = 'k2', value = 'v2')],",
        "                                     with_features = [",
        "                                         with_feature_set(",
        "                                             features = ['f1', 'f2'],",
        "                                             not_features = ['nf1', 'nf2'])]),",
        "                             env_set(actions = ['a1', 'a2'])]",
        "                )],",
        "                action_configs = [",
        "                    action_config(action_name = 'action_one', enabled=True),",
        "                    action_config(",
        "                        action_name = 'action_two',",
        "                        tools = [tool(path = 'fake/path',",
        "                                      with_features = [",
        "                                          with_feature_set(features = ['a'],",
        "                                                           not_features=['b', 'c']),",
        "                                          with_feature_set(features=['b'])],",
        "                                      execution_requirements = ['a', 'b', 'c'])],",
        "                        implies = ['compiler_input_flags', 'compiler_output_flags'],",
        "                        flag_sets = [flag_set(flag_groups = [flag_group(flags = ['a'])]",
        "                                             )])],",
        "                artifact_name_patterns = [artifact_name_pattern(",
        "                   category_name = 'static_library',",
        "                   prefix = 'prefix',",
        "                   extension = '.a')],",
        "                cxx_builtin_include_directories = ['dir1', 'dir2', 'dir3'],",
        "                toolchain_identifier = 'toolchain',",
        "                host_system_name = 'host',",
        "                target_system_name = 'target',",
        "                target_cpu = 'cpu',",
        "                target_libc = 'libc',",
        "                compiler = 'compiler',",
        "                abi_libc_version = 'abi_libc',",
        "                abi_version = 'abi',",
        "                tool_paths = [tool_path(name = 'name1', path = 'path1')],",
        "                make_variables = [make_variable(name = 'variable', value = '--a -b -c')],",
        "                builtin_sysroot = 'sysroot',",
        "                cc_target_os = 'os',",
        "        )",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo], ",
        ")");

    scratch.file(
        "foo/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "cc_toolchain_alias(name='alias')",
        "cc_toolchain_config_rule(name='r')");

    useConfiguration("--experimental_enable_cc_toolchain_config_info");
    ConfiguredTarget target = getConfiguredTarget("//foo:r");
    assertThat(target).isNotNull();
    CcToolchainConfigInfo ccToolchainConfigInfo =
        (CcToolchainConfigInfo) target.get(CcToolchainConfigInfo.PROVIDER.getKey());
    assertThat(ccToolchainConfigInfo).isNotNull();
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(ccToolchainConfigInfo.getProto(), toolchainBuilder);
    CToolchain toolchain = toolchainBuilder.build();

    assertThat(toolchain.getCxxBuiltinIncludeDirectoryList())
        .containsExactly("dir1", "dir2", "dir3");
    assertThat(toolchain.getToolchainIdentifier()).isEqualTo("toolchain");
    assertThat(toolchain.getHostSystemName()).isEqualTo("host");
    assertThat(toolchain.getTargetSystemName()).isEqualTo("target");
    assertThat(toolchain.getTargetCpu()).isEqualTo("cpu");
    assertThat(toolchain.getTargetLibc()).isEqualTo("libc");
    assertThat(toolchain.getCompiler()).isEqualTo("compiler");
    assertThat(toolchain.getAbiLibcVersion()).isEqualTo("abi_libc");
    assertThat(toolchain.getAbiVersion()).isEqualTo("abi");
    ToolPath toolPath = Iterables.getOnlyElement(toolchain.getToolPathList());
    assertThat(toolPath.getName()).isEqualTo("name1");
    assertThat(toolPath.getPath()).isEqualTo("path1");
    MakeVariable makeVariable = Iterables.getOnlyElement(toolchain.getMakeVariableList());
    assertThat(makeVariable.getName()).isEqualTo("variable");
    assertThat(makeVariable.getValue()).isEqualTo("--a -b -c");
    assertThat(toolchain.getBuiltinSysroot()).isEqualTo("sysroot");
    assertThat(toolchain.getCcTargetOs()).isEqualTo("os");
    assertThat(
            toolchain.getFeatureList().stream()
                .map(feature -> feature.getName())
                .collect(ImmutableList.toImmutableList()))
        .containsExactly("featureone", "sysroot")
        .inOrder();

    CToolchain.Feature feature = toolchain.getFeature(1);
    assertThat(feature.getName()).isEqualTo("sysroot");
    assertThat(feature.getImpliesList()).containsExactly("imply2", "imply1").inOrder();
    assertThat(feature.getProvidesList()).containsExactly("provides2", "provides1").inOrder();
    assertThat(feature.getRequires(0).getFeatureList()).containsExactly("r1", "r2");
    assertThat(feature.getRequires(1).getFeatureList()).containsExactly("r3");
    assertThat(feature.getEnvSetCount()).isEqualTo(2);
    CToolchain.EnvSet envSet = feature.getEnvSet(0);
    assertThat(envSet.getActionList()).containsExactly("a1", "a2");
    assertThat(envSet.getEnvEntry(0).getKey()).isEqualTo("k1");
    assertThat(envSet.getEnvEntry(0).getValue()).isEqualTo("v1");
    assertThat(Iterables.getOnlyElement(envSet.getWithFeatureList()).getFeatureList())
        .containsExactly("f1", "f2");
    assertThat(Iterables.getOnlyElement(envSet.getWithFeatureList()).getNotFeatureList())
        .containsExactly("nf1", "nf2");
    CToolchain.FlagSet flagSet = Iterables.getOnlyElement(feature.getFlagSetList());
    assertThat(flagSet.getActionList()).containsExactly("assemble", "preprocess-assemble");
    assertThat(flagSet.getFlagGroupCount()).isEqualTo(2);
    CToolchain.FlagGroup flagGroup = flagSet.getFlagGroup(0);
    assertThat(flagGroup.getExpandIfAllAvailableList()).containsExactly("available");
    assertThat(flagGroup.getExpandIfNoneAvailableList()).containsExactly("notavailable");
    assertThat(flagGroup.getIterateOver()).isEqualTo("iterate_over");
    assertThat(flagGroup.getExpandIfTrue()).isEqualTo("true");
    assertThat(flagGroup.getExpandIfFalse()).isEqualTo("false");
    CToolchain.VariableWithValue variableWithValue = flagGroup.getExpandIfEqual();
    assertThat(variableWithValue.getVariable()).isEqualTo("variable");
    assertThat(variableWithValue.getValue()).isEqualTo("value");
    assertThat(flagGroup.getFlagGroup(0).getFlagList())
        .containsExactly("foo%{libraries_to_link}bar", "-D__DATE__=\"redacted\"")
        .inOrder();

    assertThat(
            toolchain.getActionConfigList().stream()
                .map(actionConfig -> actionConfig.getActionName())
                .collect(ImmutableList.toImmutableList()))
        .containsExactly("action_one", "action_two")
        .inOrder();
    CToolchain.ActionConfig actionConfig = toolchain.getActionConfig(1);
    assertThat(actionConfig.getImpliesList())
        .containsExactly("compiler_input_flags", "compiler_output_flags")
        .inOrder();
    CToolchain.Tool tool = Iterables.getOnlyElement(actionConfig.getToolList());
    assertThat(tool.getToolPath()).isEqualTo("fake/path");
    assertThat(tool.getExecutionRequirementList()).containsExactly("a", "b", "c");
    assertThat(tool.getWithFeature(0).getFeatureList()).containsExactly("a");
    assertThat(tool.getWithFeature(0).getNotFeatureList()).containsExactly("b", "c");
    // When we create an ActionConfig from a proto we put the action name as the only element of its
    // FlagSet.action lists. So when we convert back to proto, we need to remove it.
    assertThat(
            ccToolchainConfigInfo.getActionConfigs().stream()
                .filter(config -> config.getName().equals("action_two"))
                .findFirst()
                .get()
                .getFlagSets()
                .get(0)
                .getActions())
        .containsExactly("action_two");
    CToolchain.FlagSet actionConfigFlagSet =
        Iterables.getOnlyElement(actionConfig.getFlagSetList());
    assertThat(actionConfigFlagSet.getActionCount()).isEqualTo(0);
    assertThat(Iterables.getOnlyElement(toolchain.getArtifactNamePatternList()))
        .isEqualTo(
            CToolchain.ArtifactNamePattern.newBuilder()
                .setCategoryName("static_library")
                .setPrefix("prefix")
                .setExtension(".a")
                .build());
  }

  // Tests that default None values in non-required fields do not cause trouble while building
  // the CToolchain
  @Test
  public void testCcToolchainInfoToProtoBareMinimum() throws Exception {
    loadCcToolchainConfigLib();
    scratch.file(
        "foo/crosstool.bzl",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "                ctx = ctx,",
        "                toolchain_identifier = 'toolchain',",
        "                host_system_name = 'host',",
        "                target_system_name = 'target',",
        "                target_cpu = 'cpu',",
        "                target_libc = 'libc',",
        "                compiler = 'compiler',",
        "                abi_libc_version = 'abi_libc',",
        "                abi_version = 'abi',",
        "        )",
        "cc_toolchain_config_rule = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo], ",
        ")");

    scratch.file(
        "foo/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "cc_toolchain_alias(name='alias')",
        "cc_toolchain_config_rule(name='r')");

    useConfiguration("--experimental_enable_cc_toolchain_config_info");
    ConfiguredTarget target = getConfiguredTarget("//foo:r");
    assertThat(target).isNotNull();
    CcToolchainConfigInfo ccToolchainConfigInfo =
        (CcToolchainConfigInfo) target.get(CcToolchainConfigInfo.PROVIDER.getKey());
    assertThat(ccToolchainConfigInfo).isNotNull();
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(ccToolchainConfigInfo.getProto(), toolchainBuilder);
    assertNoEvents();
  }

  @Test
  public void testGetLegacyCcFlagsMakeVariable() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "make_variable {",
            "  name: 'CC_FLAGS'",
            "  value: '-test-cflag1 -testcflag2'",
            "}");
    useConfiguration("--incompatible_disable_genrule_cc_toolchain_dependency");

    loadCcToolchainConfigLib();
    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  cc_flags = cc_common.legacy_cc_flags_make_variable_do_not_use(",
        "      cc_toolchain = toolchain)",
        "  return struct(",
        "    cc_flags = cc_flags)",
        "cc_flags = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_cc_toolchain': attr.label(default=Label('//a:alias'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'cc_flags')",
        "cc_toolchain_alias(name='alias')",
        "cc_flags(name='r')");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    String ccFlags = (String) r.get("cc_flags");

    assertThat(ccFlags).isEqualTo("-test-cflag1 -testcflag2");
  }

  private boolean toolchainResolutionEnabled() throws Exception {
    scratch.file(
        "a/rule.bzl",
        "def _impl(ctx):",
        "  toolchain_resolution_enabled = cc_common.is_cc_toolchain_resolution_enabled_do_not_use(",
        "      ctx = ctx)",
        "  return struct(",
        "    toolchain_resolution_enabled = toolchain_resolution_enabled)",
        "toolchain_resolution_enabled = rule(",
        "  _impl,",
        ");");

    scratch.file(
        "a/BUILD",
        "load(':rule.bzl', 'toolchain_resolution_enabled')",
        "toolchain_resolution_enabled(name='r')");

    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked") // Use an extra variable in order to suppress the warning.
    boolean toolchainResolutionEnabled = (boolean) r.get("toolchain_resolution_enabled");
    return toolchainResolutionEnabled;
  }

  @Test
  public void testIsToolchainResolutionEnabled_disabled() throws Exception {
    useConfiguration("--incompatible_enable_cc_toolchain_resolution=false");

    assertThat(toolchainResolutionEnabled()).isFalse();
  }

  @Test
  public void testIsToolchainResolutionEnabled_enabled() throws Exception {
    useConfiguration("--incompatible_enable_cc_toolchain_resolution");

    assertThat(toolchainResolutionEnabled()).isTrue();
  }
}
