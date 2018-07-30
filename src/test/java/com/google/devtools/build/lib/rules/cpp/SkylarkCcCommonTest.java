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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LibraryToLinkApi;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.testutil.TestConstants;
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
  public void setConfiguration() throws Exception {
    useConfiguration("--experimental_enable_cc_skylark_api");
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
        .isEqualTo(
            featureConfiguration
                .getToolForAction(CppActionNames.CPP_COMPILE)
                .getToolPathFragment()
                .getPathString());
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
    useConfiguration("--features=foo_feature", "--experimental_enable_cc_skylark_api");
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
        "    'CC_FLAGS_MAKE_VARIABLE_ACTION_NAME_ACTION_NAME',",
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
        "      cc_flags_make_variable_action_name_action_name=CC_FLAGS_MAKE_VARIABLE_ACTION_NAME_ACTION_NAME,",
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
  public void testEmptyLinkVariables() throws Exception {
    useConfiguration("--linkopt=-foo");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
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
    useConfiguration("--linkopt=-pie", "--experimental_enable_cc_skylark_api");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                0,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "is_linking_dynamic_library = True,",
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
                ")"))
        .contains("-pie");
  }

  @Test
  public void testIsUsingLinkerLinkVariables() throws Exception {
    useConfiguration(
        "--linkopt=-i_dont_want_to_see_this_on_archiver_command_line",
        "--experimental_enable_cc_skylark_api");
    assertThat(
            commandLineForVariables(
                CppActionNames.CPP_LINK_EXECUTABLE,
                0,
                "cc_common.create_link_variables(",
                "feature_configuration = feature_configuration,",
                "cc_toolchain = toolchain,",
                "is_using_linker = True,",
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
    @SuppressWarnings("unchecked")
    SkylarkList<String> result = (SkylarkList<String>) r.get("command_line");
    return result;
  }

  @Test
  public void testCcCompilationProvider() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//tools/build_defs/cc:rule.bzl', 'crule')",
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
        "    defines = ['DEP1'],",
        ")",
        "cc_library(",
        "    name = 'dep2',",
        "    srcs = ['dep2.cc'],",
        "    hdrs = ['dep2.h'],",
        "    includes = ['dep2/qux'],",
        "    defines = ['DEP2'],",
        ")",
        "crule(name='r')");
    scratch.file("tools/build_defs/cc/BUILD", "");
    scratch.file(
        "tools/build_defs/cc/rule.bzl",
        "def _impl(ctx):",
        "  cc_compilation_info = CcCompilationInfo(headers=depset([ctx.file._header]),",
        "    system_includes=depset([ctx.attr._include]), defines=depset([ctx.attr._define]))",
        "  cc_compilation_infos = [cc_compilation_info]",
        "  for dep in ctx.attr._deps:",
        "      cc_compilation_infos.append(dep[CcCompilationInfo])",
        "  merged_cc_compilation_info=cc_common.merge_cc_compilation_infos(",
        "      cc_compilation_infos=cc_compilation_infos)",
        "  return struct(",
        "    providers=[cc_compilation_info, cc_common.create_cc_skylark_info(ctx=ctx)],",
        "    merged_headers=merged_cc_compilation_info.headers,",
        "    merged_system_includes=merged_cc_compilation_info.system_includes,",
        "    merged_defines=merged_cc_compilation_info.defines",
        "  )",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    '_header': attr.label(allow_single_file=True, default=Label('//a:header.h')),",
        "    '_include': attr.string(default='foo/bar'),",
        "    '_define': attr.string(default='MYDEFINE'),",
        "    '_deps': attr.label_list(default=['//a:dep1', '//a:dep2'])",
        "  },",
        "  fragments = ['cpp'],",
        ");");

    ConfiguredTarget lib = getConfiguredTarget("//a:lib");
    @SuppressWarnings("unchecked")
    CcCompilationContext ccCompilationContext =
        lib.get(CcCompilationInfo.PROVIDER).getCcCompilationContext();
    assertThat(
            ccCompilationContext
                .getDeclaredIncludeSrcs()
                .toCollection()
                .stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .containsExactly("lib.h", "header.h");

    ConfiguredTarget r = getConfiguredTarget("//a:r");

    List<Artifact> mergedHeaders =
        ((SkylarkNestedSet) r.get("merged_headers")).getSet(Artifact.class).toList();
    assertThat(
            mergedHeaders
                .stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("header.h", "dep1.h", "dep2.h");

    List<String> mergedDefines =
        ((SkylarkNestedSet) r.get("merged_defines")).getSet(String.class).toList();
    assertThat(mergedDefines).containsAllOf("MYDEFINE", "DEP1", "DEP2");

    List<String> mergedIncludes =
        ((SkylarkNestedSet) r.get("merged_system_includes")).getSet(String.class).toList();
    assertThat(mergedIncludes).containsAllOf("foo/bar", "a/dep1/baz", "a/dep2/qux");
  }

  @Test
  public void testLibraryLinkerInputs() throws Exception {
    scratch.file("a/BUILD", "load('//tools/build_defs/cc:rule.bzl', 'crule')", "crule(name='r')");
    scratch.file("a/lib.a", "");
    scratch.file("a/lib.lo", "");
    scratch.file("a/lib.so", "");
    scratch.file("a/lib.ifso", "");
    scratch.file("tools/build_defs/cc/BUILD", "");
    scratch.file(
        "tools/build_defs/cc/rule.bzl",
        "def _create(ctx, lib, c):",
        "  return cc_common.create_library_to_link(ctx=ctx, library=lib, artifact_category=c)",
        "def _impl(ctx):",
        "  static_library = _create(ctx, ctx.file.liba, 'static_library')",
        "  alwayslink_static_library = _create(ctx, ctx.file.liblo, 'alwayslink_static_library')",
        "  dynamic_library = _create(ctx, ctx.file.libso, 'dynamic_library')",
        "  interface_library = _create(ctx, ctx.file.libifso, 'interface_library')",
        "  toolchain = ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]",
        "  symlink_library = cc_common.create_symlink_library_to_link(",
        "      ctx=ctx, cc_toolchain=toolchain, library=ctx.file.libso)",
        "  return struct(",
        "    static_library = static_library,",
        "    alwayslink_static_library = alwayslink_static_library,",
        "    dynamic_library = dynamic_library,",
        "    interface_library = interface_library,",
        "    symlink_library = symlink_library",
        "  )",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    'liba': attr.label(default='//a:lib.a', allow_single_file=True),",
        "    'liblo': attr.label(default='//a:lib.lo', allow_single_file=True),",
        "    'libso': attr.label(default='//a:lib.so', allow_single_file=True),",
        "    'libifso': attr.label(default='//a:lib.ifso', allow_single_file=True),",
        "     '_cc_toolchain': attr.label(default =",
        "         configuration_field(fragment = 'cpp', name = 'cc_toolchain'))",
        "  },",
        "  fragments = ['cpp'],",
        ");");
    ConfiguredTarget r = getConfiguredTarget("//a:r");
    @SuppressWarnings("unchecked")
    LibraryToLink staticLibrary = (LibraryToLink) r.get("static_library");
    assertThat(staticLibrary.getArtifact().getFilename()).isEqualTo("lib.a");
    LibraryToLink alwaysLinkStaticLibrary = (LibraryToLink) r.get("alwayslink_static_library");
    assertThat(alwaysLinkStaticLibrary.getArtifact().getFilename()).isEqualTo("lib.lo");
    LibraryToLink dynamicLibrary = (LibraryToLink) r.get("dynamic_library");
    assertThat(dynamicLibrary.getArtifact().getFilename()).isEqualTo("lib.so");
    LibraryToLink interfaceLibrary = (LibraryToLink) r.get("interface_library");
    assertThat(interfaceLibrary.getArtifact().getFilename()).isEqualTo("lib.ifso");
    LibraryToLink symlinkLibrary = (LibraryToLink) r.get("symlink_library");
    assertThat(symlinkLibrary.getArtifact().getFilename()).isEqualTo("lib.so");
    assertThat(symlinkLibrary.getArtifact().getPath())
        .isNotEqualTo(symlinkLibrary.getOriginalLibraryArtifact().getPath());
  }

  @Test
  public void testLibraryLinkerInputArtifactCategoryError() throws Exception {
    scratch.file("a/BUILD", "load('//tools/build_defs/cc:rule.bzl', 'crule')", "crule(name='r')");
    scratch.file("a/lib.a", "");
    scratch.file("tools/build_defs/cc/BUILD", "");
    scratch.file(
        "tools/build_defs/cc/rule.bzl",
        "def _impl(ctx):",
        "  executable = cc_common.create_library_to_link(",
        "    ctx=ctx, library=ctx.file.lib, artifact_category='executable')",
        "  return struct(",
        "    executable = executable,",
        "  )",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    'lib': attr.label(default='//a:lib.a', allow_single_file=True),",
        "  },",
        "  fragments = ['cpp'],",
        ");");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:r");
    assertContainsEvent(
        "Possible values for artifact_category: static_library, "
            + "alwayslink_static_library, dynamic_library, interface_library");
  }

  @Test
  public void testCcLinkingProviderParamsWithoutFlag() throws Exception {
    useConfiguration("--noexperimental_enable_cc_skylark_api");
    setUpCcLinkingProviderParamsTest();
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:r");
    assertContainsEvent(
        "Pass --experimental_enable_cc_skylark_api in order to "
            + "use the C++ API. Beware that we will be making breaking changes to this API "
            + "without prior warning.");
  }

  @Test
  public void testCcLinkingProvider() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            "supports_interface_shared_objects: false");
    useConfiguration("--experimental_enable_cc_skylark_api");
    setUpCcLinkingProviderParamsTest();
    ConfiguredTarget r = getConfiguredTarget("//a:r");

    List<String> staticSharedLinkopts =
        ((SkylarkNestedSet) r.get("sd_linkopts")).getSet(String.class).toList();
    List<String> staticNoSharedLinkopts =
        ((SkylarkNestedSet) r.get("se_linkopts")).getSet(String.class).toList();
    List<String> noStaticSharedLinkopts =
        ((SkylarkNestedSet) r.get("dd_linkopts")).getSet(String.class).toList();
    List<String> noStaticNoSharedLinkopts =
        ((SkylarkNestedSet) r.get("de_linkopts")).getSet(String.class).toList();
    assertThat(staticSharedLinkopts)
        .containsAllOf("-static_for_dynamic", "-DEP1_LINKOPT", "-DEP2_LINKOPT");
    assertThat(staticNoSharedLinkopts)
        .containsAllOf("-static_for_executable", "-DEP1_LINKOPT", "-DEP2_LINKOPT");
    assertThat(noStaticSharedLinkopts)
        .containsAllOf("-dynamic_for_dynamic", "-DEP1_LINKOPT", "-DEP2_LINKOPT");
    assertThat(noStaticNoSharedLinkopts)
        .containsAllOf("-dynamic_for_executable", "-DEP1_LINKOPT", "-DEP2_LINKOPT");

    List<LibraryToLink> staticSharedLibs =
        ((SkylarkNestedSet) r.get("sd_libs")).getSet(LibraryToLink.class).toList();
    List<LibraryToLink> staticNoSharedLibs =
        ((SkylarkNestedSet) r.get("se_libs")).getSet(LibraryToLink.class).toList();
    List<LibraryToLink> noStaticSharedLibs =
        ((SkylarkNestedSet) r.get("dd_libs")).getSet(LibraryToLink.class).toList();
    List<LibraryToLink> noStaticNoSharedLibs =
        ((SkylarkNestedSet) r.get("de_libs")).getSet(LibraryToLink.class).toList();
    assertThat(
            staticSharedLibs
                .stream()
                .map(x -> x.getOriginalLibraryArtifact().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("lib.a", "libdep1.a", "libdep2.a");
    assertThat(
            staticNoSharedLibs
                .stream()
                .map(x -> x.getOriginalLibraryArtifact().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("lib.a", "libdep1.a", "libdep2.a");

    assertThat(
            noStaticSharedLibs
                .stream()
                .map(x -> x.getOriginalLibraryArtifact().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("lib.so", "libdep1.so", "libdep2.so");
    assertThat(
            noStaticNoSharedLibs
                .stream()
                .map(x -> x.getOriginalLibraryArtifact().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("lib.so", "libdep1.so", "libdep2.so");

    List<Artifact> staticSharedRunLibs =
        ((SkylarkNestedSet) r.get("sd_runlibs")).getSet(Artifact.class).toList();
    List<Artifact> staticNoSharedRunLibs =
        ((SkylarkNestedSet) r.get("se_runlibs")).getSet(Artifact.class).toList();
    List<Artifact> noStaticSharedRunLibs =
        ((SkylarkNestedSet) r.get("dd_runlibs")).getSet(Artifact.class).toList();
    List<Artifact> noStaticNoSharedRunLibs =
        ((SkylarkNestedSet) r.get("de_runlibs")).getSet(Artifact.class).toList();
    assertThat(
            staticSharedRunLibs
                .stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .isEmpty();
    assertThat(
            staticNoSharedRunLibs
                .stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .isEmpty();
    assertThat(
            noStaticSharedRunLibs
                .stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("lib.so", "liba_Slibdep1.so", "liba_Slibdep2.so");
    assertThat(
            noStaticNoSharedRunLibs
                .stream()
                .map(Artifact::getFilename)
                .collect(ImmutableList.toImmutableList()))
        .containsAllOf("lib.so", "liba_Slibdep1.so", "liba_Slibdep2.so");
  }

  private void setUpCcLinkingProviderParamsTest() throws Exception {
    scratch.file(
        "a/BUILD",
        "load('//tools/build_defs/cc:rule.bzl', 'crule')",
        "crule(name='r')",
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
        "def _create(ctx, l, r, f):",
        "  return cc_common.create_cc_link_params(",
        "    ctx=ctx, libraries_to_link=depset(l), dynamic_libraries_for_runtime=depset(r),",
        "        user_link_flags=depset(f))",
        "def _impl(ctx):",
        "  liba = cc_common.create_library_to_link(ctx=ctx, library=ctx.file.liba,",
        "    artifact_category='static_library')",
        "  libso = cc_common.create_library_to_link(ctx=ctx, library=ctx.file.libso,",
        "    artifact_category='dynamic_library')",
        "  sd = _create(ctx, [liba], [], ['-static_for_dynamic'])",
        "  se = _create(ctx, [liba], [], ['-static_for_executable'])",
        "  dd = _create(ctx, [libso], [ctx.file.libso], ['-dynamic_for_dynamic'])",
        "  de = _create(ctx, [libso], [ctx.file.libso], ['-dynamic_for_executable'])",
        "  cc_linking_info = CcLinkingInfo(",
        "    static_mode_params_for_dynamic_library=sd, static_mode_params_for_executable=se,",
        "    dynamic_mode_params_for_dynamic_library=dd, dynamic_mode_params_for_executable=de)",
        "  cc_linking_infos = [cc_linking_info]",
        "  for dep in ctx.attr._deps:",
        "      cc_linking_infos.append(dep[CcLinkingInfo])",
        "  merged_cc_linking_info = cc_common.merge_cc_linking_infos(",
        "      cc_linking_infos=cc_linking_infos)",
        "  sd = merged_cc_linking_info.static_mode_params_for_dynamic_library",
        "  se = merged_cc_linking_info.static_mode_params_for_executable",
        "  dd = merged_cc_linking_info.dynamic_mode_params_for_dynamic_library",
        "  de = merged_cc_linking_info.dynamic_mode_params_for_executable",
        "  return struct(",
        "    cc_linking_info = cc_linking_info,",
        "    sd_linkopts = sd.linkopts,",
        "    se_linkopts = se.linkopts,",
        "    dd_linkopts = dd.linkopts,",
        "    de_linkopts = de.linkopts,",
        "    sd_libs = sd.libraries_to_link,",
        "    se_libs = se.libraries_to_link,",
        "    dd_libs = dd.libraries_to_link,",
        "    de_libs = de.libraries_to_link,",
        "    sd_runlibs = sd.dynamic_libraries_for_runtime,",
        "    se_runlibs = se.dynamic_libraries_for_runtime,",
        "    dd_runlibs = dd.dynamic_libraries_for_runtime,",
        "    de_runlibs = de.dynamic_libraries_for_runtime,",
        "  )",
        "crule = rule(",
        "  _impl,",
        "  attrs = { ",
        "    'liba': attr.label(default='//a:lib.a', allow_single_file=True),",
        "    'libso': attr.label(default='//a:lib.so', allow_single_file=True),",
        "    '_deps': attr.label_list(default=['//a:dep1', '//a:dep2']),",
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
  public void testWhitelist() throws Exception {
    SkylarkCcCommonTestHelper.createFiles(scratch, "foo/bar");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//foo:bin");
    assertContainsEvent("C++ Skylark API is for the time being");
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
    SkylarkList<LibraryToLinkApi> libraries =
        (SkylarkList<LibraryToLinkApi>) target.get("libraries");
    assertThat(
            libraries
                .stream()
                .map(x -> x.getOriginalLibraryArtifact().getFilename())
                .collect(ImmutableList.toImmutableList()))
        .contains("libskylark_lib.so");
  }

  @Test
  public void testLinkopts() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingLinking(
        scratch, "tools/build_defs/foo", "linkopts=depset(['-LINKING_OPTION'])");
    ConfiguredTarget target = getConfiguredTarget("//foo:skylark_lib");
    assertThat(target).isNotNull();
    CppLinkAction action =
        (CppLinkAction) getGeneratingAction(artifactByPath(getFilesToBuild(target), ".so"));
    assertThat(action.getArguments()).contains("-LINKING_OPTION");
  }

  @Test
  public void testSettingDynamicLibraryArtifact() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingLinking(
        scratch,
        "tools/build_defs/foo",
        "dynamic_library=ctx.actions.declare_file('dynamic_lib_artifact.so')");
    assertThat(getConfiguredTarget("//foo:skylark_lib")).isNotNull();
    ConfiguredTarget target = getConfiguredTarget("//foo:skylark_lib");
    assertThat(
            getFilesToBuild(target)
                .toCollection()
                .stream()
                .map(x -> x.getFilename())
                .collect(ImmutableList.toImmutableList()))
        .contains("dynamic_lib_artifact.so");
  }

  @Test
  public void testCcLinkingInfos() throws Exception {
    SkylarkCcCommonTestHelper.createFilesForTestingLinking(
        scratch, "tools/build_defs/foo", "cc_linking_infos=dep_cc_linking_infos");
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
}
