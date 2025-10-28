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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.rules.cpp.CompileCommandLine}, for example testing
 * the ordering of individual command line flags, or that command line is emitted differently
 * subject to the presence of certain build variables. Also used to test migration logic (removing
 * hardcoded flags and expressing them using feature configuration.
 */
@RunWith(JUnit4.class)
public class CompileCommandLineTest extends BuildViewTestCase {

  @Before
  public void initializeRuleContext() throws Exception {
    scratch.file("foo/BUILD", "cc_library(name = 'foo')");
  }

  private void loadCcToolchainConfigLib() throws IOException {
    scratch.appendFile("tools/cpp/BUILD", "");
    scratch.overwriteFile(
        "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/cc_toolchain_config_lib.bzl"));
  }

  private CcToolchainFeatures getCcToolchainFeatures(String... starlark) throws Exception {
    loadCcToolchainConfigLib();
    scratch.overwriteFile(
        "mock_crosstool/crosstool.bzl",
        "load(",
        "    '//tools/cpp:cc_toolchain_config_lib.bzl',",
        "    'action_config',",
        "    'feature',",
        "    'flag_group',",
        "    'flag_set',",
        "    'tool',",
        ")",
        "load('@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl',"
            + " 'CcToolchainConfigInfo')",
        "load('@rules_cc//cc/common:cc_common.bzl', 'cc_common')",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "        ctx = ctx,",
        String.join("\n", starlark) + ",",
        "        toolchain_identifier = 'toolchain',",
        "        host_system_name = 'host',",
        "        target_system_name = 'target',",
        "        target_cpu = 'cpu',",
        "        target_libc = 'libc',",
        "        compiler = 'compiler',",
        "    )",
        "cc_toolchain_config_rule = rule(implementation = _impl, provides ="
            + " [CcToolchainConfigInfo])");
    scratch.overwriteFile("bazel_internal/test_rules/cc/BUILD");
    scratch.overwriteFile(
        "bazel_internal/test_rules/cc/ctf_rule.bzl",
        """
        load('@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl', 'CcToolchainConfigInfo')
        load('@rules_cc//cc/common:cc_common.bzl', 'cc_common')
        MyInfo = provider()
        def _impl(ctx):
          return [MyInfo(f = cc_common.cc_toolchain_features(
                    toolchain_config_info = ctx.attr.config[CcToolchainConfigInfo],
                    tools_directory = "",
                  ))]
        cc_toolchain_features = rule(_impl, attrs = {"config":attr.label()})
        """);
    scratch.overwriteFile(
        "mock_crosstool/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "load('//bazel_internal/test_rules/cc:ctf_rule.bzl', 'cc_toolchain_features')",
        "cc_toolchain_features(name = 'f', config = ':r')",
        "cc_toolchain_config_rule(name = 'r')");

    ConfiguredTarget target = getConfiguredTarget("//mock_crosstool:f");
    assertThat(target).isNotNull();
    return (CcToolchainFeatures) getStarlarkProvider(target, "MyInfo").getValue("f");
  }

  private FeatureConfiguration getMockFeatureConfigurationFromStarlark(String... starlark)
      throws Exception {
    return getCcToolchainFeatures(starlark)
        .getFeatureConfiguration(
            ImmutableSet.of(
                CppActionNames.ASSEMBLE,
                CppActionNames.PREPROCESS_ASSEMBLE,
                CppActionNames.C_COMPILE,
                CppActionNames.CPP_COMPILE,
                CppActionNames.CPP_HEADER_PARSING,
                CppActionNames.CPP_MODULE_CODEGEN,
                CppActionNames.CPP_MODULE_COMPILE));
  }

  @Test
  public void testFeatureConfigurationCommandLineIsUsed() throws Exception {
    CompileCommandLine compileCommandLine =
        makeCompileCommandLineBuilder()
            .setFeatureConfiguration(
                getMockFeatureConfigurationFromStarlark(
                    "action_configs = [",
                    "    action_config(",
                    "        action_name = 'c++-compile',",
                    "        implies = ['some_foo_feature'],",
                    "        tools = [tool(path = 'foo/bar/DUMMY_COMPILER')],",
                    "    ),",
                    "],",
                    "features = [",
                    "    feature(",
                    "        name = 'some_foo_feature',",
                    "        flag_sets = [",
                    "            flag_set(",
                    "                actions = ['c++-compile'],",
                    "                flag_groups = [flag_group(flags = ['-some_foo_flag'])],",
                    "            ),",
                    "        ],",
                    "    ),",
                    "]"))
            .build();
    assertThat(
            compileCommandLine.getArguments(
                /* parameterFilePath= */ null, /* overwrittenVariables= */ null, PathMapper.NOOP))
        .contains("-some_foo_flag");
  }

  @Test
  public void testUnfilteredFlagsAreNotFiltered() throws Exception {
    List<String> actualCommandLine =
        getCompileCommandLineWithCoptsFilter(CppRuleClasses.UNFILTERED_COMPILE_FLAGS_FEATURE_NAME);
    assertThat(actualCommandLine).contains("-i_am_a_flag");
  }

  @Test
  public void testNonUnfilteredFlagsAreFiltered() throws Exception {
    List<String> actualCommandLine = getCompileCommandLineWithCoptsFilter("filtered_flags");
    assertThat(actualCommandLine).doesNotContain("-i_am_a_flag");
  }

  private List<String> getCompileCommandLineWithCoptsFilter(String featureName) throws Exception {
    CompileCommandLine compileCommandLine =
        makeCompileCommandLineBuilder()
            .setFeatureConfiguration(
                getMockFeatureConfigurationFromStarlark(
                    "action_configs = [",
                    "    action_config(",
                    "        action_name = 'c++-compile',",
                    "        implies = ['" + featureName + "'],",
                    "        tools = [tool(path = 'foo/bar/DUMMY_COMPILER')],",
                    "    ),",
                    "],",
                    "features = [",
                    "    feature(",
                    "        name = '" + featureName + "',",
                    "        flag_sets = [",
                    "            flag_set(",
                    "                actions = ['c++-compile'],",
                    "                flag_groups = [flag_group(flags = ['-i_am_a_flag'])],",
                    "            ),",
                    "        ],",
                    "    ),",
                    "]"))
            .setCoptsFilter(CoptsFilter.fromRegex(Pattern.compile(".*i_am_a_flag.*")))
            .build();
    return compileCommandLine.getArguments(
        /* parameterFilePath= */ null, /* overwrittenVariables= */ null, PathMapper.NOOP);
  }

  private CompileCommandLine.Builder makeCompileCommandLineBuilder() throws Exception {
    return CompileCommandLine.builder(CoptsFilter.alwaysPasses(), "c++-compile");
  }
}
