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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Common test code to test that C++ linking action is populated with the correct build variables.
 */
public class LinkBuildVariablesTestCase extends BuildViewTestCase {
  /** Name of the build variable for the sysroot path variable name. */
  public static final String SYSROOT_VARIABLE_NAME = "sysroot";

  private final AtomicInteger counter = new AtomicInteger(0);

  public enum LinkBuildVariables {
    /** Entries in the linker search path (usually set by -L flag) */
    LIBRARY_SEARCH_DIRECTORIES("library_search_directories"),
    /** Flags providing files to link as inputs in the linker invocation */
    LIBRARIES_TO_LINK("libraries_to_link"),
    /** Location of linker param file created by bazel to overcome command line length limit */
    LINKER_PARAM_FILE("linker_param_file"),
    /** execpath of the output of the linker. */
    OUTPUT_EXECPATH("output_execpath"),
    /** "yes"|"no" depending on whether interface library should be generated. */
    GENERATE_INTERFACE_LIBRARY("generate_interface_library"),
    /** Path to the interface library builder tool. */
    INTERFACE_LIBRARY_BUILDER("interface_library_builder_path"),
    /** Input for the interface library ifso builder tool. */
    INTERFACE_LIBRARY_INPUT("interface_library_input_path"),
    /** Path where to generate interface library using the ifso builder tool. */
    INTERFACE_LIBRARY_OUTPUT("interface_library_output_path"),
    /** Linker flags coming from the --linkopt or linkopts attribute. */
    USER_LINK_FLAGS("user_link_flags"),
    /** A build variable giving linkstamp paths. */
    LINKSTAMP_PATHS("linkstamp_paths"),
    /** Presence of this variable indicates that PIC code should be generated. */
    FORCE_PIC("force_pic"),
    /** Presence of this variable indicates that the debug symbols should be stripped. */
    STRIP_DEBUG_SYMBOLS("strip_debug_symbols"),
    /** Truthy when current action is a cc_test linking action, falsey otherwise. */
    IS_CC_TEST("is_cc_test"),
    /**
     * Presence of this variable indicates that files were compiled with fission (debug info is in
     * .dwo files instead of .o files and linker needs to know).
     */
    IS_USING_FISSION("is_using_fission");

    /** Path to the fdo instrument. */
    private final String variableName;

    LinkBuildVariables(String variableName) {
      this.variableName = variableName;
    }

    public String getVariableName() {
      return variableName;
    }
  }

  protected SpawnAction getCppLinkAction(ConfiguredTarget target, Link.LinkTargetType type) {
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
      case NODEPS_DYNAMIC_LIBRARY:
        linkerOutput = getBinArtifact("lib" + target.getLabel().getName() + ".so", target);
        break;
      case DYNAMIC_LIBRARY:
        linkerOutput = getBinArtifact(target.getLabel().getName(), target);
        break;
      case EXECUTABLE:
        linkerOutput = getExecutable(target);
        break;
      default:
        throw new IllegalArgumentException(
            String.format("Cannot get SpawnAction for link type %s", type));
    }
    return (SpawnAction) getGeneratingAction(linkerOutput);
  }

  protected LinkCommandLine getLinkCommandLine(SpawnAction cppLinkAction) {
    var commandLines = cppLinkAction.getCommandLines().unpack();
    assertThat(commandLines).hasSize(2);
    assertThat(commandLines.get(1).commandLine).isInstanceOf(LinkCommandLine.class);
    return (LinkCommandLine) commandLines.get(1).commandLine;
  }

  /** Returns active build variables for a link action of given type for given target. */
  protected CcToolchainVariables getLinkBuildVariables(
      ConfiguredTarget target, Link.LinkTargetType type) {
    return getLinkCommandLine(getCppLinkAction(target, type)).getBuildVariables();
  }

  /** Returns the value of a given sequence variable in context of the given Variables instance. */
  protected List<String> getSequenceVariableValue(CcToolchainVariables variables, String variable)
      throws Exception {
    FeatureConfiguration mockFeatureConfiguration =
        buildFeatures(
                "features = [feature(",
                "  name = 'a',",
                "  flag_sets = [flag_set(",
                "    actions = ['foo'],",
                "    flag_groups = [flag_group(",
                "      iterate_over = '" + variable + "',",
                "      flags = ['%{" + variable + "}'],",
                "    )],",
                "  )],",
                ")]")
            .getFeatureConfiguration(ImmutableSet.of("a"));
    return mockFeatureConfiguration.getCommandLine("foo", variables);
  }

  /** Returns the value of a given string variable in context of the given Variables instance. */
  protected String getVariableValue(CcToolchainVariables variables, String variable)
      throws Exception {
    FeatureConfiguration mockFeatureConfiguration =
        buildFeatures(
                "features = [feature(",
                "  name = 'a',",
                "  flag_sets = [flag_set(",
                "    actions = ['foo'],",
                "    flag_groups = [flag_group(",
                "      flags = ['%{" + variable + "}'],",
                "    )],",
                "  )],",
                ")]")
            .getFeatureConfiguration(ImmutableSet.of("a"));
    return Iterables.getOnlyElement(mockFeatureConfiguration.getCommandLine("foo", variables));
  }

  private void loadCcToolchainConfigLib() throws Exception {
    scratch.appendFile("tools/cpp/BUILD", "");
    scratch.overwriteFile(
        "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/cc_toolchain_config_lib.bzl"));
  }

  private CcToolchainFeatures buildFeatures(String... content) throws Exception {
    loadCcToolchainConfigLib();
    String packageName = "crosstool" + counter.incrementAndGet();
    scratch.overwriteFile(
        packageName + "/crosstool.bzl",
        "load(",
        "    '//tools/cpp:cc_toolchain_config_lib.bzl',",
        "    'feature',",
        "    'flag_group',",
        "    'flag_set',",
        ")",
        "load('@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl',"
            + " 'CcToolchainConfigInfo')",
        "load('@rules_cc//cc/common:cc_common.bzl', 'cc_common')",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "        ctx = ctx,",
        String.join("\n", content) + ",",
        "        toolchain_identifier = 'toolchain',",
        "        host_system_name = 'host',",
        "        target_system_name = 'target',",
        "        target_cpu = 'cpu',",
        "        target_libc = 'libc',",
        "        compiler = 'compiler',",
        "    )",
        "",
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
                    tools_directory = "crosstool",
                  ))]
        cc_toolchain_features = rule(_impl, attrs = {"config":attr.label()})
        """);
    scratch.overwriteFile(
        packageName + "/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "load('//bazel_internal/test_rules/cc:ctf_rule.bzl', 'cc_toolchain_features')",
        "cc_toolchain_features(name = 'f', config = ':r')",
        "cc_toolchain_config_rule(name = 'r')");

    ConfiguredTarget target = getConfiguredTarget("//" + packageName + ":f");
    assertThat(target).isNotNull();
    return (CcToolchainFeatures) getStarlarkProvider(target, "MyInfo").getValue("f");
  }
}
