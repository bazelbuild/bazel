// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.starlark.StarlarkExecGroupCollection;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Test for aspect automatic exec groups. */
@RunWith(TestParameterInjector.class)
public class AspectAutoExecGroupsTest extends BuildViewTestCase {
  /**
   * Sets up two toolchains types, each with a single toolchain implementation and a single
   * exec_compatible_with platform.
   *
   * <p>toolchain_type_1 -> foo_toolchain -> exec_compatible_with platform_1 toolchain_type_2 ->
   * bar_toolchain -> exec_compatible_with platform_2
   */
  public void createToolchainsAndPlatforms() throws Exception {
    scratch.overwriteFile(
        "rule/test_toolchain.bzl",
        """
def _impl(ctx):
    return [platform_common.ToolchainInfo(
        tool = ctx.executable._tool,
        files_to_run = ctx.attr._tool[DefaultInfo].files_to_run,
    )]

test_toolchain = rule(
    implementation = _impl,
    attrs = {
        "_tool": attr.label(default = "//toolchain:b_tool", executable = True, cfg = "exec"),
    },
)
""");
    scratch.overwriteFile(
        "rule/BUILD",
        """
        exports_files(["test_toolchain/bzl"])

        toolchain_type(name = "toolchain_type_1")

        toolchain_type(name = "toolchain_type_2")
        """);
    scratch.overwriteFile(
        "toolchain/BUILD",
        """
        load("//rule:test_toolchain.bzl", "test_toolchain")

        genrule(
            name = "a_tool",
            outs = ["atool"],
            cmd = "",
            executable = True,
        )

        genrule(
            name = "b_tool",
            outs = ["btool"],
            cmd = "",
            executable = True,
        )

        test_toolchain(
            name = "foo",
        )

        toolchain(
            name = "foo_toolchain",
            exec_compatible_with = ["//platforms:constraint_1"],
            target_compatible_with = ["//platforms:constraint_1"],
            toolchain = ":foo",
            toolchain_type = "//rule:toolchain_type_1",
        )

        test_toolchain(
            name = "bar",
        )

        toolchain(
            name = "bar_toolchain",
            exec_compatible_with = ["//platforms:constraint_2"],
            target_compatible_with = ["//platforms:constraint_1"],
            toolchain = ":bar",
            toolchain_type = "//rule:toolchain_type_2",
        )
        """);

    scratch.overwriteFile(
        "platforms/BUILD",
        """
        constraint_setting(name = "setting")

        constraint_value(
            name = "constraint_1",
            constraint_setting = ":setting",
        )

        constraint_value(
            name = "constraint_2",
            constraint_setting = ":setting",
        )

        platform(
            name = "platform_1",
            constraint_values = [":constraint_1"],
        )

        platform(
            name = "platform_2",
            constraint_values = [":constraint_2"],
            exec_properties = {
                "watermelon.ripeness": "unripe",
                "watermelon.color": "red",
            },
        )
        """);
  }

  @Before
  public void setup() throws Exception {
    useConfiguration();
  }

  @Override
  public void useConfiguration(String... args) throws Exception {
    // These need to be defined before the configuration is parsed.
    createToolchainsAndPlatforms();
    String[] flags = {
      "--extra_toolchains=//toolchain:foo_toolchain,//toolchain:bar_toolchain",
      "--platforms=//platforms:platform_1",
      "--extra_execution_platforms=//platforms:platform_1,//platforms:platform_2"
    };

    super.useConfiguration(ObjectArrays.concat(flags, args, String.class));
  }

  /**
   * Creates custom rule which produces action with {@code actionParameters}, adds {@code
   * extraAttributes}, defines {@code toolchains}, and adds custom exec groups from {@code
   * execGroups}. Depending on {@code actionRunCommand} parameter, {@code actions.run} or {@code
   * actions.run_shell} is created. This rule also defines an aspect on its {@code deps} attribute.
   */
  private void createCustomRule(
      String action,
      String actionParameters,
      String extraAttributes,
      String toolchains,
      String execGroups)
      throws Exception {
    scratch.file(
        "test/defs.bzl",
        "load('//test:aspect.bzl', 'custom_aspect')",
        "def _impl(ctx):",
        "  output_jar = ctx.actions.declare_file(ctx.label.name + '_dummy_output.jar')",
        "  " + action + "(",
        actionParameters,
        "    outputs = [output_jar],",
        action.equals("ctx.actions.run")
            ? (actionParameters.contains("executable =") // avoid adding executable parameter twice
                ? ""
                : "executable = ctx.toolchains['//rule:toolchain_type_1'].tool,")
            : "    command = 'echo',",
        "  )",
        "  return [DefaultInfo(files = depset([output_jar]))]",
        "custom_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    '_tool': attr.label(default = '//toolchain:a_tool', cfg = 'exec', executable = True),",
        "    'deps': attr.label_list(aspects = [custom_aspect]),",
        extraAttributes,
        "  },",
        "  exec_groups = {",
        execGroups,
        "  },",
        "  toolchains = " + toolchains + ",",
        ")");
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "custom_rule")

        custom_rule(name = "custom_rule_dep")

        custom_rule(
            name = "custom_rule_name",
            deps = ["custom_rule_dep"],
        )
        """);
  }

  /**
   * Creates custom aspect which produces action with `{@code actionParameters}, adds {@code
   * extraAttributes}, defines {@code toolchains}, and adds custom exec groups from {@code
   * execGroups}. Depending on {@code actionRunCommand} parameter, {@code actions.run} or {@code
   * actions.run_shell} is created.
   */
  private void createCustomAspect(
      String action,
      String actionParameters,
      String extraAttributes,
      String toolchains,
      String execGroups)
      throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "OutputFile = provider(fields = {'file': 'Output file', 'exec_groups': 'Exec groups'})",
        "def _impl(target, ctx):",
        "  output_jar = ctx.actions.declare_file(ctx.label.name + '_dummy_output_aspect.jar')",
        "  " + action + "(",
        actionParameters,
        "    outputs = [output_jar],",
        action.equals("ctx.actions.run")
            ? "    executable = '//toolchain:foo_toolchain',"
            : "    command = 'echo',",
        "  )",
        "  exec_groups = ctx.exec_groups",
        "  return [OutputFile(file = output_jar, exec_groups = exec_groups)]",
        "custom_aspect = aspect(",
        "  implementation = _impl,",
        "  attrs = {",
        "     ",
        extraAttributes,
        "  },",
        "  attr_aspects = ['deps'],",
        "  exec_groups = {",
        execGroups,
        "  },",
        "  toolchains = " + toolchains + ",",
        ")");
  }

  /**
   * Creates empty rule and custom aspect on rule's dependencies. Custom aspect produces action with
   * {@code actionParameters}, adds {@code extraAttributes}, defines {@code toolchains}, and adds
   * custom exec groups from {@code execGroups}. Depending on {@code action} parameter, {@code
   * actions.run} or {@code actions.run_shell} is created. This function is used only for testing
   * the aspect, not the rule.
   */
  private void createCustomAspectAndEmptyRule(
      String action,
      String actionParameters,
      String extraAttributes,
      String toolchains,
      String execGroups)
      throws Exception {
    scratch.file(
        "test/aspect.bzl",
        "OutputFile = provider(fields = {'file': 'Output file', 'exec_groups': 'Exec groups'})",
        "def _impl(target, ctx):",
        "  output_jar = ctx.actions.declare_file(ctx.label.name + '_dummy_output_aspect.jar')",
        "  " + action + "(",
        actionParameters,
        "    outputs = [output_jar],",
        action.equals("ctx.actions.run")
            ? "    executable = '//toolchain:foo_toolchain',"
            : "    command = 'echo',",
        "  )",
        "  exec_groups = ctx.exec_groups",
        "  return [OutputFile(file = output_jar, exec_groups = exec_groups)]",
        "custom_aspect = aspect(",
        "  implementation = _impl,",
        "  attrs = {",
        "     ",
        extraAttributes,
        "  },",
        "  attr_aspects = ['deps'],",
        "  exec_groups = {",
        execGroups,
        "  },",
        "  toolchains = " + toolchains + ",",
        ")");
    scratch.file(
        "test/defs.bzl",
        """
        load("//test:aspect.bzl", "custom_aspect")

        def _impl(ctx):
            return []

        custom_rule = rule(
            implementation = _impl,
            attrs = {
                "deps": attr.label_list(aspects = [custom_aspect]),
            },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:defs.bzl", "custom_rule")

        custom_rule(name = "custom_rule_dep")

        custom_rule(
            name = "custom_rule_name",
            deps = ["custom_rule_dep"],
        )
        """);
  }

  private StarlarkExecGroupCollection getExecGroupsFromAspectProvider(
      ConfiguredAspect configuredAspect) throws Exception {
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(Label.parseCanonical("//test:aspect.bzl"), "OutputFile");
    StarlarkInfo keyInfo = (StarlarkInfo) configuredAspect.get(key);
    return (StarlarkExecGroupCollection) keyInfo.getValue("exec_groups");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void automaticExecutionGroups_disabledAndAttributeFalse_disabled(String action)
      throws Exception {
    createCustomAspectAndEmptyRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = False),",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");

    getConfiguredTarget("//test:custom_rule_name");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    StarlarkExecGroupCollection aspectExecGroups =
        getExecGroupsFromAspectProvider(configuredAspect);

    assertThat(aspectExecGroups.getToolchainCollectionForTesting().keySet())
        .containsExactly("default-exec-group");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void automaticExecutionGroups_disabledAndAttributeTrue_enabled(String action)
      throws Exception {
    createCustomAspectAndEmptyRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = True),",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");

    getConfiguredTarget("//test:custom_rule_name");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    StarlarkExecGroupCollection aspectExecGroups =
        getExecGroupsFromAspectProvider(configuredAspect);

    assertThat(aspectExecGroups.getToolchainCollectionForTesting().keySet())
        .containsExactly("default-exec-group", "//rule:toolchain_type_1");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void automaticExecutionGroups_disabledAndAttributeNotSet_disabled(String action)
      throws Exception {
    createCustomAspectAndEmptyRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");
    useConfiguration("--incompatible_auto_exec_groups=False");

    getConfiguredTarget("//test:custom_rule_name");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    StarlarkExecGroupCollection aspectExecGroups =
        getExecGroupsFromAspectProvider(configuredAspect);

    assertThat(aspectExecGroups.getToolchainCollectionForTesting().keySet())
        .containsExactly("default-exec-group");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void automaticExecutionGroups_enabledAndAttributeFalse_disabled(String action)
      throws Exception {
    createCustomAspectAndEmptyRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = False),",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");
    useConfiguration("--incompatible_auto_exec_groups");

    getConfiguredTarget("//test:custom_rule_name");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    StarlarkExecGroupCollection aspectExecGroups =
        getExecGroupsFromAspectProvider(configuredAspect);

    assertThat(aspectExecGroups.getToolchainCollectionForTesting().keySet())
        .containsExactly("default-exec-group");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void automaticExecutionGroups_enabledAndAttributeTrue_enabled(String action)
      throws Exception {
    createCustomAspectAndEmptyRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "'_use_auto_exec_groups': attr.bool(default = True),",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");
    useConfiguration("--incompatible_auto_exec_groups");

    getConfiguredTarget("//test:custom_rule_name");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    StarlarkExecGroupCollection aspectExecGroups =
        getExecGroupsFromAspectProvider(configuredAspect);

    assertThat(aspectExecGroups.getToolchainCollectionForTesting().keySet())
        .containsExactly("default-exec-group", "//rule:toolchain_type_1");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void automaticExecutionGroups_enabledAndAttributeNotSet_enabled(String action)
      throws Exception {
    createCustomAspectAndEmptyRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");
    useConfiguration("--incompatible_auto_exec_groups");

    getConfiguredTarget("//test:custom_rule_name");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    StarlarkExecGroupCollection aspectExecGroups =
        getExecGroupsFromAspectProvider(configuredAspect);

    assertThat(aspectExecGroups.getToolchainCollectionForTesting().keySet())
        .containsExactly("default-exec-group", "//rule:toolchain_type_1");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void execGroups_customAspectOnCustomRule(String action) throws Exception {
    String customExecGroups =
        "    'aspect_custom_exec_group': exec_group(\n"
            + "      exec_compatible_with = ['//platforms:constraint_1'],\n"
            + "      toolchains = ['//rule:toolchain_type_1'],\n"
            + "    ),\n";
    createCustomAspect(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_2',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_2']",
        /* execGroups= */ customExecGroups);
    createCustomRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");
    useConfiguration("--incompatible_auto_exec_groups");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ConfiguredTarget targetDep = (ConfiguredTarget) getRuleContext(target).getPrerequisite("deps");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    StarlarkExecGroupCollection aspectExecGroups =
        getExecGroupsFromAspectProvider(configuredAspect);

    assertThat(getRuleContext(target).getExecGroups().execGroups().keySet())
        .containsExactly("//rule:toolchain_type_1");
    assertThat(getRuleContext(targetDep).getExecGroups().execGroups().keySet())
        .containsExactly("//rule:toolchain_type_1");
    assertThat(aspectExecGroups.getToolchainCollectionForTesting().keySet())
        .containsExactly(
            "//rule:toolchain_type_2", "default-exec-group", "aspect_custom_exec_group");
  }

  @Test
  @TestParameters({
    "{action: ctx.actions.run}",
    "{action: ctx.actions.run_shell}",
  })
  public void execPlatforms_customAspectOnCustomRule(String action) throws Exception {
    String customExecGroups =
        "    'aspect_custom_exec_group': exec_group(\n"
            + "      exec_compatible_with = ['//platforms:constraint_1'],\n"
            + "      toolchains = ['//rule:toolchain_type_1'],\n"
            + "    ),\n";
    createCustomAspect(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_2',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_2']",
        /* execGroups= */ customExecGroups);
    createCustomRule(
        /* action= */ action,
        /* actionParameters= */ "toolchain = '//rule:toolchain_type_1',",
        /* extraAttributes= */ "",
        /* toolchains= */ "['//rule:toolchain_type_1']",
        /* execGroups= */ "");
    useConfiguration("--incompatible_auto_exec_groups");

    ConfiguredTarget target = getConfiguredTarget("//test:custom_rule_name");
    ConfiguredTarget targetDep = (ConfiguredTarget) getRuleContext(target).getPrerequisite("deps");
    ConfiguredAspect configuredAspect = getAspect("//test:aspect.bzl%custom_aspect");
    Action targetAction = getGeneratingAction(target, "test/custom_rule_name_dummy_output.jar");
    Action targetDepAction =
        getGeneratingAction(targetDep, "test/custom_rule_dep_dummy_output.jar");
    Action aspectAction = (Action) configuredAspect.getActions().get(0);

    assertThat(targetAction.getOwner().getExecutionPlatform().label())
        .isEqualTo(Label.parseCanonical("//platforms:platform_1"));
    assertThat(targetDepAction.getOwner().getExecutionPlatform().label())
        .isEqualTo(Label.parseCanonical("//platforms:platform_1"));
    assertThat(aspectAction.getOwner().getExecutionPlatform().label())
        .isEqualTo(Label.parseCanonical("//platforms:platform_2"));
  }
}
