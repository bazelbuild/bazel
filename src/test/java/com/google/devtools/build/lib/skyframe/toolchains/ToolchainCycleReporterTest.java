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
package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.toolchains.SingleToolchainResolutionValue.SingleToolchainResolutionKey;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link AbstractToolchainCycleReporter}, {@link
 * RegisteredExecutionPlatformsCycleReporter}, and {@link RegisteredToolchainsCycleReporter}.
 */
@RunWith(JUnit4.class)
public final class ToolchainCycleReporterTest extends ToolchainTestCase {

  @Test
  public void registeredExecutionPlatformsCycleReporter_reportsCycle() {
    ConfiguredTargetKey platformKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//platforms:bad_platform"))
            .setConfiguration(targetConfig)
            .build();
    ConfiguredTargetKey depKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//platforms:dep"))
            .setConfiguration(targetConfig)
            .build();
    AspectDescriptor baseAspectDescriptor =
        AspectDescriptor.of(
            new StarlarkAspectClass(
                keyForBuild(Label.parseCanonicalUnchecked("//aspects:defs.bzl")), "base_aspect"),
            AspectParameters.EMPTY);
    AspectDescriptor topAspectDescriptor =
        AspectDescriptor.of(
            new StarlarkAspectClass(
                keyForBuild(Label.parseCanonicalUnchecked("//aspects:defs.bzl")), "top_aspect"),
            AspectParameters.EMPTY);
    AspectKey simpleAspectKey = AspectKeyCreator.createAspectKey(baseAspectDescriptor, depKey);
    AspectKey aspectKeyWithBaseAspects =
        AspectKeyCreator.createAspectKey(
            topAspectDescriptor, ImmutableList.of(simpleAspectKey), depKey);
    SkyKey registeredExecPlatformsKey =
        RegisteredExecutionPlatformsValue.key(targetConfigKey, /* debug= */ false);
    SkyKey registeredToolchainsKey =
        RegisteredToolchainsValue.key(targetConfigKey, /* debug= */ false);
    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType)
            .build();
    SingleToolchainResolutionKey singleToolchainResolutionKey =
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            depKey,
            ImmutableList.of(platformKey));

    CycleInfo cycleInfo =
        CycleInfo.createCycleInfo(
            ImmutableList.of(
                registeredExecPlatformsKey,
                platformKey,
                simpleAspectKey,
                aspectKeyWithBaseAspects,
                toolchainContextKey,
                singleToolchainResolutionKey,
                registeredToolchainsKey));

    reporter.removeHandler(failFastHandler);
    assertThat(
            new RegisteredExecutionPlatformsCycleReporter()
                .maybeReportCycle(platformKey, cycleInfo, /* alreadyReported= */ false, reporter))
        .isTrue();
    assertContainsEvent(
        """
        Misconfigured execution platforms: //platforms:bad_platform is declared as a platform but\
         has inappropriate dependencies. Execution platforms should not have dependencies that\
         themselves require toolchain resolution.
        .-> RegisteredExecutionPlatforms
        |   //platforms:bad_platform
        |   //platforms:dep with aspect //aspects:defs.bzl%%base_aspect
        |   %s
        |   toolchain types //toolchain:test_toolchain
        |   toolchain type //toolchain:test_toolchain
        |   RegisteredToolchains
        `-- RegisteredExecutionPlatforms\
        """
            .formatted(aspectKeyWithBaseAspects.prettyPrint()));
  }

  @Test
  public void registeredToolchainsCycleReporter_reportsCycle() {
    ConfiguredTargetKey toolchainKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//toolchain:bad_toolchain"))
            .setConfiguration(targetConfig)
            .build();
    ConfiguredTargetKey depKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//toolchain:dep"))
            .setConfiguration(targetConfig)
            .build();
    AspectDescriptor baseAspectDescriptor =
        AspectDescriptor.of(
            new StarlarkAspectClass(
                keyForBuild(Label.parseCanonicalUnchecked("//aspects:defs.bzl")), "base_aspect"),
            AspectParameters.EMPTY);
    AspectDescriptor topAspectDescriptor =
        AspectDescriptor.of(
            new StarlarkAspectClass(
                keyForBuild(Label.parseCanonicalUnchecked("//aspects:defs.bzl")), "top_aspect"),
            AspectParameters.EMPTY);
    AspectKey simpleAspectKey = AspectKeyCreator.createAspectKey(baseAspectDescriptor, depKey);
    AspectKey aspectKeyWithBaseAspects =
        AspectKeyCreator.createAspectKey(
            topAspectDescriptor, ImmutableList.of(simpleAspectKey), depKey);
    SkyKey registeredToolchainsKey =
        RegisteredToolchainsValue.key(targetConfigKey, /* debug= */ false);
    SkyKey registeredExecPlatformsKey =
        RegisteredExecutionPlatformsValue.key(targetConfigKey, /* debug= */ false);
    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType)
            .build();
    SingleToolchainResolutionKey singleToolchainResolutionKey =
        SingleToolchainResolutionValue.key(
            targetConfigKey,
            testToolchainType,
            testToolchainTypeInfo,
            depKey,
            ImmutableList.of(depKey));

    CycleInfo cycleInfo =
        CycleInfo.createCycleInfo(
            ImmutableList.of(
                registeredToolchainsKey,
                toolchainKey,
                simpleAspectKey,
                aspectKeyWithBaseAspects,
                toolchainContextKey,
                registeredExecPlatformsKey,
                singleToolchainResolutionKey));

    reporter.removeHandler(failFastHandler);
    assertThat(
            new RegisteredToolchainsCycleReporter()
                .maybeReportCycle(toolchainKey, cycleInfo, /* alreadyReported= */ false, reporter))
        .isTrue();
    assertContainsEvent(
        """
        Misconfigured toolchains: //toolchain:bad_toolchain is declared as a toolchain but has\
         inappropriate dependencies. Declared toolchains should be created with the 'toolchain'\
         rule and should not have dependencies that themselves require toolchains.
        .-> RegisteredToolchains
        |   //toolchain:bad_toolchain
        |   //toolchain:dep with aspect //aspects:defs.bzl%%base_aspect
        |   %s
        |   toolchain types //toolchain:test_toolchain
        |   RegisteredExecutionPlatforms
        |   toolchain type //toolchain:test_toolchain
        `-- RegisteredToolchains\
        """
            .formatted(aspectKeyWithBaseAspects.prettyPrint()));
  }

  @Test
  public void maybeReportCycle_ignoresCyclesWithoutMatchingRootKey() {
    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.builder()
            .setLabel(Label.parseCanonicalUnchecked("//pkg:target"))
            .setConfiguration(targetConfig)
            .build();
    SkyKey registeredExecPlatformsKey =
        RegisteredExecutionPlatformsValue.key(targetConfigKey, /* debug= */ false);
    SkyKey registeredToolchainsKey =
        RegisteredToolchainsValue.key(targetConfigKey, /* debug= */ false);

    CycleInfo execPlatformsOnlyCycle =
        CycleInfo.createCycleInfo(ImmutableList.of(registeredExecPlatformsKey, targetKey));
    CycleInfo toolchainsOnlyCycle =
        CycleInfo.createCycleInfo(ImmutableList.of(registeredToolchainsKey, targetKey));
    CycleInfo targetOnlyCycle = CycleInfo.createCycleInfo(ImmutableList.of(targetKey));

    assertThat(
            new RegisteredToolchainsCycleReporter()
                .maybeReportCycle(
                    targetKey, execPlatformsOnlyCycle, /* alreadyReported= */ false, reporter))
        .isFalse();
    assertThat(
            new RegisteredExecutionPlatformsCycleReporter()
                .maybeReportCycle(
                    targetKey, toolchainsOnlyCycle, /* alreadyReported= */ false, reporter))
        .isFalse();
    assertThat(
            new RegisteredExecutionPlatformsCycleReporter()
                .maybeReportCycle(
                    targetKey, targetOnlyCycle, /* alreadyReported= */ false, reporter))
        .isFalse();
    assertThat(
            new RegisteredExecutionPlatformsCycleReporter()
                .maybeReportCycle(
                    targetKey, targetOnlyCycle, /* alreadyReported= */ true, reporter))
        .isTrue();
  }

  @Test
  public void executionPlatformCycleWithExecAspectsAndBaseAspect() throws Exception {
    scratch.file(
        "cycle/defs.bzl",
        """
        BaseInfo = provider(fields = [])

        def _base_aspect_impl(target, ctx):
            return [BaseInfo()]

        base_aspect = aspect(
            implementation = _base_aspect_impl,
            provides = [BaseInfo],
        )

        def _top_aspect_impl(target, ctx):
            return []

        top_aspect = aspect(
            implementation = _top_aspect_impl,
            requires = [base_aspect],
        )

        def _exec_cfg_impl(settings, attr):
            return {"//command_line_option:is exec configuration": True}

        exec_cfg = transition(
            implementation = _exec_cfg_impl,
            inputs = [],
            outputs = ["//command_line_option:is exec configuration"],
        )

        def _noop_impl(ctx):
            return []

        inner_rule = rule(
            implementation = _noop_impl,
        )

        outer_rule = rule(
            implementation = _noop_impl,
            attrs = {
                "tool": attr.label(cfg = exec_cfg),
            },
        )
        """);
    scratch.file(
        "cycle/BUILD",
        """
        load(":defs.bzl", "inner_rule", "outer_rule")

        platform(name = "parent_platform")

        platform(
            name = "child_platform",
            parents = [":parent_platform"],
        )

        inner_rule(name = "inner")

        outer_rule(
            name = "outer",
            tool = ":inner",
        )
        """);
    rewriteModuleDotBazel(
        """
        register_execution_platforms("//cycle:child_platform")
        """);
    useConfiguration("--exec_aspects=//cycle:defs.bzl%top_aspect");

    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//cycle:outer")).isNull();
    assertContainsEvent(
        "Misconfigured execution platforms: //cycle:child_platform is declared as a platform but"
            + " has inappropriate dependencies.");
    assertContainsEvent(
        "//cycle:parent_platform with aspect //cycle:defs.bzl%top_aspect (over [SimpleAspectKey{");
  }

  @Test
  public void toolchainCycleWithExecAspectsAndBaseAspect() throws Exception {
    scratch.file(
        "cycle/defs.bzl",
        """
        BaseInfo = provider(fields = [])

        def _base_aspect_impl(target, ctx):
            return [BaseInfo()]

        base_aspect = aspect(
            implementation = _base_aspect_impl,
            provides = [BaseInfo],
        )

        def _top_aspect_impl(target, ctx):
            return []

        top_aspect = aspect(
            implementation = _top_aspect_impl,
            requires = [base_aspect],
            toolchains = ["//toolchain:test_toolchain"],
        )

        def _exec_cfg_impl(settings, attr):
            return {"//command_line_option:is exec configuration": True}

        exec_cfg = transition(
            implementation = _exec_cfg_impl,
            inputs = [],
            outputs = ["//command_line_option:is exec configuration"],
        )

        def _noop_impl(ctx):
            return []

        inner_rule = rule(
            implementation = _noop_impl,
            toolchains = ["//toolchain:test_toolchain"],
        )

        outer_rule = rule(
            implementation = _noop_impl,
            attrs = {
                "tool": attr.label(cfg = exec_cfg),
            },
        )
        """);
    scratch.file(
        "cycle/BUILD",
        """
        load(":defs.bzl", "inner_rule", "outer_rule")

        toolchain(
            name = "bad_toolchain",
            toolchain = "//toolchain:toolchain_1_impl",
            toolchain_type = "//toolchain:test_toolchain",
        )

        inner_rule(name = "inner")

        outer_rule(
            name = "outer",
            tool = ":inner",
        )
        """);
    rewriteModuleDotBazel(
        """
        register_execution_platforms("//platforms:linux")
        register_toolchains("//cycle:bad_toolchain")
        """);
    useConfiguration("--exec_aspects=//cycle:defs.bzl%top_aspect");

    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//cycle:outer")).isNull();
    assertContainsEvent(
        // The target in the event text varies depending on whether we are in Bazel or Blaze.
        Pattern.compile(
            "Misconfigured toolchains: .+ is declared as a toolchain but has inappropriate"
                + " dependencies\\."));
    assertContainsEvent(
        Pattern.compile(
            ".+ with aspect //cycle:defs\\.bzl%top_aspect \\(over \\[SimpleAspectKey\\{"));
  }

  @Test
  public void crossConfigurationExecutionPlatformAndToolchainCycle() throws Exception {
    scratch.file(
        "cycle/defs.bzl",
        """
        def _to_toolchain_cfg_impl(settings, attr):
            return {
                "//command_line_option:extra_execution_platforms": ["//platforms:linux"],
                "//command_line_option:extra_toolchains": ["//cycle:bad_toolchain"],
            }

        to_toolchain_cfg = transition(
            implementation = _to_toolchain_cfg_impl,
            inputs = [],
            outputs = [
                "//command_line_option:extra_execution_platforms",
                "//command_line_option:extra_toolchains",
            ],
        )

        def _to_platform_cfg_impl(settings, attr):
            return {
                "//command_line_option:extra_execution_platforms": ["//cycle:bad_platform"],
                "//command_line_option:extra_toolchains": [],
            }

        to_platform_cfg = transition(
            implementation = _to_platform_cfg_impl,
            inputs = [],
            outputs = [
                "//command_line_option:extra_execution_platforms",
                "//command_line_option:extra_toolchains",
            ],
        )

        def _noop_impl(ctx):
            return []

        flag_with_dep = rule(
            implementation = _noop_impl,
            build_setting = config.string(flag = True),
            attrs = {
                "dep": attr.label(cfg = to_toolchain_cfg),
            },
        )

        bad_toolchain_rule = rule(
            implementation = _noop_impl,
            attrs = {
                "dep": attr.label(cfg = to_platform_cfg),
            },
        )

        needs_toolchain_rule = rule(
            implementation = _noop_impl,
            toolchains = ["//toolchain:test_toolchain"],
        )

        needs_exec_platform_rule = rule(
            implementation = _noop_impl,
        )
        """);
    scratch.file(
        "cycle/BUILD",
        """
        load(
            ":defs.bzl",
            "bad_toolchain_rule",
            "flag_with_dep",
            "needs_exec_platform_rule",
            "needs_toolchain_rule",
        )

        needs_exec_platform_rule(name = "needs_exec_platform")

        needs_toolchain_rule(name = "needs_toolchain")

        flag_with_dep(
            name = "my_flag",
            build_setting_default = "default",
            dep = ":needs_toolchain",
        )

        config_setting(
            name = "my_setting",
            flag_values = {":my_flag": "default"},
        )

        platform(
            name = "bad_platform",
            required_settings = [":my_setting"],
        )

        bad_toolchain_rule(
            name = "bad_toolchain",
            dep = ":needs_exec_platform",
        )
        """);
    useConfiguration("--extra_execution_platforms=//cycle:bad_platform");

    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//cycle:needs_exec_platform")).isNull();
    assertContainsEvent(
        "Misconfigured toolchains: //cycle:bad_toolchain is declared as a toolchain but has"
            + " inappropriate dependencies.");
    assertContainsEvent("RegisteredExecutionPlatforms");
    assertContainsEvent("RegisteredToolchains");
    assertContainsEvent("toolchain type //toolchain:test_toolchain");
  }
}
