// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import net.starlark.java.eval.Sequence;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the action properties on rule configured targets of Apple related rules. */
@RunWith(JUnit4.class)
public class AppleRulesTest extends AnalysisTestCase {

  @Before
  public void setup() throws Exception {
    MockObjcSupport.setup(mockToolsConfig);
    scratch.file(
        "test/aspect.bzl",
        """
        foo = provider()

        def _impl(target, ctx):
            return [foo(actions = target.actions)]

        MyAspect = aspect(implementation = _impl)
        """);
    scratch.file(
        "xcode/BUILD",
        """
        xcode_version(
            name = "version10_1_0",
            aliases = [
                "10.1",
                "10.1.0",
            ],
            default_ios_sdk_version = "12.1",
            default_macos_sdk_version = "10.14",
            default_tvos_sdk_version = "12.1",
            default_watchos_sdk_version = "5.1",
            version = "10.1.0",
        )

        xcode_version(
            name = "version10_2_1",
            aliases = [
                "10.2.1",
                "10.2",
            ],
            default_ios_sdk_version = "12.2",
            default_macos_sdk_version = "10.14",
            default_tvos_sdk_version = "12.2",
            default_watchos_sdk_version = "5.2",
            version = "10.2.1",
        )

        available_xcodes(
            name = "xcodes_a",
            default = ":version10_1_0",
            versions = [":version10_1_0"],
        )

        available_xcodes(
            name = "xcodes_b",
            default = ":version10_2_1",
            versions = [":version10_2_1"],
        )

        xcode_config(
            name = "local",
            local_versions = ":xcodes_a",
            remote_versions = ":xcodes_b",
        )

        xcode_config(
            name = "mutual",
            local_versions = ":xcodes_b",
            remote_versions = ":xcodes_b",
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        cc_library(
            name = "xxx",
            srcs = ["dep1.cc"],
            hdrs = ["dep1.h"],
            defines = ["DEP1"],
            includes = ["dep1/baz"],
        )
        """);
  }

  @Test
  public void executionRequirementsSetCcLibrary() throws Exception {
    ImmutableList<String> flags =
        ImmutableList.<String>builder()
            .addAll(MockObjcSupport.requiredObjcCrosstoolFlagsNoXcodeConfig())
            .add("--xcode_version_config=//xcode:local")
            .add("--cpu=darwin_x86_64")
            .build();
    useConfiguration(flags.toArray(new String[1]));
    AnalysisResult analysisResult =
        update(ImmutableList.of("test/aspect.bzl%MyAspect"), "//test:xxx");

    ConfiguredAspect configuredAspect =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());

    StarlarkProvider.Key fooKey =
        new StarlarkProvider.Key(keyForBuild(Label.parseCanonical("//test:aspect.bzl")), "foo");

    StructImpl fooProvider = (StructImpl) configuredAspect.get(fooKey);
    assertThat(fooProvider.getValue("actions")).isNotNull();
    @SuppressWarnings("unchecked")
    Sequence<ActionAnalysisMetadata> actions =
        (Sequence<ActionAnalysisMetadata>) fooProvider.getValue("actions");
    assertThat(actions).isNotEmpty();

    for (ActionAnalysisMetadata action : actions) {
      assertThat(action).isInstanceOf(AbstractAction.class);
      if (action.getExecutionInfo().containsKey("requires-darwin")) {
        assertThat(action.getExecutionInfo()).containsKey("supports-xcode-requirements-set");
        assertThat(action.getExecutionInfo()).containsKey("no-remote");
      }
    }
  }

  @Test
  public void dottedVersionOptionIsReadableFromStarlarkTransition() throws Exception {
    // Test that DottedVersion.Option is readable from a Starlark transition, since it is a distinct
    // type from DottedVersion (see the documentation comment on DottedVersion.Option for the
    // rationale).
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        """
        package_group(
            name = "function_transition_allowlist",
            packages = ["//..."],
        )

        filegroup(
            name = "srcs",
            srcs = glob(["**"]),
            visibility = ["//tools/allowlists:__pkg__"],
        )
        """);
    scratch.file(
        "transition/transition.bzl",
        """
        def _silly_transition_impl(settings, attr):
            version = str(settings["//command_line_option:ios_minimum_os"])
            next = version if version.endswith(".1") else version + ".1"
            return {"//command_line_option:ios_minimum_os": next}

        silly_transition = transition(
            implementation = _silly_transition_impl,
            inputs = ["//command_line_option:ios_minimum_os"],
            outputs = ["//command_line_option:ios_minimum_os"],
        )

        def _my_rule_impl(ctx):
            return []

        my_rule = rule(
            cfg = silly_transition,
            implementation = _my_rule_impl,
        )
        """);
    scratch.file(
        "transition/BUILD",
        """
        load("//transition:transition.bzl", "my_rule")

        my_rule(name = "xxx")
        """);

    useConfiguration("--ios_minimum_os=10.0");
    AnalysisResult result = update("//transition:xxx");
    BuildConfigurationValue configuration =
        Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs()).getConfiguration();
    AppleCommandLineOptions appleOptions =
        configuration.getOptions().get(AppleCommandLineOptions.class);
    assertThat(appleOptions.iosMinimumOs).isNotNull();
    DottedVersion version = DottedVersion.maybeUnwrap(appleOptions.iosMinimumOs);
    assertThat(version.toString()).isEqualTo("10.0.1");
  }
}
