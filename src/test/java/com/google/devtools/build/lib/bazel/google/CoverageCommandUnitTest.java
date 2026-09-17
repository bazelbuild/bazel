// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.google;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.buildtool.InstrumentationFilterSupport.getInstrumentedPrefix;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.coverage.CoverageArgs;
import com.google.devtools.build.lib.bazel.coverage.CoverageReportActionBuilder;
import com.google.devtools.build.lib.skyframe.CoverageReportValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for Bazel coverage. */
@RunWith(JUnit4.class)
public final class CoverageCommandUnitTest extends BuildViewTestCase {
  @Test
  public void testGetInstrumentedPrefixJavatests() throws Exception {
    // Make sure that javatests dir still gets replaced even when immediately under top-level dir
    scratch.file(
        "javatests/com/google/foo/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "l",
            srcs = ["foo.java"],
        )
        """);
    String packageName = getConfiguredTarget("//javatests/com/google/foo:l").getLabel()
        .getPackageName();
    assertThat(packageName).isEqualTo("javatests/com/google/foo"); // No leading slashes
    assertThat(getInstrumentedPrefix(packageName)).isEqualTo("java/com/google/foo");
  }

  @Test
  public void testGetInstrumentedPrefix() {
    assertThat(getInstrumentedPrefix("javatests/foo")).isEqualTo("java/foo");
    assertThat(getInstrumentedPrefix("third_party/foo/javatests/foo"))
        .isEqualTo("third_party/foo/java/foo");
    assertThat(getInstrumentedPrefix("third_party/foo/javatest/foo"))
        .isEqualTo("third_party/foo/javatest/foo"); // No substitution of javatest without the s
    assertThat(getInstrumentedPrefix("third_party/foo/src/test/java/foo"))
        .isEqualTo("third_party/foo/src/main/java/foo");
    assertThat(getInstrumentedPrefix("test/java/foo")).isEqualTo("main/java/foo");
    assertThat(getInstrumentedPrefix("foo/internal")).isEqualTo("foo");
    assertThat(getInstrumentedPrefix("foo/public")).isEqualTo("foo");
    assertThat(getInstrumentedPrefix("foo/tests")).isEqualTo("foo");
  }

  @Test
  public void testCoverageReportActionBuilder_disabledCoverageDoesNotNpe() throws Exception {
    useConfiguration("--collect_code_coverage");

    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        """
        package_group(
            name = "function_transition_allowlist",
            packages = ["//..."],
        )
        """);

    scratch.file(
        "repro/repro.bzl",
        """
        def _disable_coverage_transition_impl(settings, attr):
            return {"//command_line_option:collect_code_coverage": False}

        disable_coverage_transition = transition(
            implementation = _disable_coverage_transition_impl,
            inputs = [],
            outputs = ["//command_line_option:collect_code_coverage"],
        )

        def _my_test_impl(ctx):
            executable = ctx.actions.declare_file(ctx.label.name + ".sh")
            ctx.actions.write(
                output = executable,
                content = "#!/bin/bash\\nexit 0\\n",
                is_executable = True,
            )
            return [
                DefaultInfo(executable = executable),
            ]

        my_test = rule(
            implementation = _my_test_impl,
            test = True,
            cfg = disable_coverage_transition,
            attrs = {
                "_allowlist_function_transition": attr.label(
                    default = "//tools/allowlists/function_transition_allowlist"
                ),
            },
        )

        normal_test = rule(
            implementation = _my_test_impl,
            test = True,
        )
        """);

    scratch.file(
        "repro/BUILD",
        """
        load("//repro:repro.bzl", "my_test", "normal_test")

        normal_test(
            name = "normal_test_target",
        )

        my_test(
            name = "transitioned_test_target",
        )
        """);

    ConfiguredTarget normalTarget = getConfiguredTarget("//repro:normal_test_target");
    ConfiguredTarget transitionedTarget = getConfiguredTarget("//repro:transitioned_test_target");

    ImmutableList<ConfiguredTarget> targetsToTest =
        ImmutableList.of(normalTarget, transitionedTarget);

    CoverageReportActionBuilder builder = new CoverageReportActionBuilder();
    CoverageReportActionBuilder.CoverageHelper dummyHelper =
        new CoverageReportActionBuilder.CoverageHelper() {
          @Override
          public ImmutableList<String> getArgs(CoverageArgs args, Artifact lcovOutput) {
            return ImmutableList.of();
          }

          @Override
          public String getLocationMessage(CoverageArgs args, Artifact lcovOutput) {
            return "";
          }
        };

    // This should not throw NullPointerException
    Object unused =
        builder.createCoverageActionsWrapper(
            reporter,
            directories,
            /* configuredTargets= */ ImmutableList.of(),
            targetsToTest,
            view.getArtifactFactory(),
            actionKeyContext,
            CoverageReportValue.COVERAGE_REPORT_KEY,
            /* workspaceName= */ "workspace",
            dummyHelper,
            /* htmlReport= */ null);
  }
}
