// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests use of the --target_environment flag. */
@RunWith(TestParameterInjector.class)
public class EnvironmentRestrictedBuildTest extends BuildIntegrationTestCase {
  @TestParameter boolean mergedSkyframeAnalysisExecution;

  @Before
  public final void addNoBuildOption() throws Exception  {
    if (mergedSkyframeAnalysisExecution) {
      // TODO(b/223761810): Add --nobuild after Skymeld supports it.
      addOptions("--experimental_merged_skyframe_analysis_execution");
    } else {
      addOptions("--nobuild"); // Target enforcement happens before the execution phase.
    }
  }

  private void writeEnvironmentRules(String... defaults) throws Exception {
    StringBuilder defaultsBuilder = new StringBuilder();
    for (String defaultEnv : defaults) {
      defaultsBuilder.append("'" + defaultEnv + "', ");
    }

    write("buildenv/BUILD",
        "environment_group(",
        "    name = 'group',",
        "    environments = [':one', ':two'],",
        "    defaults = [" + defaultsBuilder + "])",
        "environment(name = 'one')",
        "environment(name = 'two')");
  }

  @Test
  public void testTargetEnvironmentError() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'])");
    addOptions("--target_environment=//buildenv:one");
    assertThat(assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:bar")))
        .hasMessageThat()
        .contains(
            ""
                + "//foo:bar declares compatibility with:\n"
                + "  []\n"
                + "but does not support:\n"
                + "  //buildenv:one");
  }

  @Test
  public void testTargetEnvironmentSuccess() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'], compatible_with = ['//buildenv:one'])");
    write("foo/bar.sh");
    addOptions("--target_environment=//buildenv:one");
    buildTarget("//foo:bar");
    assertThat(getResult().getSuccess()).isTrue();
  }

  @Test
  public void testMultipleTargetEnvironments() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'], compatible_with = ['//buildenv:one'])");

    addOptions("--target_environment=//buildenv:one", "--target_environment=//buildenv:two");
    assertThat(assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:bar")))
        .hasMessageThat()
        .contains(
            ""
                + "//foo:bar declares compatibility with:\n"
                + "  [//buildenv:one]\n"
                + "but does not support:\n"
                + "  //buildenv:two");
  }

  @Test
  public void testTargetEnvironmentIsDefault() throws Exception {
    writeEnvironmentRules(":one");
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'])");
    write("foo/bar.sh");
    addOptions("--target_environment=//buildenv:one");
    buildTarget("//foo:bar");
    assertThat(getResult().getSuccess()).isTrue();
  }

  @Test
  public void testEmptyTargetEnvironment() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'])");
    write("foo/bar.sh");
    buildTarget("//foo:bar");
    assertThat(getResult().getSuccess()).isTrue();
  }

  @Test
  public void testOnlySomeTargetsQualify() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        """
        load('//test_defs:foo_library.bzl', 'foo_library')
        foo_library(
            name = "good_bar",
            srcs = ["bar.sh"],
            compatible_with = ["//buildenv:one"],
        )

        foo_library(
            name = "bad_bar",
            srcs = ["bar.sh"],
            compatible_with = ["//buildenv:two"],
        )
        """);
    write("foo/bar.sh");
    addOptions("--target_environment=//buildenv:one");
    assertThat(assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:all")))
        .hasMessageThat()
        .contains(
            ""
                + "//foo:bad_bar declares compatibility with:\n"
                + "  [//buildenv:two]\n"
                + "but does not support:\n"
                + "  //buildenv:one");
  }

  @Test
  public void testNoConstraintEnforcement() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'])");
    write("foo/bar.sh");
    addOptions("--target_environment=//buildenv:one", "--noenforce_constraints");
    buildTarget("//foo:bar");
    assertThat(getResult().getSuccess()).isTrue();
  }

  @Test
  public void testFlagUsesNonexistentTarget() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'])");

    addOptions("--target_environment=//buildenv:nada");
    assertThat(assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:bar")))
        .hasMessageThat()
        .contains("invalid target environment: no such target '//buildenv:nada'");
  }

  @Test
  public void testFlagUsesWrongTargetType() throws Exception {
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'])");

    addOptions("--target_environment=//foo:bar");
    assertThat(assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:bar")))
        .hasMessageThat()
        .contains("//foo:bar is not a valid environment definition");
  }

  @Test
  public void testRefinedEnvironmentCheckValidTarget() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        """
        load('//test_defs:foo_library.bzl', 'foo_library')
        config_setting(
            name = "config_one",
            values = {"define": "mode=one"},
        )

        config_setting(
            name = "config_two",
            values = {"define": "mode=two"},
        )

        foo_library(
            name = "lib_one",
            srcs = [],
            compatible_with = ["//buildenv:one"],
        )

        foo_library(
            name = "lib_two",
            srcs = [],
            compatible_with = ["//buildenv:two"],
        )

        foo_library(
            name = "toplevel",
            srcs = ["toplevel.sh"],
            compatible_with = [
                "//buildenv:one",
                "//buildenv:two",
            ],
            deps = select({
                ":config_one": [":lib_one"],
                ":config_two": [":lib_two"],
            }),
        )
        """);
    // "--define mode=one" refines :toplevel to (matching) ["//buildenv:one"]:
    addOptions("--target_environment=//buildenv:one", "--define", "mode=one");
    write("foo/toplevel.sh");
    buildTarget("//foo:toplevel");
    assertThat(getResult().getSuccess()).isTrue();
  }

  @Test
  public void testRefinedEnvironmentCheckBadTarget() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        """
        load('//test_defs:foo_library.bzl', 'foo_library')
        config_setting(
            name = "config_one",
            values = {"define": "mode=one"},
        )

        config_setting(
            name = "config_two",
            values = {"define": "mode=two"},
        )

        foo_library(
            name = "lib_one",
            srcs = [],
            compatible_with = ["//buildenv:one"],
        )

        foo_library(
            name = "lib_two",
            srcs = [],
            compatible_with = ["//buildenv:two"],
        )

        foo_library(
            name = "toplevel",
            srcs = ["toplevel.sh"],
            compatible_with = [
                "//buildenv:one",
                "//buildenv:two",
            ],
            deps = select({
                ":config_one": [":lib_one"],
                ":config_two": [":lib_two"],
            }),
        )
        """);
    // "--define mode=two" refines :toplevel to (non-matching) ["//buildenv:two"]:
    addOptions("--target_environment=//buildenv:one", "--define", "mode=two");
    assertThat(assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:toplevel")))
        .hasMessageThat()
        .contains(
            ""
                + "//foo:toplevel declares compatibility with:\n"
                + "  [//buildenv:one, //buildenv:two]\n"
                + "but does not support:\n"
                + "  environment: //buildenv:one\n"
                + "    removed by: //foo:toplevel");
  }

  @Test
  public void topLevelOutputFile() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/rule.bzl",
        """
        def _impl(ctx):
            file = ctx.actions.declare_file("libbar.a")
            ctx.actions.write(file, "hello")
            return [DefaultInfo(files = depset([file]))]

        crule = rule(
            _impl,
            outputs = {
                "archive": "lib%{name}.a",
            },
        )
        """);
    write(
        "foo/BUILD",
        """
        load(":rule.bzl", "crule")

        crule(
            name = "bar",
            compatible_with = ["//buildenv:one"],
        )
        """);
    addOptions("--target_environment=//buildenv:one");
    buildTarget("//foo:libbar.a");
    assertThat(getResult().getSuccess()).isTrue();
  }

  @Test
  public void topLevelAliasToCompatibleOutputFile() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        """
        genrule(
            name = "goodgen",
            srcs = [],
            outs = ["goodgen.out"],
            cmd = "touch $@",
            compatible_with = ["//buildenv:one"],
        )

        alias(
            name = "goodalias",
            actual = "goodgen.out",
        )
        """);
    addOptions("--target_environment=//buildenv:one");
    buildTarget("//foo:goodalias");
    assertThat(getResult().getSuccess()).isTrue();
  }

  @Test
  public void topLevelAliasToBadOutputFile() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        """
        genrule(
            name = "badgen",
            srcs = [],
            outs = ["badgen.out"],
            cmd = "",
        )

        alias(
            name = "badalias",
            actual = "badgen.out",
        )
        """);
    addOptions("--target_environment=//buildenv:one");
    assertThat(assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:badalias")))
        .hasMessageThat()
        .contains(
            ""
                + "//foo:badgen.out declares compatibility with:\n"
                + "  []\n"
                + "but does not support:\n"
                + "  //buildenv:one");
  }

  @Test
  public void doesNotCheckDefaultEnvironments() throws Exception {
    write(
        "buildenv/a/BUILD",
        """
        environment_group(
            name = "a",
            defaults = [":a1"],
            environments = [
                ":a1",
                ":a2",
            ],
        )

        environment(name = "a1")

        environment(name = "a2")
        """);
    write(
        "buildenv/b/BUILD",
        """
        environment_group(
            name = "b",
            defaults = [":b1"],
            environments = [
                ":b1",
                ":b2",
            ],
        )

        environment(name = "b1")

        environment(name = "b2")
        """);

    write("foo/bar.sh", "echo Bar!");
    write(
        "foo/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'bar', srcs = ['bar.sh'], restricted_to = ['//buildenv/b:b2'])");

    addOptions("--target_environment=//buildenv/a:a1");
    buildTarget("//foo:bar");
    assertThat(getResult().getSuccess()).isTrue();
  }
}
