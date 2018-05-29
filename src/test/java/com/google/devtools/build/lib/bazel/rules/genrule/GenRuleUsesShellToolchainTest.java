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

package com.google.devtools.build.lib.bazel.rules.genrule;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.TestConstants.GENRULE_SETUP;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Assert that genrule uses the shell toolchain. */
@RunWith(JUnit4.class)
public class GenRuleUsesShellToolchainTest extends BuildViewTestCase {

  private static final Pattern SETUP_COMMAND_PATTERN =
      Pattern.compile(".*/genrule-setup.sh;\\s+(?<command>.*)");

  @Test
  public void testActionIsShellCommandUsingShellFromShellConfig() throws Exception {
    useConfiguration("--shell_executable=/custom/shell/executable");
    assertActionIsShellCommand("/custom/shell/executable");
  }

  @Test
  public void testActionIsShellCommandUsingShellFromShellToolchain() throws Exception {
    useConfiguration("--shell_executable=");
    assertActionIsShellCommand("/mock/shell/toolchain");
  }

  @Test
  public void testActionIsShellCommandUsingShellFromExtraToolchain() throws Exception {
    useConfiguration("--shell_executable=", "--extra_toolchains=//pkg1:tc");
    scratch.file(
        "pkg1/BUILD",
        "load('@bazel_tools//tools/sh:sh_toolchain.bzl', 'sh_toolchain')",
        "",
        "sh_toolchain(",
        "    name = 'tc_def',",
        "    path = '/foo/bar/baz',",
        "    visibility = ['//visibility:public'],",
        ")",
        "",
        "toolchain(",
        "    name = 'tc',",
        "    toolchain = 'tc_def',",
        "    toolchain_type = '@bazel_tools//tools/sh:toolchain_type',",
        ")");

    assertActionIsShellCommand("/foo/bar/baz");
  }

  private void assertActionIsShellCommand(String expectedShell) throws Exception {
    scratch.file(
        "genrule1/BUILD",
        "genrule(",
        "    name = 'genrule1',",
        "    srcs = ['input.txt'],",
        "    outs = ['output.txt'],",
        "    cmd  = 'dummy >$@',",
        ")");

    Artifact input = getFileConfiguredTarget("//genrule1:input.txt").getArtifact();
    Artifact genruleSetup = getFileConfiguredTarget(GENRULE_SETUP).getArtifact();

    Artifact out = getFileConfiguredTarget("//genrule1:output.txt").getArtifact();
    SpawnAction shellAction = (SpawnAction) getGeneratingAction(out);
    assertThat(shellAction).isNotNull();
    assertThat(shellAction.getInputs()).containsExactly(input, genruleSetup);
    assertThat(shellAction.getOutputs()).containsExactly(out);

    String expected = "dummy >" + out.getExecPathString();
    assertThat(shellAction.getArguments().get(0)).isEqualTo(expectedShell);
    assertThat(shellAction.getArguments().get(1)).isEqualTo("-c");
    Matcher m = SETUP_COMMAND_PATTERN.matcher(shellAction.getArguments().get(2));
    assertThat(m.matches()).isTrue();
    assertThat(m.group("command")).isEqualTo(expected);
  }
}
