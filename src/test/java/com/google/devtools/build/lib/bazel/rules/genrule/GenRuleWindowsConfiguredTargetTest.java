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

package com.google.devtools.build.lib.bazel.rules.genrule;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assume.assumeTrue;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link BazelGenRule} on Windows. */
@RunWith(JUnit4.class)
public class GenRuleWindowsConfiguredTargetTest extends BuildViewTestCase {

  private static final Pattern POWERSHELL_COMMAND_PATTERN =
      Pattern.compile(".*'utf8';\\s+(?<command>.*)");

  private static final Pattern BASH_COMMAND_PATTERN =
      Pattern.compile(".*/genrule-setup.sh;\\s+(?<command>.*)");

  private static void assertCommandEquals(String expected, String command, Pattern pattern) {
    // Ensure the command after the genrule setup is correct.
    Matcher m = pattern.matcher(command);
    if (m.matches()) {
      command = m.group("command");
    }
    assertThat(command).isEqualTo(expected);
  }

  private static void assertPowershellCommandEquals(String expected, String command) {
    assertCommandEquals(expected, command, POWERSHELL_COMMAND_PATTERN);
  }

  private static void assertBashCommandEquals(String expected, String command) {
    assertCommandEquals(expected, command, BASH_COMMAND_PATTERN);
  }

  private static String getWindowsPath(Artifact artifact) {
    return artifact.getExecPathString().replace('/', '\\');
  }

  @Before
  public void assumeBazel() throws Exception {
    // The cmd_{bash,bat,ps} attributes don't exist in Blaze.
    assumeTrue(analysisMock.isThisBazel());
  }

  @Before
  public void createWindowsPlatform() throws Exception {
    scratch.file(
        "platforms/BUILD",
        "platform(name = 'windows', constraint_values = ['@platforms//os:windows'])");
    useConfiguration("--host_platform=//platforms:windows");
  }

  @Test
  public void testCmdBatchIsPreferred() throws Exception {
    scratch.file(
        "genrule1/BUILD",
        "genrule(name = 'hello_world',",
        "outs = ['message.txt'],",
        "cmd  = 'echo \"Hello, default cmd.\" >$(location message.txt)',",
        "cmd_bash  = 'echo \"Hello, Bash cmd.\" >$(location message.txt)',",
        "cmd_bat  = 'echo \"Hello, Batch cmd.\" >$(location message.txt)')");

    Artifact messageArtifact = getFileConfiguredTarget("//genrule1:message.txt").getArtifact();
    SpawnAction shellAction = (SpawnAction) getGeneratingAction(messageArtifact);

    assertThat(shellAction).isNotNull();
    assertThat(shellAction.getOutputs()).containsExactly(messageArtifact);

    String expected = "echo \"Hello, Batch cmd.\" >" + getWindowsPath(messageArtifact);
    assertThat(shellAction.getArguments().get(0)).isEqualTo("cmd.exe");
    int last = shellAction.getArguments().size() - 1;
    assertThat(shellAction.getArguments().get(last - 1)).isEqualTo("/c");
    assertThat(shellAction.getArguments().get(last)).isEqualTo(expected);
  }

  @Test
  public void testCmdPsIsPreferred() throws Exception {
    scratch.file(
        "genrule1/BUILD",
        "genrule(name = 'hello_world',",
        "outs = ['message.txt'],",
        "cmd  = 'echo \"Hello, default cmd.\" >$(location message.txt)',",
        "cmd_bash  = 'echo \"Hello, Bash cmd.\" >$(location message.txt)',",
        "cmd_bat  = 'echo \"Hello, Batch cmd.\" >$(location message.txt)',",
        "cmd_ps  = 'echo \"Hello, Powershell cmd.\" >$(location message.txt)')");

    Artifact messageArtifact = getFileConfiguredTarget("//genrule1:message.txt").getArtifact();
    SpawnAction shellAction = (SpawnAction) getGeneratingAction(messageArtifact);

    assertThat(shellAction).isNotNull();
    assertThat(shellAction.getOutputs()).containsExactly(messageArtifact);

    String expected = "echo \"Hello, Powershell cmd.\" >" + messageArtifact.getExecPathString();
    assertThat(shellAction.getArguments().get(0)).isEqualTo("powershell.exe");
    assertThat(shellAction.getArguments().get(1)).isEqualTo("/c");
    assertPowershellCommandEquals(expected, shellAction.getArguments().get(2));
  }

  @Test
  public void testScriptFileIsUsedForBatchCmd() throws Exception {
    scratch.file(
        "genrule1/BUILD",
        "genrule(name = 'hello_world',",
        "outs = ['message.txt'],",
        "cmd_bat  = ' && '.join([\"echo \\\"Hello, Batch cmd, %s.\\\" >$(location message.txt)\" %"
            + " i for i in range(1, 1000)]),)");

    Artifact messageArtifact = getFileConfiguredTarget("//genrule1:message.txt").getArtifact();
    SpawnAction shellAction = (SpawnAction) getGeneratingAction(messageArtifact);

    assertThat(shellAction).isNotNull();
    assertThat(shellAction.getOutputs()).containsExactly(messageArtifact);

    String expected = "bazel-out\\k8-fastbuild\\bin\\genrule1\\hello_world.genrule_script.bat";
    assertThat(shellAction.getArguments().get(0)).isEqualTo("cmd.exe");
    int last = shellAction.getArguments().size() - 1;
    assertThat(shellAction.getArguments().get(last - 1)).isEqualTo("/c");
    assertPowershellCommandEquals(expected, shellAction.getArguments().get(last));
  }

  @Test
  public void testScriptFileIsUsedForPowershellCmd() throws Exception {
    scratch.file(
        "genrule1/BUILD",
        "genrule(name = 'hello_world',",
        "outs = ['message.txt'],",
        "cmd_ps  = '; '.join([\"echo \\\"Hello, Powershell cmd, %s.\\\" >$(location message.txt)\""
            + " % i for i in range(1, 1000)]),)");

    Artifact messageArtifact = getFileConfiguredTarget("//genrule1:message.txt").getArtifact();
    SpawnAction shellAction = (SpawnAction) getGeneratingAction(messageArtifact);

    assertThat(shellAction).isNotNull();
    assertThat(shellAction.getOutputs()).containsExactly(messageArtifact);

    String expected = ".\\bazel-out\\k8-fastbuild\\bin\\genrule1\\hello_world.genrule_script.ps1";
    assertThat(shellAction.getArguments().get(0)).isEqualTo("powershell.exe");
    assertThat(shellAction.getArguments().get(1)).isEqualTo("/c");
    assertPowershellCommandEquals(expected, shellAction.getArguments().get(2));
  }

  @Test
  public void testCmdBashIsPreferred() throws Exception {
    scratch.file(
        "genrule1/BUILD",
        "genrule(name = 'hello_world',",
        "outs = ['message.txt'],",
        "cmd  = 'echo \"Hello, default cmd.\" >$(location message.txt)',",
        "cmd_bash  = 'echo \"Hello, Bash cmd.\" >$(location message.txt)')");

    Artifact messageArtifact = getFileConfiguredTarget("//genrule1:message.txt").getArtifact();
    SpawnAction shellAction = (SpawnAction) getGeneratingAction(messageArtifact);

    assertThat(shellAction).isNotNull();
    assertThat(shellAction.getOutputs()).containsExactly(messageArtifact);

    String expected = "echo \"Hello, Bash cmd.\" >" + messageArtifact.getExecPathString();
    assertThat(shellAction.getArguments().get(0)).isEqualTo("c:/tools/msys64/usr/bin/bash.exe");
    assertThat(shellAction.getArguments().get(1)).isEqualTo("-c");
    assertBashCommandEquals(expected, shellAction.getArguments().get(2));
  }

  @Test
  public void testMissingCmdAttributeError() throws Exception {
    checkError(
        "foo",
        "bar",
        "missing value for `cmd` attribute, you can also set `cmd_ps` or `cmd_bat` on"
            + " Windows and `cmd_bash` on other platforms.",
        "genrule(name='bar'," + "      srcs = []," + "      outs=['out'])");
  }

  @Test
  public void testMissingCmdAttributeErrorOnNonWindowsPlatform() throws Exception {
    scratch.overwriteFile(
        "platforms/BUILD",
        "platform(name = 'nonwindows', constraint_values = ['@platforms//os:linux'])");
    useConfiguration("--host_platform=//platforms:nonwindows");

    checkError(
        "foo",
        "bar",
        "missing value for `cmd` attribute, you can also set `cmd_ps` or `cmd_bat` on"
            + " Windows and `cmd_bash` on other platforms.",
        "genrule(name='bar',"
            + "      srcs = [],"
            + "      outs=['out'],"
            + "      cmd_bat='echo hello > $(@)')");
  }
}
