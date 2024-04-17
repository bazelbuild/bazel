// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RunCommandLineTest {

  @Test
  public void linuxFormatter_formatArgv_requiresShExecutable() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new RunCommandLine.LinuxFormatter()
                .formatArgv(
                    /* shExecutable= */ null, "run under prefix", ImmutableList.of("argv")));
  }

  @Test
  public void windowsFormatter_formatArgv_runUnderRequiresShExecutable() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new RunCommandLine.WindowsFormatter()
                .formatArgv(
                    /* shExecutable= */ null, "run under prefix", ImmutableList.of("argv")));
  }

  @Test
  public void linuxFormatter_formatArgv_returnsEscapedCommandLine() {
    RunCommandLine.LinuxFormatter underTest = new RunCommandLine.LinuxFormatter();
    ImmutableList<String> result =
        underTest.formatArgv(
            "/bin/bash",
            /* runUnderPrefix= */ null,
            ImmutableList.of("executable", "argv1", "arg w spaces"));
    assertThat(result)
        .containsExactly("/bin/bash", "-c", "executable argv1 'arg w spaces'")
        .inOrder();
  }

  @Test
  public void windowsFormatter_formatArgv_returnsEscapedCommandLine() {
    RunCommandLine.WindowsFormatter underTest = new RunCommandLine.WindowsFormatter();
    ImmutableList<String> result =
        underTest.formatArgv(
            /* shExecutable= */ null,
            /* runUnderPrefix= */ null,
            ImmutableList.of("C:/unescaped executable", "argv1", "arg w spaces"));
    assertThat(result)
        .containsExactly("C:/unescaped executable", "argv1", "\"arg w spaces\"")
        .inOrder();
  }

  @Test
  public void linuxFormatter_formatArgv_runUnderPrefixPrependedToEscapedCommandLine() {
    RunCommandLine.LinuxFormatter underTest = new RunCommandLine.LinuxFormatter();
    ImmutableList<String> result =
        underTest.formatArgv(
            "/bin/bash",
            "unescaped run-under prefix &&",
            ImmutableList.of("executable", "argv1", "arg w spaces"));
    assertThat(result)
        .containsExactly(
            "/bin/bash", "-c", "unescaped run-under prefix && executable argv1 'arg w spaces'")
        .inOrder();
  }

  @Test
  public void windowsFormatter_formatArgv_runUnderPrefixPrependedToEscapedCommandLine() {
    RunCommandLine.WindowsFormatter underTest = new RunCommandLine.WindowsFormatter();
    ImmutableList<String> result =
        underTest.formatArgv(
            "unescaped /bin/bash",
            "unescaped run-under prefix &&",
            ImmutableList.of("C:/unescaped executable", "argv1", "arg w spaces"));
    assertThat(result)
        .containsExactly(
            "unescaped /bin/bash",
            "-c",
            "\"unescaped run-under prefix && 'C:/unescaped executable' argv1 'arg w spaces'\"")
        .inOrder();
  }

  @Test
  public void linuxFormatter_formatScriptPathCommandLine_returnsConcatenatedEscapedCommand() {
    RunCommandLine.LinuxFormatter underTest = new RunCommandLine.LinuxFormatter();
    String result =
        underTest.formatScriptPathCommandLine(
            "/bin/bash",
            /* runUnderPrefix= */ null,
            ImmutableList.of("executable", "argv1", "arg w spaces"));
    assertThat(result).isEqualTo("executable argv1 'arg w spaces'");
  }

  @Test
  public void linuxFormatter_formatScriptPathCommandLine_runUnderPrefixPrependedToEscapedCommand() {
    RunCommandLine.LinuxFormatter underTest = new RunCommandLine.LinuxFormatter();
    String result =
        underTest.formatScriptPathCommandLine(
            "/bin/bash",
            "unescaped run-under prefix &&",
            ImmutableList.of("executable", "argv1", "arg w spaces"));
    assertThat(result)
        .isEqualTo(
            "/bin/bash -c 'unescaped run-under prefix && executable argv1 '\\''arg w spaces'\\'''");
  }

  @Test
  public void
      windowsFormatter_formatScriptPathCommandLine_runUnderPrefixPrependedToEscapedCommand() {
    RunCommandLine.WindowsFormatter underTest = new RunCommandLine.WindowsFormatter();
    String result =
        underTest.formatScriptPathCommandLine(
            "/bin/bash",
            /* runUnderPrefix= */ "echo hello &&",
            ImmutableList.of("C:/executable", "argv1", "arg w spaces"));
    // TODO: https://github.com/bazelbuild/bazel/issues/21940 - Fix escaping.
    assertThat(result).isEqualTo("/bin/bash -c 'echo hello && C:\\executable argv1 arg w spaces'");
  }

  @Test
  public void windowsFormatter_formatScriptPathCommandLine_returnsConcatenatedEscapedCommand() {
    RunCommandLine.WindowsFormatter underTest = new RunCommandLine.WindowsFormatter();
    String result =
        underTest.formatScriptPathCommandLine(
            "/bin/bash",
            /* runUnderPrefix= */ null,
            ImmutableList.of("C:/executable", "argv1", "arg w spaces"));
    // TODO: https://github.com/bazelbuild/bazel/issues/21940 - Fix escaping.
    assertThat(result).isEqualTo("C:\\executable argv1 arg w spaces");
  }
}
