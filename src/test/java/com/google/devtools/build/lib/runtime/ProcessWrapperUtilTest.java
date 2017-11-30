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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.expectThrows;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ProcessWrapperUtil}. */
@RunWith(JUnit4.class)
public final class ProcessWrapperUtilTest {

  @Test
  public void testProcessWrapperCommandLineBuilder_ProcessWrapperPathIsRequired() {
    List<String> commandArguments = new ArrayList<>();
    commandArguments.add("echo");
    commandArguments.add("hello, world");

    Exception e =
        expectThrows(
            IllegalStateException.class,
            () ->
                ProcessWrapperUtil.commandLineBuilder()
                    .setCommandArguments(commandArguments)
                    .build());
    assertThat(e).hasMessageThat().contains("processWrapperPath");
  }

  @Test
  public void testProcessWrapperCommandLineBuilder_CommandAgumentsAreRequired() {
    String processWrapperPath = "process-wrapper";

    Exception e =
        expectThrows(
            IllegalStateException.class,
            () ->
                ProcessWrapperUtil.commandLineBuilder()
                    .setProcessWrapperPath(processWrapperPath)
                    .build());
    assertThat(e).hasMessageThat().contains("commandArguments");
  }

  @Test
  public void testProcessWrapperCommandLineBuilder_BuildsWithoutOptionalArguments() {
    String processWrapperPath = "process-wrapper";

    List<String> commandArguments = new ArrayList<>();
    commandArguments.add("echo");
    commandArguments.add("hello, world");

    List<String> expectedCommandLine = new ArrayList<>();
    expectedCommandLine.add(processWrapperPath);
    expectedCommandLine.addAll(commandArguments);

    List<String> commandLine =
        ProcessWrapperUtil.commandLineBuilder()
            .setProcessWrapperPath(processWrapperPath)
            .setCommandArguments(commandArguments)
            .build();

    assertThat(commandLine).containsExactlyElementsIn(expectedCommandLine);
  }

  @Test
  public void testProcessWrapperCommandLineBuilder_BuildsWithOptionalArguments() {
    String processWrapperPath = "process-wrapper";

    List<String> commandArguments = new ArrayList<>();
    commandArguments.add("echo");
    commandArguments.add("hello, world");

    Duration timeout = Duration.ofSeconds(10);
    Duration killDelay = Duration.ofSeconds(2);
    String stdoutPath = "stdout.txt";
    String stderrPath = "stderr.txt";

    List<String> expectedCommandLine = new ArrayList<>();
    expectedCommandLine.add(processWrapperPath);
    expectedCommandLine.add("--timeout=" + timeout.getSeconds());
    expectedCommandLine.add("--kill_delay=" + killDelay.getSeconds());
    expectedCommandLine.add("--stdout=" + stdoutPath);
    expectedCommandLine.add("--stderr=" + stderrPath);
    expectedCommandLine.addAll(commandArguments);

    List<String> commandLine =
        ProcessWrapperUtil.commandLineBuilder()
            .setProcessWrapperPath(processWrapperPath)
            .setCommandArguments(commandArguments)
            .setTimeout(timeout)
            .setKillDelay(killDelay)
            .setStdoutPath(stdoutPath)
            .setStderrPath(stderrPath)
            .build();

    assertThat(commandLine).containsExactlyElementsIn(expectedCommandLine);
  }
}
