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

package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Command}s that are wrapped using the {@code process-wrapper}. */
@RunWith(JUnit4.class)
public final class CommandUsingProcessWrapperTest {
  private FileSystem testFS;

  @Before
  public final void createFileSystem() throws Exception {
    testFS = new UnixFileSystem(DigestHashFunction.getDefaultUnchecked());
  }

  private ProcessWrapper getProcessWrapper() {
    return new ProcessWrapper(
        testFS
            .getPath(BlazeTestUtils.runfilesDir())
            .getRelative(TestConstants.PROCESS_WRAPPER_PATH),
        /*killDelay=*/ null,
        /*extraFlags=*/ ImmutableList.of());
  }

  private String getCpuTimeSpenderPath() {
    return BlazeTestUtils.runfilesDir() + "/" + TestConstants.CPU_TIME_SPENDER_PATH;
  }

  @Test
  public void testCommand_Echo() throws Exception {
    ImmutableList<String> commandArguments = ImmutableList.of("echo", "worker bees can leave");

    Command command = new Command(commandArguments.toArray(new String[0]));
    CommandResult commandResult = command.execute();

    assertThat(commandResult.getTerminationStatus().success()).isTrue();
    assertThat(commandResult.getStdoutStream().toString()).contains("worker bees can leave");
  }

  @Test
  public void testProcessWrappedCommand_Echo() throws Exception {
    ImmutableList<String> commandArguments = ImmutableList.of("echo", "even drones can fly away");

    List<String> fullCommandLine = getProcessWrapper().commandLineBuilder(commandArguments).build();

    Command command = new Command(fullCommandLine.toArray(new String[0]));
    CommandResult commandResult = command.execute();

    assertThat(commandResult.getTerminationStatus().success()).isTrue();
    assertThat(commandResult.getStdoutStream().toString()).contains("even drones can fly away");
  }

  private void checkProcessWrapperStatistics(Duration userTimeToSpend, Duration systemTimeToSpend)
      throws IOException, CommandException {
    ImmutableList<String> commandArguments =
        ImmutableList.of(
            getCpuTimeSpenderPath(),
            Long.toString(userTimeToSpend.getSeconds()),
            Long.toString(systemTimeToSpend.getSeconds()));

    Path outputDir = testFS.getPath(TestUtils.makeTempDir().getCanonicalPath());
    Path statisticsFilePath = outputDir.getRelative("stats.out");

    List<String> fullCommandLine =
        getProcessWrapper()
            .commandLineBuilder(commandArguments)
            .setStatisticsPath(statisticsFilePath)
            .build();

    ExecutionStatisticsTestUtil.executeCommandAndCheckStatisticsAboutCpuTimeSpent(
        userTimeToSpend, systemTimeToSpend, fullCommandLine, statisticsFilePath);
  }

  @Test
  public void testProcessWrappedCommand_WithStatistics_SpendUserTime()
      throws CommandException, IOException {
    Duration userTimeToSpend = Duration.ofSeconds(10);
    Duration systemTimeToSpend = Duration.ZERO;

    checkProcessWrapperStatistics(userTimeToSpend, systemTimeToSpend);
  }

  @Test
  public void testProcessWrappedCommand_WithStatistics_SpendSystemTime()
      throws CommandException, IOException {
    Duration userTimeToSpend = Duration.ZERO;
    Duration systemTimeToSpend = Duration.ofSeconds(10);

    checkProcessWrapperStatistics(userTimeToSpend, systemTimeToSpend);
  }

  @Test
  public void testProcessWrappedCommand_WithStatistics_SpendUserAndSystemTime()
      throws CommandException, IOException {
    Duration userTimeToSpend = Duration.ofSeconds(10);
    Duration systemTimeToSpend = Duration.ofSeconds(10);

    checkProcessWrapperStatistics(userTimeToSpend, systemTimeToSpend);
  }
}
