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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ProcessWrapper}. */
@RunWith(JUnit4.class)
public final class ProcessWrapperTest {

  private FileSystem testFS;

  @Before
  public void setUp() {
    testFS = new InMemoryFileSystem();
  }

  private ProcessWrapper getProcessWrapper(String path, @Nullable Duration killDelay)
      throws IOException {
    Path processWrapperPath = testFS.getPath(path);
    processWrapperPath.getParentDirectory().createDirectoryAndParents();
    processWrapperPath.getOutputStream().close();
    return new ProcessWrapper(processWrapperPath, killDelay);
  }

  @Test
  public void testProcessWrapperCommandLineBuilder_BuildsWithoutOptionalArguments()
      throws IOException {
    ImmutableList<String> commandArguments = ImmutableList.of("echo", "hello, world");

    ImmutableList<String> expectedCommandLine =
        ImmutableList.<String>builder().add("/some/bin/path").addAll(commandArguments).build();

    ProcessWrapper processWrapper = getProcessWrapper("/some/bin/path", /*killDelay=*/ null);
    List<String> commandLine = processWrapper.commandLineBuilder(commandArguments).build();

    assertThat(commandLine).containsExactlyElementsIn(expectedCommandLine).inOrder();
  }

  @Test
  public void testProcessWrapperCommandLineBuilder_BuildsWithOptionalArguments()
      throws IOException {
    ImmutableList<String> commandArguments = ImmutableList.of("echo", "hello, world");

    Duration timeout = Duration.ofSeconds(10);
    Duration killDelay = Duration.ofSeconds(2);
    Path stdoutPath = testFS.getPath("/stdout.txt");
    Path stderrPath = testFS.getPath("/stderr.txt");
    Path statisticsPath = testFS.getPath("/stats.out");

    ImmutableList<String> expectedCommandLine =
        ImmutableList.<String>builder()
            .add("/path/to/process-wrapper")
            .add("--timeout=" + timeout.getSeconds())
            .add("--kill_delay=" + killDelay.getSeconds())
            .add("--stdout=" + stdoutPath)
            .add("--stderr=" + stderrPath)
            .add("--stats=" + statisticsPath)
            .addAll(commandArguments)
            .build();

    ProcessWrapper processWrapper = getProcessWrapper("/path/to/process-wrapper", killDelay);
    List<String> commandLine =
        processWrapper
            .commandLineBuilder(commandArguments)
            .setTimeout(timeout)
            .setStdoutPath(stdoutPath)
            .setStderrPath(stderrPath)
            .setStatisticsPath(statisticsPath)
            .build();

    assertThat(commandLine).containsExactlyElementsIn(expectedCommandLine).inOrder();
  }
}
