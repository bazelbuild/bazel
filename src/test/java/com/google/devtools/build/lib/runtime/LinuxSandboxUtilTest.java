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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.time.Duration;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link LinuxSandboxUtil}. */
@RunWith(JUnit4.class)
public final class LinuxSandboxUtilTest {
  @Test
  public void testLinuxSandboxCommandLineBuilder_fakeRootAndFakeUsernameAreExclusive() {
    String linuxSandboxPath = "linux-sandbox";
    ImmutableList<String> commandArguments = ImmutableList.of("echo", "hello, flo");

    Exception e =
        expectThrows(
            IllegalStateException.class,
            () ->
                LinuxSandboxUtil.commandLineBuilder(linuxSandboxPath, commandArguments)
                    .setUseFakeRoot(true)
                    .setUseFakeUsername(true)
                    .build());
    assertThat(e).hasMessageThat().contains("exclusive");
  }

  @Test
  public void testLinuxSandboxCommandLineBuilder_BuildsWithoutOptionalArguments() {
    String linuxSandboxPath = "linux-sandbox";

    ImmutableList<String> commandArguments = ImmutableList.of("echo", "hello, max");

    ImmutableList<String> expectedCommandLine =
        ImmutableList.<String>builder()
            .add(linuxSandboxPath)
            .add("--")
            .addAll(commandArguments)
            .build();

    List<String> commandLine =
        LinuxSandboxUtil.commandLineBuilder(linuxSandboxPath, commandArguments).build();

    assertThat(commandLine).containsExactlyElementsIn(expectedCommandLine).inOrder();
  }

  @Test
  public void testLinuxSandboxCommandLineBuilder_BuildsWithOptionalArguments() {
    String linuxSandboxPath = "linux-sandbox";

    ImmutableList<String> commandArguments = ImmutableList.of("echo", "hello, tom");

    Duration timeout = Duration.ofSeconds(10);
    Duration killDelay = Duration.ofSeconds(2);
    String statisticsPath = "stats.out";

    String workingDirectory = "/all-work-and-no-play";
    String stdoutPath = "stdout.txt";
    String stderrPath = "stderr.txt";

    // These two flags are exclusive.
    boolean useFakeUsername = true;
    boolean useFakeRoot = false;

    boolean createNetworkNamespace = true;
    boolean useFakeHostname = true;
    boolean useDebugMode = true;

    FileSystem fileSystem = new InMemoryFileSystem();
    Path workDir = fileSystem.getPath("/work");
    Path concreteDir = workDir.getRelative("concrete");
    Path sandboxDir = workDir.getRelative("sandbox");

    Path bindMountSource1 = concreteDir.getRelative("bindMountSource1");
    Path bindMountSource2 = concreteDir.getRelative("bindMountSource2");
    Path mountDir = sandboxDir.getRelative("mount");
    Path bindMountTarget1 = mountDir.getRelative("bindMountTarget1");
    Path bindMountTarget2 = mountDir.getRelative("bindMountTarget2");
    Path bindMountSameSourceAndTarget = mountDir.getRelative("bindMountSourceAndTarget");

    Path writableDir1 = sandboxDir.getRelative("writable1");
    Path writableDir2 = sandboxDir.getRelative("writable2");

    Path tmpfsDir1 = sandboxDir.getRelative("tmpfs1");
    Path tmpfsDir2 = sandboxDir.getRelative("tmpfs2");

    ImmutableList<Path> writableFilesAndDirectories = ImmutableList.of(writableDir1, writableDir2);

    ImmutableList<Path> tmpfsDirectories = ImmutableList.of(tmpfsDir1, tmpfsDir2);

    ImmutableSortedMap<Path, Path> bindMounts =
        ImmutableSortedMap.<Path, Path>naturalOrder()
            .put(bindMountTarget1, bindMountSource1)
            .put(bindMountTarget2, bindMountSource2)
            .put(bindMountSameSourceAndTarget, bindMountSameSourceAndTarget)
            .build();

    ImmutableList<String> expectedCommandLine =
        ImmutableList.<String>builder()
            .add(linuxSandboxPath)
            .add("-W", workingDirectory)
            .add("-T", Long.toString(timeout.getSeconds()))
            .add("-t", Long.toString(killDelay.getSeconds()))
            .add("-l", stdoutPath)
            .add("-L", stderrPath)
            .add("-w", writableDir1.getPathString())
            .add("-w", writableDir2.getPathString())
            .add("-e", tmpfsDir1.getPathString())
            .add("-e", tmpfsDir2.getPathString())
            .add("-M", bindMountSameSourceAndTarget.getPathString())
            .add("-M", bindMountSource1.getPathString())
            .add("-m", bindMountTarget1.getPathString())
            .add("-M", bindMountSource2.getPathString())
            .add("-m", bindMountTarget2.getPathString())
            .add("-S", statisticsPath)
            .add("-H")
            .add("-N")
            .add("-U")
            .add("-D")
            .add("--")
            .addAll(commandArguments)
            .build();

    List<String> commandLine =
        LinuxSandboxUtil.commandLineBuilder(linuxSandboxPath, commandArguments)
            .setWorkingDirectory(workingDirectory)
            .setStdoutPath(stdoutPath)
            .setStderrPath(stderrPath)
            .setTimeout(timeout)
            .setKillDelay(killDelay)
            .setWritableFilesAndDirectories(writableFilesAndDirectories)
            .setTmpfsDirectories(tmpfsDirectories)
            .setBindMounts(bindMounts)
            .setUseFakeHostname(useFakeHostname)
            .setCreateNetworkNamespace(createNetworkNamespace)
            .setUseFakeRoot(useFakeRoot)
            .setStatisticsPath(statisticsPath)
            .setUseFakeUsername(useFakeUsername)
            .setUseDebugMode(useDebugMode)
            .build();

    assertThat(commandLine).containsExactlyElementsIn(expectedCommandLine).inOrder();
  }
}
