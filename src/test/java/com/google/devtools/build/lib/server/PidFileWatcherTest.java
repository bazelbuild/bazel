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

package com.google.devtools.build.lib.server;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PidFileWatcher}. */
@RunWith(JUnit4.class)
public class PidFileWatcherTest {

  private static final int EXPECTED_PID = 42;
  private static final IllegalStateException THROWN_ON_HALT = new IllegalStateException("crash!");

  private Path pidFile;
  private PidFileWatcher underTest;

  @Before
  public void setUp() {
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    pidFile = fileSystem.getPath("/pid");
    underTest =
        new PidFileWatcher(
            pidFile,
            EXPECTED_PID,
            () -> {
              throw THROWN_ON_HALT;
            });
  }

  @Test
  public void testMissingPidFileHaltsProgram() throws IOException {
    // Delete just in case.
    pidFile.delete();

    assertPidCheckHaltsProgram();
  }

  @Test
  @SuppressWarnings("DoNotCall") // testing that the watcher exits without erroring
  public void testStoppedWatcherDoesNotHaltProgram() throws IOException {
    underTest.endWatch();
    pidFile.delete();

    underTest.run();
  }

  @Test
  public void testEmptyPidFileCountsAsChanged() throws IOException {
    FileSystemUtils.writeContent(pidFile, new byte[0]);

    assertPidCheckHaltsProgram();
  }

  @Test
  public void testGarbagePidFileCountsAsChanged() throws IOException {
    FileSystemUtils.writeContent(pidFile, "junk".getBytes(US_ASCII));

    assertPidCheckHaltsProgram();
  }

  @Test
  public void testPidFileContinuesExecution() throws IOException {
    FileSystemUtils.writeContent(pidFile, "42".getBytes(US_ASCII));

    assertThat(underTest.runPidFileChecks()).isTrue();
  }

  @Test
  public void testPidFileTrailingWhitespaceNotTolerated() throws IOException {
    FileSystemUtils.writeContent(pidFile, "42\n".getBytes(US_ASCII));

    assertPidCheckHaltsProgram();
  }

  @Test
  public void testPidFileChangeAfterShutdownNotificationStopsWatcher() throws IOException {
    FileSystemUtils.writeContent(pidFile, "42\n".getBytes(US_ASCII));

    underTest.signalShutdown();
    assertThat(underTest.runPidFileChecks()).isFalse();
  }

  private void assertPidCheckHaltsProgram() {
    IllegalStateException expected =
        assertThrows(IllegalStateException.class, underTest::runPidFileChecks);
    assertThat(expected).isSameInstanceAs(THROWN_ON_HALT);
  }
}
