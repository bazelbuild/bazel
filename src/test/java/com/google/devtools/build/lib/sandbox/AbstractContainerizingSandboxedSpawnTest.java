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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxedSpawn}. */
@RunWith(JUnit4.class)
public class AbstractContainerizingSandboxedSpawnTest {

  @Test
  public void testMoveOutputs() throws Exception {
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path testRoot = fileSystem.getPath(TestUtils.tmpDir());
    testRoot.createDirectoryAndParents();

    Path execRoot = testRoot.getRelative("execroot");
    execRoot.createDirectory();

    Path outputFile = execRoot.getRelative("very/output.txt");
    Path outputLink = execRoot.getRelative("very/output.link");
    Path outputDangling = execRoot.getRelative("very/output.dangling");
    Path outputDir = execRoot.getRelative("very/output.dir");
    Path outputInUncreatedTargetDir = execRoot.getRelative("uncreated/output.txt");

    ImmutableSet<PathFragment> outputs =
        ImmutableSet.of(
            outputFile.relativeTo(execRoot),
            outputLink.relativeTo(execRoot),
            outputDangling.relativeTo(execRoot),
            outputInUncreatedTargetDir.relativeTo(execRoot));
    ImmutableSet<PathFragment> outputDirs = ImmutableSet.of(outputDir.relativeTo(execRoot));
    for (PathFragment path : outputs) {
      execRoot.getRelative(path).getParentDirectory().createDirectoryAndParents();
    }
    for (PathFragment path : outputDirs) {
      execRoot.getRelative(path).createDirectoryAndParents();
    }

    FileSystemUtils.createEmptyFile(outputFile);
    outputLink.createSymbolicLink(PathFragment.create("output.txt"));
    outputDangling.createSymbolicLink(PathFragment.create("doesnotexist"));
    outputDir.createDirectory();
    FileSystemUtils.createEmptyFile(outputDir.getRelative("test.txt"));
    FileSystemUtils.createEmptyFile(outputInUncreatedTargetDir);

    Path outputsDir = testRoot.getRelative("outputs");
    outputsDir.createDirectory();
    outputsDir.getRelative("very").createDirectory();
    SandboxHelpers.moveOutputs(SandboxOutputs.create(outputs, outputDirs), execRoot, outputsDir);

    assertThat(outputsDir.getRelative("very/output.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
    assertThat(outputsDir.getRelative("very/output.link").isSymbolicLink()).isTrue();
    assertThat(outputsDir.getRelative("very/output.link").resolveSymbolicLinks())
        .isEqualTo(outputsDir.getRelative("very/output.txt"));
    assertThat(outputsDir.getRelative("very/output.dangling").isSymbolicLink()).isTrue();
    assertThrows(
        IOException.class,
        () -> outputsDir.getRelative("very/output.dangling").resolveSymbolicLinks());
    assertThat(outputsDir.getRelative("very/output.dir").isDirectory(Symlinks.NOFOLLOW)).isTrue();
    assertThat(outputsDir.getRelative("very/output.dir/test.txt").isFile(Symlinks.NOFOLLOW))
        .isTrue();
    assertThat(outputsDir.getRelative("uncreated/output.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
  }

  /** Watches a logger for file copy warnings (instead of moves) and counts them. */
  private static class FileCopyWarningTracker extends Handler {
    int warningsCounter = 0;

    @Override
    public void publish(LogRecord record) {
      if (record.getMessage().contains("different file systems")) {
        warningsCounter++;
      }
    }

    @Override
    public void flush() {}

    @Override
    public void close() {}
  }

  @Test
  public void testMoveOutputs_warnOnceIfCopyHappened() throws Exception {
    class MultipleDeviceFS extends InMemoryFileSystem {
      MultipleDeviceFS() {
        super(DigestHashFunction.SHA256);
      }

      @Override
      public void renameTo(Path source, Path target) throws IOException {
        throw new IOException("EXDEV");
      }
    }
    FileSystem fileSystem = new MultipleDeviceFS();
    Path testRoot = fileSystem.getPath(TestUtils.tmpDir());
    testRoot.createDirectoryAndParents();

    FileCopyWarningTracker tracker = new FileCopyWarningTracker();
    Logger logger = Logger.getLogger(SandboxHelpers.class.getName());
    logger.setUseParentHandlers(false);
    logger.addHandler(tracker);

    Path execRoot = testRoot.getRelative("execroot");
    execRoot.createDirectory();

    Path outputFile1 = execRoot.getRelative("very/output1.txt");
    Path outputFile2 = execRoot.getRelative("much/output2.txt");

    ImmutableSet<PathFragment> outputs =
        ImmutableSet.of(outputFile1.relativeTo(execRoot), outputFile2.relativeTo(execRoot));
    for (PathFragment path : outputs) {
      execRoot.getRelative(path).getParentDirectory().createDirectoryAndParents();
    }

    FileSystemUtils.createEmptyFile(outputFile1);
    FileSystemUtils.createEmptyFile(outputFile2);

    Path outputsDir = testRoot.getRelative("outputs");
    outputsDir.createDirectory();
    outputsDir.getRelative("very").createDirectory();
    outputsDir.getRelative("much").createDirectory();
    SandboxHelpers.moveOutputs(
        SandboxOutputs.create(outputs, ImmutableSet.of()), execRoot, outputsDir);

    assertThat(outputsDir.getRelative("very/output1.txt").isFile(Symlinks.NOFOLLOW)).isTrue();
    assertThat(outputsDir.getRelative("much/output2.txt").isFile(Symlinks.NOFOLLOW)).isTrue();

    assertThat(tracker.warningsCounter).isEqualTo(1);
  }
}
