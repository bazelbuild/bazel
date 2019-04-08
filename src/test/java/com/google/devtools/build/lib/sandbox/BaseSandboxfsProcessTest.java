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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.sandbox.SandboxfsProcess.Mapping;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Common tests for all implementations of {@link SandboxfsProcess}.
 *
 * <p>Subclasses must define the provided hooks to configure the file system the tests run in
 * (which can be real or virtual), and a mechanism to "mount" a sandboxfs instance.
 *
 * <p>Subclasses inherit and run all the tests in this class.
 */
abstract class BaseSandboxfsProcessTest {

  /** Test-specific temporary directory and file system. */
  protected Path tmpDir;

  /** Hook to obtain the path to a test-specific temporary directory and file system. */
  abstract Path newTmpDir() throws IOException;

  /** Hook to mount a new test-specific sandboxfs instance. */
  abstract SandboxfsProcess mount(Path mountPoint) throws IOException;

  @Before
  public void setUp() throws IOException {
    tmpDir = newTmpDir();
  }

  @After
  public void tearDown() throws IOException {
    tmpDir.deleteTreesBelow();
    tmpDir = null;
  }

  @Test
  public void testMount_MissingDirectory() throws IOException {
    IOException expected = assertThrows(
        IOException.class, () -> mount(tmpDir.getRelative("missing")));
    assertThat(expected).hasMessageThat().matches(".*(/missing.*does not exist|failed to start).*");
  }

  @Test
  public void testLifeCycle() throws IOException {
    Path mountPoint = tmpDir.getRelative("mnt");
    mountPoint.createDirectory();
    SandboxfsProcess process = mount(mountPoint);
    try {
      assertThat(process.isAlive()).isTrue();
      process.destroy();
      assertThat(process.isAlive()).isFalse();
      process.destroy();
      assertThat(process.isAlive()).isFalse();
    } finally {
      process.destroy();
    }
  }

  @Test
  public void testReconfigure() throws IOException {
    Path mountPoint = tmpDir.getRelative("mnt");
    mountPoint.createDirectory();
    SandboxfsProcess process = mount(mountPoint);
    try {
      // Start by ensuring the mount point is empty.
      assertThat(mountPoint.getDirectoryEntries()).isEmpty();

      // Create a file outside of the mount point to ensure it's not touched.
      mountPoint.getRelative("../unrelated").createDirectory();

      // Create twp mappings: one to be deleted and one to be kept around throughout the test.
      Path keepMeFile = tmpDir.getRelative("one");
      FileSystemUtils.createEmptyFile(keepMeFile);
      Path oneFile = tmpDir.getRelative("one");
      FileSystemUtils.writeContent(oneFile, UTF_8, "One test data");
      process.map(
          ImmutableList.of(
              Mapping.builder()
                  .setPath(PathFragment.create("/keep-me"))
                  .setTarget(keepMeFile.asFragment())
                  .setWritable(false)
                  .build(),
              Mapping.builder()
                  .setPath(PathFragment.create("/foo"))
                  .setTarget(oneFile.asFragment())
                  .setWritable(false)
                  .build()));
      assertThat(
          mountPoint.getDirectoryEntries())
          .containsExactly(mountPoint.getRelative("foo"), mountPoint.getRelative("keep-me"));
      assertThat(
          FileSystemUtils.readContent(mountPoint.getRelative("foo"), UTF_8))
          .isEqualTo("One test data");

      // Replace the previous mapping and create a new one.
      Path twoFile = tmpDir.getRelative("two");
      FileSystemUtils.writeContent(twoFile, UTF_8, "Two test data");
      Path bazFile = tmpDir.getRelative("baz");
      FileSystemUtils.writeContent(bazFile, UTF_8, "Baz test data");
      process.unmap(PathFragment.create("/foo"));
      process.map(
          ImmutableList.of(
              Mapping.builder()
                  .setPath(PathFragment.create("/foo"))
                  .setTarget(twoFile.asFragment())
                  .setWritable(false)
                  .build(),
              Mapping.builder()
                  .setPath(PathFragment.create("/bar"))
                  .setTarget(bazFile.asFragment())
                  .setWritable(true)
                  .build()));
      assertThat(
          mountPoint.getDirectoryEntries())
          .containsExactly(mountPoint.getRelative("foo"), mountPoint.getRelative("bar"),
              mountPoint.getRelative("keep-me"));
      assertThat(
          FileSystemUtils.readContent(mountPoint.getRelative("foo"), UTF_8))
          .isEqualTo("Two test data");
      assertThat(
          FileSystemUtils.readContent(mountPoint.getRelative("bar"), UTF_8))
          .isEqualTo("Baz test data");

      // Replace all existing mappings, and try with a nested one.
      Path longLink = tmpDir.getRelative("long/link");
      longLink.getParentDirectory().createDirectoryAndParents();
      longLink.createSymbolicLink(oneFile);  // The target is irrelevant but must exist.
      process.unmap(PathFragment.create("/foo"));
      process.unmap(PathFragment.create("/bar"));
      process.map(
          ImmutableList.of(
              Mapping.builder()
                  .setPath(PathFragment.create("/something/complex"))
                  .setTarget(longLink.asFragment())
                  .setWritable(false)
                  .build()));
      assertThat(
          mountPoint.getDirectoryEntries())
          .containsExactly(mountPoint.getRelative("keep-me"), mountPoint.getRelative("something"));
      assertThat(
          FileSystemUtils.readContent(mountPoint.getRelative("something/complex"), UTF_8))
          .isEqualTo("One test data");

      // Ensure that files that should not have been touched throughout the test are still there.
      assertThat(mountPoint.getRelative("keep-me").exists()).isTrue();
      assertThat(mountPoint.getRelative("../unrelated").exists()).isTrue();
    } finally {
      process.destroy();
    }
  }
}
