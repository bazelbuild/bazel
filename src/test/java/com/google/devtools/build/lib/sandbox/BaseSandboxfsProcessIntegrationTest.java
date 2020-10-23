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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

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
 * <p>Subclasses must define the provided hooks to configure the file system the tests run in (which
 * can be real or virtual), and a mechanism to "mount" a sandboxfs instance.
 *
 * <p>Subclasses inherit and run all the tests in this class.
 */
abstract class BaseSandboxfsProcessIntegrationTest {

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
  public void testMount_missingDirectory() throws IOException {
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

      // Create first sandbox.
      Path oneFile = tmpDir.getRelative("one");
      FileSystemUtils.writeContent(oneFile, UTF_8, "One test data");
      process.createSandbox(
          "first",
          (mapper) -> mapper.map(PathFragment.create("/foo"), oneFile.asFragment(), false));
      Path first = mountPoint.getRelative("first");
      assertThat(mountPoint.getDirectoryEntries()).containsExactly(first);
      assertThat(first.getDirectoryEntries()).containsExactly(first.getRelative("foo"));
      assertThat(FileSystemUtils.readContent(first.getRelative("foo"), UTF_8))
          .isEqualTo("One test data");

      // Create second sandbox, which is expected to be fully disjoint from the first one.
      Path twoFile = tmpDir.getRelative("two");
      FileSystemUtils.writeContent(twoFile, UTF_8, "Two test data");
      Path longLink = tmpDir.getRelative("long/link");
      longLink.getParentDirectory().createDirectoryAndParents();
      longLink.createSymbolicLink(oneFile); // The target is irrelevant but must exist.
      process.createSandbox(
          "second",
          (mapper) -> {
            mapper.map(PathFragment.create("/foo"), twoFile.asFragment(), false);
            mapper.map(PathFragment.create("/something/complex"), longLink.asFragment(), false);
          });
      Path second = mountPoint.getRelative("second");
      assertThat(mountPoint.getDirectoryEntries()).containsExactly(first, second);
      assertThat(second.getDirectoryEntries())
          .containsExactly(second.getRelative("foo"), second.getRelative("something"));
      assertThat(FileSystemUtils.readContent(first.getRelative("foo"), UTF_8))
          .isEqualTo("One test data");
      assertThat(FileSystemUtils.readContent(second.getRelative("foo"), UTF_8))
          .isEqualTo("Two test data");
      assertThat(FileSystemUtils.readContent(second.getRelative("something/complex"), UTF_8))
          .isEqualTo("One test data");

      // Destroy one sandbox and ensure the other remains.
      process.destroySandbox("first");
      assertThat(mountPoint.getDirectoryEntries()).containsExactly(second);

      // Ensure that files that should not have been touched throughout the test are still there.
      assertThat(mountPoint.getRelative("../unrelated").exists()).isTrue();
    } finally {
      process.destroy();
    }
  }
}
