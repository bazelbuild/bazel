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
package com.google.devtools.build.lib.remote.disk;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DiskCacheLock}. */
@RunWith(JUnit4.class)
public final class DiskCacheLockTest {

  private Path lockPath;

  @Before
  public void setUp() throws Exception {
    var rootDir = TestUtils.createUniqueTmpDir(null);
    lockPath = rootDir.getRelative("subdir/lock");
  }

  @Test
  public void getShared_whenNotLocked_succeeds() throws Exception {
    try (var lock = DiskCacheLock.getShared(lockPath)) {
      assertThat(lock.isShared()).isTrue();
    }
  }

  @Test
  public void getShared_whenLockedForSharedUse_succeeds() throws Exception {
    try (var externalLock = ExternalLock.getShared(lockPath);
        var lock = DiskCacheLock.getShared(lockPath)) {
      assertThat(lock.isShared()).isTrue();
    }
  }

  @Test
  public void getShared_whenLockedForExclusiveUse_fails() throws Exception {
    try (var externalLock = ExternalLock.getExclusive(lockPath)) {
      IOException e = assertThrows(IOException.class, () -> DiskCacheLock.getShared(lockPath));
      assertThat(e).hasMessageThat().contains("failed to acquire shared disk cache lock");
    }
  }

  @Test
  public void getExclusive_whenNotLocked_succeeds() throws Exception {
    try (var lock = DiskCacheLock.getExclusive(lockPath)) {
      assertThat(lock.isExclusive()).isTrue();
    }
  }

  @Test
  public void getExclusive_whenLockedForSharedUse_fails() throws Exception {
    try (var externalLock = ExternalLock.getShared(lockPath)) {
      IOException e = assertThrows(IOException.class, () -> DiskCacheLock.getExclusive(lockPath));
      assertThat(e).hasMessageThat().contains("failed to acquire exclusive disk cache lock");
    }
  }

  @Test
  public void getExclusive_whenLockedForExclusiveUse_fails() throws Exception {
    try (var lock = ExternalLock.getExclusive(lockPath)) {
      IOException e = assertThrows(IOException.class, () -> DiskCacheLock.getExclusive(lockPath));
      assertThat(e).hasMessageThat().contains("failed to acquire exclusive disk cache lock");
    }
  }

  /**
   * Runs an external process that holds a shared or exclusive lock on a file.
   *
   * <p>This is needed because the JVM does not allow overlapping locks.
   */
  private static class ExternalLock implements AutoCloseable {
    private static final String HELPER_PATH =
        "io_bazel/src/test/java/com/google/devtools/build/lib/remote/disk/external_lock_helper"
            + (OS.getCurrent() == OS.WINDOWS ? ".exe" : "");

    private final Subprocess subprocess;

    static ExternalLock getShared(Path lockPath) throws IOException {
      return new ExternalLock(lockPath, true);
    }

    static ExternalLock getExclusive(Path lockPath) throws IOException {
      return new ExternalLock(lockPath, false);
    }

    ExternalLock(Path lockPath, boolean shared) throws IOException {
      String binaryPath = Runfiles.preload().withSourceRepository("").rlocation(HELPER_PATH);
      this.subprocess =
          new SubprocessBuilder()
              .setArgv(
                  ImmutableList.of(
                      binaryPath, lockPath.getPathString(), shared ? "shared" : "exclusive"))
              .start();
      // Wait for child to report that the lock has been acquired.
      // We could read the entire stdout/stderr here to obtain additional debugging information,
      // but for some reason that hangs forever on Windows, even if we close them on the child side.
      if (subprocess.getInputStream().read() != '!') {
        throw new IOException("external helper process failed");
      }
    }

    @Override
    public void close() throws IOException {
      // Wait for process to exit and release the lock.
      subprocess.destroyAndWait();
    }
  }
}
