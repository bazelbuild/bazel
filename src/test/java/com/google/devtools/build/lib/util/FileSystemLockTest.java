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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.testutil.ExternalFileSystemLock;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.FileSystemLock.LockAlreadyHeldException;
import com.google.devtools.build.lib.util.FileSystemLock.LockMode;
import com.google.devtools.build.lib.vfs.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileSystemLock}. */
@RunWith(JUnit4.class)
public final class FileSystemLockTest {

  private Path lockPath;

  @Before
  public void setUp() throws Exception {
    var rootDir = TestUtils.createUniqueTmpDir(null);
    lockPath = rootDir.getRelative("subdir/lock");
  }

  @Test
  public void tryGet_shared_whenNotLocked_succeeds() throws Exception {
    try (var lock = FileSystemLock.tryGet(lockPath, LockMode.SHARED)) {
      assertThat(lock.isShared()).isTrue();
    }
  }

  @Test
  public void get_shared_whenNotLocked_succeeds() throws Exception {
    try (var lock = FileSystemLock.get(lockPath, LockMode.SHARED)) {
      assertThat(lock.isShared()).isTrue();
    }
  }

  @Test
  public void tryGet_shared_whenLockedForSharedUse_succeeds() throws Exception {
    try (var externalLock = ExternalFileSystemLock.getShared(lockPath);
        var lock = FileSystemLock.tryGet(lockPath, LockMode.SHARED)) {
      assertThat(lock.isShared()).isTrue();
    }
  }

  @Test
  public void get_shared_whenLockedForSharedUse_succeeds() throws Exception {
    try (var externalLock = ExternalFileSystemLock.getShared(lockPath);
        var lock = FileSystemLock.get(lockPath, LockMode.SHARED)) {
      assertThat(lock.isShared()).isTrue();
    }
  }

  @Test
  public void tryGet_shared_whenLockedForExclusiveUse_fails() throws Exception {
    try (var externalLock = ExternalFileSystemLock.getExclusive(lockPath)) {
      LockAlreadyHeldException e =
          assertThrows(
              LockAlreadyHeldException.class,
              () -> FileSystemLock.tryGet(lockPath, LockMode.SHARED));
      assertThat(e).hasMessageThat().contains("failed to acquire shared filesystem lock");
    }
  }

  @Test
  public void get_shared_whenLockedForExclusiveUse_blocks() throws Exception {
    testBlocks(ExternalFileSystemLock.getExclusive(lockPath), LockMode.SHARED);
  }

  @Test
  public void tryGet_exclusive_whenNotLocked_succeeds() throws Exception {
    try (var lock = FileSystemLock.tryGet(lockPath, LockMode.EXCLUSIVE)) {
      assertThat(lock.isExclusive()).isTrue();
    }
  }

  @Test
  public void get_exclusive_whenNotLocked_succeeds() throws Exception {
    try (var lock = FileSystemLock.get(lockPath, LockMode.EXCLUSIVE)) {
      assertThat(lock.isExclusive()).isTrue();
    }
  }

  @Test
  public void tryGet_exclusive_whenLockedForSharedUse_fails() throws Exception {
    try (var externalLock = ExternalFileSystemLock.getShared(lockPath)) {
      LockAlreadyHeldException e =
          assertThrows(
              LockAlreadyHeldException.class,
              () -> FileSystemLock.tryGet(lockPath, LockMode.EXCLUSIVE));
      assertThat(e).hasMessageThat().contains("failed to acquire exclusive filesystem lock");
    }
  }

  @Test
  public void get_exclusive_whenLockedForSharedUse_blocks() throws Exception {
    testBlocks(ExternalFileSystemLock.getShared(lockPath), LockMode.EXCLUSIVE);
  }

  @Test
  public void tryGet_exclusive_whenLockedForExclusiveUse_fails() throws Exception {
    try (var lock = ExternalFileSystemLock.getExclusive(lockPath)) {
      LockAlreadyHeldException e =
          assertThrows(
              LockAlreadyHeldException.class,
              () -> FileSystemLock.tryGet(lockPath, LockMode.EXCLUSIVE));
      assertThat(e).hasMessageThat().contains("failed to acquire exclusive filesystem lock");
    }
  }

  @Test
  public void get_exclusive_whenLockedForExclusiveUse_blocks() throws Exception {
    testBlocks(ExternalFileSystemLock.getExclusive(lockPath), LockMode.EXCLUSIVE);
  }

  private void testBlocks(ExternalFileSystemLock externalLock, LockMode mode) throws Exception {
    Future<Boolean> future;
    try {
      var latch = new CountDownLatch(1);
      var externalLockReleased = new AtomicBoolean();
      future =
          Executors.newSingleThreadExecutor()
              .submit(
                  () -> {
                    latch.countDown();
                    try (var lock = FileSystemLock.get(lockPath, mode)) {
                      return externalLockReleased.get();
                    }
                  });
      latch.await();
      Thread.sleep(1);
      externalLockReleased.set(true);
    } finally {
      externalLock.close();
    }
    assertThat(future.get()).isTrue();
  }
}
