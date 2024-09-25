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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.StandardOpenOption;

/** Manages shared or exclusive access to the disk cache by concurrent processes. */
public final class DiskCacheLock implements AutoCloseable {
  private final FileChannel channel;
  private final FileLock lock;

  private DiskCacheLock(FileChannel channel, FileLock lock) {
    this.channel = channel;
    this.lock = lock;
  }

  /**
   * Acquires shared access to the disk cache.
   *
   * @param path the path to the lock file
   * @throws IOException if an error occurred, including the lock currently being exclusively held
   *     by another process
   */
  public static DiskCacheLock getShared(Path path) throws IOException {
    return get(path, true);
  }

  /**
   * Acquires exclusive access to the disk cache.
   *
   * @param path the path to the lock file
   * @throws IOException if an error occurred, including the lock currently being exclusively held
   *     by another process
   */
  public static DiskCacheLock getExclusive(Path path) throws IOException {
    return get(path, false);
  }

  private static DiskCacheLock get(Path path, boolean shared) throws IOException {
    path.getParentDirectory().createDirectoryAndParents();
    FileChannel channel =
        FileChannel.open(
            java.nio.file.Path.of(path.getPathString()),
            StandardOpenOption.READ,
            StandardOpenOption.WRITE,
            StandardOpenOption.CREATE);
    FileLock lock = channel.tryLock(0, Long.MAX_VALUE, shared);
    if (lock == null) {
      throw new IOException(
          "failed to acquire %s disk cache lock".formatted(shared ? "shared" : "exclusive"));
    }
    return new DiskCacheLock(channel, lock);
  }

  @VisibleForTesting
  boolean isShared() {
    return lock.isShared();
  }

  @VisibleForTesting
  boolean isExclusive() {
    return !isShared();
  }

  /** Releases access to the disk cache. */
  @Override
  public void close() throws IOException {
    try {
      lock.release();
    } finally {
      channel.close();
    }
  }
}
