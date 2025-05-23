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


import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.StandardOpenOption;

/**
 * Manages shared or exclusive access to the filesystem by concurrent processes through a lock file.
 */
public final class FileSystemLock implements AutoCloseable {
  private final FileChannel channel;
  private final FileLock lock;

  /**
   * The exception thrown when a lock cannot be acquired because it is already exclusively held by
   * another process.
   */
  public static class LockAlreadyHeldException extends IOException {
    LockAlreadyHeldException(LockMode mode, Path path) {
      super("failed to acquire %s filesystem lock on %s".formatted(mode, path));
    }
  }

  /** The mode of a lock. */
  public enum LockMode {
    SHARED,
    EXCLUSIVE;

    @Override
    public String toString() {
      return switch (this) {
        case SHARED -> "shared";
        case EXCLUSIVE -> "exclusive";
      };
    }
  }

  private FileSystemLock(FileChannel channel, FileLock lock) {
    this.channel = channel;
    this.lock = lock;
  }

  private static FileChannel prepareChannel(Path path) throws IOException {
    path.getParentDirectory().createDirectoryAndParents();
    return FileChannel.open(
        // Correctly handle non-ASCII paths by converting from the internal string encoding.
        java.nio.file.Path.of(StringEncoding.internalToPlatform(path.getPathString())),
        StandardOpenOption.READ,
        StandardOpenOption.WRITE,
        StandardOpenOption.CREATE);
  }

  /**
   * Tries to acquires a lock on the given path with the given mode. Throws an exception if the lock
   * is already held by another process.
   *
   * @throws LockAlreadyHeldException if the lock is already exclusively held by another process
   * @throws IOException if another error occurred
   */
  public static FileSystemLock tryGet(Path path, LockMode mode) throws IOException {
    FileChannel channel = prepareChannel(path);
    FileLock lock = channel.tryLock(0, Long.MAX_VALUE, mode == LockMode.SHARED);
    if (lock == null) {
      throw new LockAlreadyHeldException(mode, path);
    }
    return new FileSystemLock(channel, lock);
  }

  /** Acquires a lock on the given path with the given mode. Blocks until the lock is acquired. */
  public static FileSystemLock get(Path path, LockMode mode) throws IOException {
    FileChannel channel = prepareChannel(path);
    FileLock lock = channel.lock(0, Long.MAX_VALUE, mode == LockMode.SHARED);
    return new FileSystemLock(channel, lock);
  }

  @VisibleForTesting
  boolean isShared() {
    return lock.isShared();
  }

  @VisibleForTesting
  boolean isExclusive() {
    return !isShared();
  }

  /** Releases access to the lock file. */
  @Override
  public void close() throws IOException {
    try {
      lock.release();
    } finally {
      channel.close();
    }
  }
}
