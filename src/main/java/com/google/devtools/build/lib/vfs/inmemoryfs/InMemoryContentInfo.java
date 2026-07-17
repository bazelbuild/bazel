// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem.InodeOrErrno;
import javax.annotation.concurrent.GuardedBy;

/**
 * This interface defines the function directly supported by the "files" stored in a
 * InMemoryFileSystem. This corresponds to a file or inode in UNIX: it doesn't have a path (it could
 * have many paths due to hard links, or none if it's unlinked, i.e. garbage).
 *
 * <p>This class is thread-safe: instances may be accessed and modified from concurrent threads.
 * Subclasses must preserve this property.
 */
@ThreadSafe
public abstract non-sealed class InMemoryContentInfo implements FileStatus, InodeOrErrno {

  protected final Clock clock;

  /**
   * Stores the time when the file was last modified. This is atomically updated whenever the file
   * changes, so all accesses must be synchronized.
   */
  @GuardedBy("this")
  private long lastModifiedTime;

  /**
   * Stores the time when the file information was changed. This is atomically updated whenever the
   * file changes, so all accesses must be synchronized.
   */
  @GuardedBy("this")
  private long lastChangeTime;

  /** Stores the file's permission bits. */
  @GuardedBy("this")
  private int permissions = 0644;

  protected InMemoryContentInfo(Clock clock) {
    this.clock = checkNotNull(clock, "clock");
    // When we create the file, it is modified.
    markModificationTime();
  }

  /**
   * Returns true if the current object is a directory.
   */
  @Override
  public abstract boolean isDirectory();

  /**
   * Returns true if the current object is a symbolic link.
   */
  @Override
  public abstract boolean isSymbolicLink();

  /**
   * Returns true if the current object is a regular or special file.
   */
  @Override
  public abstract boolean isFile();

  /**
   * Returns true if the current object is a special file.
   */
  @Override
  public abstract boolean isSpecialFile();

  /**
   * Returns the size of the entity denoted by the current object. For files,
   * this is the length in bytes, for directories the number of children. The
   * size of links is unspecified.
   */
  @Override
  public abstract long getSize();

  /**
   * Returns the time when the entity denoted by the current object was last
   * modified.
   */
  @Override
  public synchronized long getLastModifiedTime() {
    return lastModifiedTime;
  }

  /**
   * Returns the time when the entity denoted by the current object was last
   * changed.
   */
  @Override
  public synchronized long getLastChangeTime() {
    return lastChangeTime;
  }

  /**
   * Returns the file node id for the given instance, emulated by the
   * identity hash code.
   */
  @Override
  public long getNodeId() {
    return System.identityHashCode(this);
  }

  @Override
  public final synchronized int getPermissions() {
    return permissions;
  }

  @Override
  public final InMemoryContentInfo inodeOrThrow(PathFragment path) {
    return this;
  }

  /**
   * Sets the time that denotes when the entity denoted by this object was last
   * modified.
   */
  synchronized void setLastModifiedTime(long newTime) {
    lastModifiedTime = newTime;
    markChangeTime();
  }

  /** Sets the last modification and change times to the current time. */
  synchronized void markModificationTime() {
    lastModifiedTime = clock.currentTimeMillis();
    lastChangeTime = lastModifiedTime;
  }

  /** Sets the last change time to the current time. */
  private synchronized void markChangeTime() {
    lastChangeTime = clock.currentTimeMillis();
  }

  /** Returns whether the current file is readable. */
  boolean isReadable() {
    return checkPermissions(0400);
  }

  /** Sets whether the current file is readable. */
  void setReadable(boolean readable) {
    updatePermissions(0400, readable);
  }

  /** Returns whether the current file is writable. */
  boolean isWritable() {
    return checkPermissions(0200);
  }

  /** Sets whether the current file is writable. */
  void setWritable(boolean writable) {
    updatePermissions(0200, writable);
  }

  /** Returns whether the current file is executable. */
  boolean isExecutable() {
    return checkPermissions(0100);
  }

  /** Sets whether the current file is executable. */
  void setExecutable(boolean executable) {
    updatePermissions(0111, executable);
  }

  /** Sets the permissions on the current file. */
  synchronized void chmod(int permissions) {
    this.permissions = permissions;
    markChangeTime();
  }

  private synchronized boolean checkPermissions(int mask) {
    return (permissions & mask) != 0;
  }

  private synchronized void updatePermissions(int mask, boolean set) {
    chmod(set ? (permissions | mask) : (permissions & ~mask));
  }

  InMemoryDirectoryInfo asDirectory() {
    throw new IllegalStateException("Not a directory: " + this);
  }
}
