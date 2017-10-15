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

import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Set;

/**
 * A directory status that signifies a path has left this file system's
 * scope. All methods beside {@link #outOfScope} and {@link #getEscapingPath}
 * are disabled.
 */
final class OutOfScopeDirectoryStatus extends InMemoryDirectoryInfo {
  /**
   * Contains the requested path resolved up to the point where it
   * first escapes the scope. See
   * {@link ScopeEscapableStatus#getEscapingPath} for an example.
   */
  private final PathFragment escapingPath;

  public OutOfScopeDirectoryStatus(PathFragment escapingPath) {
    super(null, false);
    this.escapingPath = escapingPath;
  }

  @Override
  public boolean outOfScope() {
    return true;
  }

  @Override
  public PathFragment getEscapingPath() {
    return escapingPath;
  }

  private static UnsupportedOperationException failure() {
    return new UnsupportedOperationException();
  }

  @Override public boolean isDirectory() {
    throw failure();
  }

  @Override public boolean isSymbolicLink() {
    throw failure();
  }

  @Override public boolean isFile() {
    throw failure();
  }

  @Override public long getSize() {
    throw failure();
  }

  @Override protected synchronized void markModificationTime() {
    throw failure();
  }

  @Override public synchronized long getLastModifiedTime() {
    throw failure();
  }

  @Override synchronized void setLastModifiedTime(long newTime) {
    throw failure();
  }

  @Override public synchronized long getLastChangeTime() {
    throw failure();
  }

  @Override boolean isReadable() {
    throw failure();
  }

  @Override void setReadable(boolean readable) {
    throw failure();
  }

  @Override void setWritable(boolean writable) {
    throw failure();
  }

  @Override void setExecutable(boolean executable) {
    throw failure();
  }

  @Override boolean isWritable() {
    throw failure();
  }

  @Override boolean isExecutable() {
    throw failure();
  }

  @Override synchronized void addChild(String name, InMemoryContentInfo inode) {
    throw failure();
  }

  @Override synchronized InMemoryContentInfo getChild(String name) {
    throw failure();
  }

  @Override synchronized void removeChild(String name) {
    throw failure();
  }

  @Override Set<String> getAllChildren() {
    throw failure();
  }
}
