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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.clock.Clock;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Represents a directory stored in an {@link InMemoryFileSystem}.
 *
 * <p>Not thread-safe. Access should be synchronized from the referencing {@link
 * InMemoryFileSystem}.
 */
final class InMemoryDirectoryInfo extends InMemoryContentInfo {

  private final Map<String, InMemoryContentInfo> directoryContent = new HashMap<>();

  InMemoryDirectoryInfo(Clock clock) {
    super(clock);
    setExecutable(true);
  }

  /**
   * Adds a new child to this directory under the given name. Callers must ensure that no entry of
   * that name exists already.
   */
  void addChild(String name, InMemoryContentInfo inode) {
    Preconditions.checkNotNull(name);
    Preconditions.checkNotNull(inode);
    if (directoryContent.put(name, inode) != null) {
      throw new IllegalArgumentException("File already exists: " + name);
    }
    markModificationTime();
  }

  /**
   * Does a directory lookup, and returns the inode for the specified name, or null if the child is
   * not found.
   */
  @Nullable
  InMemoryContentInfo getChild(String name) {
    return directoryContent.get(name);
  }

  /** Removes a previously existing child from the directory specified by this object. */
  void removeChild(String name) {
    if (directoryContent.remove(name) == null) {
      throw new IllegalArgumentException(name + " is not a member of this directory");
    }
    markModificationTime();
  }

  /** Returns the contents of this directory. */
  Collection<String> getAllChildren() {
    return directoryContent.keySet();
  }

  @Override
  public boolean isDirectory() {
    return true;
  }

  @Override
  public boolean isSymbolicLink() {
    return false;
  }

  @Override
  public boolean isFile() {
    return false;
  }

  @Override
  public boolean isSpecialFile() {
    return false;
  }

  /**
   * In the InMemory hierarchy, the getSize on a directory always returns the number of children in
   * the directory.
   */
  @Override
  public long getSize() {
    return directoryContent.size();
  }

  @Override
  InMemoryDirectoryInfo asDirectory() {
    return this;
  }
}
