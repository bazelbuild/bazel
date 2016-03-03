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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Clock;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * This class represents a directory stored in an {@link InMemoryFileSystem}.
 */
@ThreadSafe
class InMemoryDirectoryInfo extends InMemoryContentInfo {

  private final ConcurrentMap<String, InMemoryContentInfo> directoryContent =
      new ConcurrentHashMap<>();

  InMemoryDirectoryInfo(Clock clock) {
    this(clock, true);
  }

  protected InMemoryDirectoryInfo(Clock clock, boolean isMutable) {
    super(clock, isMutable);
    if (isMutable) {
      setExecutable(true);
    }
  }

  /**
   * Adds a new child to this directory under the name "name". Callers must
   * ensure that no entry of that name exists already.
   */
  synchronized void addChild(String name, InMemoryContentInfo inode) {
    if (name == null) { throw new NullPointerException(); }
    if (inode == null) { throw new NullPointerException(); }
    if (directoryContent.put(name, inode) != null) {
      throw new IllegalArgumentException("File already exists: " + name);
    }
    markModificationTime();
  }

  /**
   * Does a directory lookup, and returns the "inode" for the specified name.
   * Returns null if the child is not found.
   */
  synchronized InMemoryContentInfo getChild(String name) {
    return directoryContent.get(name);
  }

  /**
   * Removes a previously existing child from the directory specified by this
   * object.
   */
  synchronized void removeChild(String name) {
    if (directoryContent.remove(name) == null) {
      throw new IllegalArgumentException(name + " is not a member of this directory");
    }
    markModificationTime();
  }

  /**
   * This function returns the content of a directory. For now, it returns a set
   * to reflect the semantics of the value returned (ie. unordered, no
   * duplicates). If thats too slow, it should be changed later.
   */
  Set<String> getAllChildren() {
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
   * In the InMemory hierarchy, the getSize on a directory always returns the
   * number of children in the directory.
   */
  @Override
  public long getSize() {
    return directoryContent.size();
  }

}
