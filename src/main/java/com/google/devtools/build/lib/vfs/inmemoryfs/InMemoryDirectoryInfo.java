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

import com.google.common.collect.Collections2;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.OS;
import java.util.Collection;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * This class represents a directory stored in an {@link InMemoryFileSystem}.
 */
@ThreadSafe
class InMemoryDirectoryInfo extends InMemoryContentInfo {
  private final ConcurrentMap<InMemoryFileName, InMemoryContentInfo> directoryContent =
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
    if (directoryContent.put(new InMemoryFileName(name), inode) != null) {
      throw new IllegalArgumentException("File already exists: " + name);
    }
    markModificationTime();
  }

  /**
   * Does a directory lookup, and returns the "inode" for the specified name.
   * Returns null if the child is not found.
   */
  synchronized InMemoryContentInfo getChild(String name) {
    return directoryContent.get(new InMemoryFileName(name));
  }

  /**
   * Removes a previously existing child from the directory specified by this
   * object.
   */
  synchronized void removeChild(String name) {
    if (directoryContent.remove(new InMemoryFileName(name)) == null) {
      throw new IllegalArgumentException(name + " is not a member of this directory");
    }
    markModificationTime();
  }

  /**
   * This function returns the content of a directory. For now, it returns a set to reflect the
   * semantics of the value returned (ie. unordered, no duplicates). If thats too slow, it should be
   * changed later.
   */
  Collection<String> getAllChildren() {
    return Collections2.transform(
        directoryContent.keySet(), inMemoryFileName -> inMemoryFileName.value);
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

  @ThreadSafety.Immutable
  private static final class InMemoryFileName {
    private final String value;

    private InMemoryFileName(String value) {
      this.value = value;
    }

    @Override
    public int hashCode() {
      return OS.getCurrent() == OS.WINDOWS ? value.toLowerCase().hashCode() : value.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof InMemoryFileName)) {
        return false;
      }
      InMemoryFileName that = (InMemoryFileName) obj;
      if (OS.getCurrent() != OS.WINDOWS) {
        return Objects.equals(this.value, that.value);
      } else {
        return Objects.equals(this.value.toLowerCase(), that.value.toLowerCase());
      }
    }
  }
}
