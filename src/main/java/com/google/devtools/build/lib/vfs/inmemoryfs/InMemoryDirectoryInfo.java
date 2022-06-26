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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.OsPathPolicy;
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

  private static final boolean CASE_SENSITIVE = OsPathPolicy.getFilePathOs().isCaseSensitive();

  // Keys in this map are usually strings, except on case-insensitive operating systems (e.g.
  // Windows) where we use CaseInsensitiveFilename.
  private final Map<Object, InMemoryContentInfo> directoryContent = new HashMap<>();

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
    if (directoryContent.put(key(name), inode) != null) {
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
    return directoryContent.get(key(name));
  }

  /** Removes a previously existing child from the directory specified by this object. */
  void removeChild(String name) {
    if (directoryContent.remove(key(name)) == null) {
      throw new IllegalArgumentException(name + " is not a member of this directory");
    }
    markModificationTime();
  }

  /** Returns the contents of this directory. */
  Collection<String> getAllChildren() {
    return Collections2.transform(
        directoryContent.keySet(),
        CASE_SENSITIVE ? String.class::cast : name -> ((CaseInsensitiveFilename) name).name);
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

  @Override
  InMemoryDirectoryInfo asDirectory() {
    return this;
  }

  private static Object key(String name) {
    return CASE_SENSITIVE ? name : new CaseInsensitiveFilename(name);
  }

  @ThreadSafety.Immutable
  private static final class CaseInsensitiveFilename {
    private final String name;

    private CaseInsensitiveFilename(String name) {
      this.name = name;
    }

    @Override
    public int hashCode() {
      return OsPathPolicy.getFilePathOs().hash(name);
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof CaseInsensitiveFilename)) {
        return false;
      }
      return OsPathPolicy.getFilePathOs().equals(this.name, ((CaseInsensitiveFilename) obj).name);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("name", name).toString();
    }
  }
}
