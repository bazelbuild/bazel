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
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Preconditions;
import java.util.Objects;

/** Directory entry representation returned by {@link Path#readdir}. */
public final class Dirent implements Comparable<Dirent> {
  /** Type of the directory entry */
  public enum Type {
    // A regular file.
    FILE,
    // A directory.
    DIRECTORY,
    // A symlink.
    SYMLINK,
    // None of the above.
    // For example, a special file, or a path that could not be resolved while following symlinks.
    UNKNOWN;
  }

  private final String name;
  private final Type type;

  /** Creates a new dirent with the given name and type, both of which must be non-null. */
  public Dirent(String name, Type type) {
    this.name = Preconditions.checkNotNull(name);
    this.type = Preconditions.checkNotNull(type);
  }

  public String getName() {
    return name;
  }

  public Type getType() {
    return type;
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, type);
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof Dirent otherDirent)) {
      return false;
    }
    if (this == other) {
      return true;
    }
    return name.equals(otherDirent.name) && type.equals(otherDirent.type);
  }

  @Override
  public String toString() {
    return name + "[" + type.toString().toLowerCase() + "]";
  }

  @Override
  public int compareTo(Dirent other) {
    return this.getName().compareTo(other.getName());
  }
}
