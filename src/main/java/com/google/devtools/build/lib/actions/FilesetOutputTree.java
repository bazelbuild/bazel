// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;

/** A collection of {@link FilesetOutputSymlink}s comprising the output tree of a fileset. */
public final class FilesetOutputTree {

  public static final FilesetOutputTree EMPTY = new FilesetOutputTree(ImmutableList.of());

  public static FilesetOutputTree create(ImmutableList<FilesetOutputSymlink> symlinks) {
    return symlinks.isEmpty() ? EMPTY : new FilesetOutputTree(symlinks);
  }

  private final ImmutableList<FilesetOutputSymlink> symlinks;

  private FilesetOutputTree(ImmutableList<FilesetOutputSymlink> symlinks) {
    this.symlinks = checkNotNull(symlinks);
  }

  public ImmutableList<FilesetOutputSymlink> symlinks() {
    return symlinks;
  }

  public int size() {
    return symlinks.size();
  }

  public boolean isEmpty() {
    return symlinks.isEmpty();
  }

  @Override
  public int hashCode() {
    return symlinks.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof FilesetOutputTree that)) {
      return false;
    }
    return symlinks.equals(that.symlinks);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("symlinks", symlinks).toString();
  }
}
