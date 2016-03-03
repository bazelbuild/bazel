// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.FilesetTraversalParams;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** Output symlinks produced by a whole FilesetEntry or by a single file in FilesetEntry.files. */
public final class FilesetEntryValue implements SkyValue {
  static final FilesetEntryValue EMPTY =
      new FilesetEntryValue(ImmutableSet.<FilesetOutputSymlink>of());

  private final ImmutableSet<FilesetOutputSymlink> symlinks;

  private FilesetEntryValue(ImmutableSet<FilesetOutputSymlink> symlinks) {
    this.symlinks = symlinks;
  }

  static FilesetEntryValue of(ImmutableSet<FilesetOutputSymlink> symlinks) {
    if (symlinks.isEmpty()) {
      return EMPTY;
    } else {
      return new FilesetEntryValue(symlinks);
    }
  }

  /** Returns the list of output symlinks. */
  public ImmutableSet<FilesetOutputSymlink> getSymlinks() {
    return symlinks;
  }

  public static SkyKey key(FilesetTraversalParams params) {
    return SkyKey.create(SkyFunctions.FILESET_ENTRY, params);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof FilesetEntryValue)) {
      return false;
    }
    return symlinks.equals(((FilesetEntryValue) obj).symlinks);
  }

  @Override
  public int hashCode() {
    return symlinks.hashCode();
  }
}
