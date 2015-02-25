// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Definition of a symlink in the output tree of a Fileset rule. */
public final class FilesetOutputSymlink {
  private static final String STRIPPED_METADATA = "<stripped-for-testing>";

  /** Final name of the symlink relative to the Fileset's output directory. */
  public final PathFragment name;

  /** Target of the symlink. Depending on FilesetEntry.symlinks it may be relative or absolute. */
  public final PathFragment target;

  /** Opaque metadata about the link and its target; should change if either of them changes. */
  public final String metadata;

  @VisibleForTesting
  public FilesetOutputSymlink(PathFragment name, PathFragment target) {
    this.name = name;
    this.target = target;
    this.metadata = STRIPPED_METADATA;
  }

  /**
   * @param name relative path under the Fileset's output directory, including FilesetEntry.destdir
   *        with and FilesetEntry.strip_prefix applied (if applicable)
   * @param target relative or absolute value of the link
   * @param metadata opaque metadata about the link and its target; should change if either the link
   *        or its target changes
   */
  public FilesetOutputSymlink(PathFragment name, PathFragment target, String metadata) {
    this.name = Preconditions.checkNotNull(name);
    this.target = Preconditions.checkNotNull(target);
    this.metadata = Preconditions.checkNotNull(metadata);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null || !obj.getClass().equals(getClass())) {
      return false;
    }
    FilesetOutputSymlink o = (FilesetOutputSymlink) obj;
    return name.equals(o.name) && target.equals(o.target) && metadata.equals(o.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(name, target, metadata);
  }

  @Override
  public String toString() {
    if (metadata.equals(STRIPPED_METADATA)) {
      return String.format("FilesetOutputSymlink(%s -> %s)",
          name.getPathString(), target.getPathString());
    } else {
      return String.format("FilesetOutputSymlink(%s -> %s | metadata=%s)",
          name.getPathString(), target.getPathString(), metadata);
    }
  }
}
