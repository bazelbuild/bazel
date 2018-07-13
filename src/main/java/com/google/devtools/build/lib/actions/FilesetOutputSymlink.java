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
package com.google.devtools.build.lib.actions;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Definition of a symlink in the output tree of a Fileset rule. */
@AutoValue
public abstract class FilesetOutputSymlink {
  private static final Integer STRIPPED_METADATA = new Integer(-1);

  /** Final name of the symlink relative to the Fileset's output directory. */
  public abstract PathFragment getName();

  /**
   * Target of the symlink. This may be relative to the target's location if the target itself is a
   * relative symlink. We can override it by using FilesetEntry.symlinks = 'dereference'.
   */
  public abstract PathFragment getTargetPath();

  /**
   * Return the best effort metadata about the target. Currently this will be a FileStateValue for
   * source targets. For generated targets we try to return a FileArtifactValue when possible, or
   * else this will be a Integer hashcode of the target.
   */
  public abstract Object getMetadata();

  /** true if the target is a generated artifact */
  public abstract boolean isGeneratedTarget();

  @Override
  public String toString() {
    if (getMetadata() == STRIPPED_METADATA) {
      return String.format(
          "FilesetOutputSymlink(%s -> %s)",
          getName().getPathString(), getTargetPath().getPathString());
    } else {
      return String.format(
          "FilesetOutputSymlink(%s -> %s | metadataHash=%s)",
          getName().getPathString(), getTargetPath().getPathString(), getMetadata());
    }
  }

  @VisibleForTesting
  public static FilesetOutputSymlink createForTesting(PathFragment name, PathFragment target) {
    return new AutoValue_FilesetOutputSymlink(name, target, STRIPPED_METADATA, false);
  }

  /**
   * @param name relative path under the Fileset's output directory, including FilesetEntry.destdir
   *     with and FilesetEntry.strip_prefix applied (if applicable)
   * @param target relative or absolute value of the link
   * @param metadata metadata corresponding to the target.
   * @param isGeneratedTarget true if the target is generated.
   */
  public static FilesetOutputSymlink create(
      PathFragment name, PathFragment target, Object metadata, boolean isGeneratedTarget) {
    return new AutoValue_FilesetOutputSymlink(name, target, metadata, isGeneratedTarget);
  }
}
