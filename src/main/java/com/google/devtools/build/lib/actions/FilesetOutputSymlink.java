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
import com.google.devtools.build.lib.util.BigIntegerFingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.math.BigInteger;

/** Definition of a symlink in the output tree of a Fileset rule. */
@AutoValue
public abstract class FilesetOutputSymlink {
  private static final Integer STRIPPED_METADATA = new Integer(-1);

  /** Final name of the symlink relative to the Fileset's output directory. */
  public abstract PathFragment getName();

  /**
   * Target of the symlink.
   *
   * <p>This path is one of the following:
   *
   * <ol>
   *   <li>Relative to the execution root, in which case {@link #isRelativeToExecRoot} will return
   *       {@code true}.
   *   <li>An absolute path to the source tree.
   *   <li>A relative path that should be considered relative to the link.
   * </ol>
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

  /** Returns {@code true} if this symlink is relative to the execution root. */
  public abstract boolean isRelativeToExecRoot();

  /**
   * Reconstitutes the original target path of this symlink.
   *
   * <p>This method essentially performs the inverse of what is done in {@link #create}. If the
   * execution root was stripped originally, it is re-prepended.
   */
  public final PathFragment reconstituteTargetPath(PathFragment execRoot) {
    return isRelativeToExecRoot() ? execRoot.getRelative(getTargetPath()) : getTargetPath();
  }

  @Override
  public final String toString() {
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

  public BigInteger getFingerprint() {
    return new BigIntegerFingerprint()
        .addPath(getName())
        .addPath(getTargetPath())
        .addBoolean(isGeneratedTarget())
        .addBoolean(isRelativeToExecRoot())
        .getFingerprint();
  }

  @VisibleForTesting
  public static FilesetOutputSymlink createForTesting(
      PathFragment name, PathFragment target, PathFragment execRoot) {
    return create(name, target, STRIPPED_METADATA, false, execRoot);
  }

  @VisibleForTesting
  public static FilesetOutputSymlink createAlreadyRelativizedForTesting(
      PathFragment name, PathFragment target, boolean isRelativeToExecRoot) {
    return createAlreadyRelativized(name, target, STRIPPED_METADATA, false, isRelativeToExecRoot);
  }

  /**
   * Creates a {@link FilesetOutputSymlink}.
   *
   * <p>To facilitate cross-device sharing, {@code target} will have the machine-local {@code
   * execRoot} stripped if necessary. If this happens, {@link #isRelativeToExecRoot} will return
   * {@code true}.
   *
   * @param name relative path under the Fileset's output directory, including FilesetEntry.destdir
   *     with and FilesetEntry.strip_prefix applied (if applicable)
   * @param target relative or absolute value of the link
   * @param metadata metadata corresponding to the target.
   * @param isGeneratedTarget true if the target is generated.
   * @param execRoot the execution root
   */
  public static FilesetOutputSymlink create(
      PathFragment name,
      PathFragment target,
      Object metadata,
      boolean isGeneratedTarget,
      PathFragment execRoot) {
    boolean isRelativeToExecRoot = false;
    // Check if the target is under the execution root. This is not always the case because the
    // target may point to a source artifact or it may point to another symlink, in which case the
    // target path is already relative.
    if (target.startsWith(execRoot)) {
      target = target.relativeTo(execRoot);
      isRelativeToExecRoot = true;
    }
    return createAlreadyRelativized(
        name, target, metadata, isGeneratedTarget, isRelativeToExecRoot);
  }

  /**
   * Same as {@link #create}, except assumes that {@code target} already had the execution root
   * stripped if necessary.
   */
  public static FilesetOutputSymlink createAlreadyRelativized(
      PathFragment name,
      PathFragment target,
      Object metadata,
      boolean isGeneratedTarget,
      boolean isRelativeToExecRoot) {
    return new AutoValue_FilesetOutputSymlink(
        name, target, metadata, isGeneratedTarget, isRelativeToExecRoot);
  }
}
