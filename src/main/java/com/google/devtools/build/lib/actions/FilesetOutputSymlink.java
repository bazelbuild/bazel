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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Definition of a symlink in the output tree of a Fileset rule.
 *
 * @param name Final name of the symlink relative to the Fileset's output directory.
 * @param targetPath Target of the symlink.
 *     <p>This path is one of the following:
 *     <ol>
 *       <li>Relative to the execution root, in which case {@link #isRelativeToExecRoot} will return
 *           {@code true}.
 *       <li>An absolute path to the source tree.
 *     </ol>
 *
 * @param metadata {@link FileArtifactValue} representing metadata of the symlink target; guaranteed
 *     to have a non-null {@link FileArtifactValue#getDigest()}
 * @param relativeToExecRoot Returns {@code true} if this symlink is relative to the execution root.
 * @param enclosingTreeArtifactExecPath If this symlink points to a file inside a tree artifact,
 *     returns the exec path of that file's {@linkplain Artifact#getParent parent} tree artifact.
 *     Otherwise, returns {@code null} .
 *     <p>To simplify serialization, only the exec path is stored, not the whole {@link
 *     SpecialArtifact} .
 */
// TODO: b/403610723 - Relativization can be simplified now that all symlinks point to an Artifact.
@AutoCodec
public record FilesetOutputSymlink(
    PathFragment name,
    PathFragment targetPath,
    FileArtifactValue metadata,
    boolean relativeToExecRoot,
    @Nullable PathFragment enclosingTreeArtifactExecPath) {
  public FilesetOutputSymlink {
    checkNotNull(name, "name");
    checkNotNull(targetPath, "targetPath");
    checkNotNull(metadata, "metadata");
    checkNotNull(metadata.getDigest(), "digest of %s", metadata);
  }

  private static final FileArtifactValue EMPTY_METADATA_FOR_TESTING =
      FileArtifactValue.createForNormalFile(new byte[] {}, null, 0);

  /**
   * Reconstitutes the original target path of this symlink.
   *
   * <p>This method essentially performs the inverse of what is done in {@link #create}. If the
   * execution root was stripped originally, it is re-prepended.
   */
  public final PathFragment reconstituteTargetPath(PathFragment execRoot) {
    return relativeToExecRoot() ? execRoot.getRelative(targetPath()) : targetPath();
  }

  @Override
  public final String toString() {
    return String.format(
        "FilesetOutputSymlink(%s -> %s | metadata=%s)", name(), targetPath(), metadata());
  }

  @VisibleForTesting
  public static FilesetOutputSymlink createForTesting(
      PathFragment name, PathFragment target, PathFragment execRoot) {
    return create(
        name, target, EMPTY_METADATA_FOR_TESTING, execRoot, /* enclosingTreeArtifact= */ null);
  }

  @VisibleForTesting
  public static FilesetOutputSymlink createAlreadyRelativizedForTesting(
      PathFragment name, PathFragment target, boolean isRelativeToExecRoot) {
    return createAlreadyRelativized(
        name,
        target,
        EMPTY_METADATA_FOR_TESTING,
        isRelativeToExecRoot,
        /* enclosingTreeArtifactExecPath= */ null);
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
   * @param execRoot the execution root
   * @param enclosingTreeArtifact if {@code target} is a tree artifact file, its {@linkplain
   *     Artifact#getParent parent} tree artifact, otherwise {@code null}
   */
  public static FilesetOutputSymlink create(
      PathFragment name,
      PathFragment target,
      FileArtifactValue metadata,
      PathFragment execRoot,
      @Nullable SpecialArtifact enclosingTreeArtifact) {
    boolean isRelativeToExecRoot = false;
    // Check if the target is under the execution root. This is not always the case because the
    // target may point to a source artifact or it may point to another symlink, in which case the
    // target path is already relative.
    if (target.startsWith(execRoot)) {
      target = target.relativeTo(execRoot);
      isRelativeToExecRoot = true;
    }
    PathFragment enclosingTreeArtifactExecPath;
    if (enclosingTreeArtifact == null) {
      enclosingTreeArtifactExecPath = null;
    } else {
      checkArgument(enclosingTreeArtifact.isTreeArtifact(), enclosingTreeArtifact);
      enclosingTreeArtifactExecPath = enclosingTreeArtifact.getExecPath();
    }
    return createAlreadyRelativized(
        name, target, metadata, isRelativeToExecRoot, enclosingTreeArtifactExecPath);
  }

  /**
   * Same as {@link #create}, except assumes that {@code target} already had the execution root
   * stripped if necessary.
   */
  public static FilesetOutputSymlink createAlreadyRelativized(
      PathFragment name,
      PathFragment target,
      FileArtifactValue metadata,
      boolean isRelativeToExecRoot,
      @Nullable PathFragment enclosingTreeArtifactExecPath) {
    checkArgument(!target.isEmpty(), "Empty symlink target for %s", name);
    return new FilesetOutputSymlink(
        name, target, metadata, isRelativeToExecRoot, enclosingTreeArtifactExecPath);
  }
}
