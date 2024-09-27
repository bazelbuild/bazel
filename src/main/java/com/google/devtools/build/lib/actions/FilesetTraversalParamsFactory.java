// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversalRoot;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Set;
import javax.annotation.Nullable;

/** Factory of {@link FilesetTraversalParams}. */
public final class FilesetTraversalParamsFactory {

  /**
   * Creates parameters for a file traversal request.
   *
   * <p>Such a traversal is created for every entry in FilesetEntry.files, when it is specified.
   *
   * @param ownerLabel the rule that created this object
   * @param fileToTraverse the file to traverse; "traversal" means that if this file is actually a
   *     directory or a symlink to one then it'll be traversed as one
   * @param destPath path in the Fileset's output directory that will be the name of this file's
   *     respective symlink there, or the root of files found (in case this is a directory)
   * @param pkgBoundaryMode what to do when the traversal hits a subdirectory that is also a
   * @param strictFilesetOutput whether Fileset assumes that output Artifacts are regular files.
   */
  public static FilesetTraversalParams fileTraversal(
      Label ownerLabel,
      Artifact fileToTraverse,
      PathFragment destPath,
      PackageBoundaryMode pkgBoundaryMode,
      boolean strictFilesetOutput,
      boolean permitDirectories) {
    return DirectoryTraversalParams.getDirectoryTraversalParams(
        ownerLabel,
        DirectTraversalRoot.forFileOrDirectory(fileToTraverse),
        destPath,
        null,
        pkgBoundaryMode,
        strictFilesetOutput,
        permitDirectories,
        !fileToTraverse.isSourceArtifact());
  }

  /**
   * Creates traversal request parameters for a FilesetEntry wrapping another Fileset.
   *
   * @param ownerLabel the rule that created this object
   * @param artifact the Fileset Artifact of traversal params that were used for the nested (inner)
   *     Fileset
   * @param destPath path in the Fileset's output directory that will be the root of files coming
   *     from the nested Fileset
   * @param excludes optional; set of files directly below (not in a subdirectory of) the nested
   *     Fileset that should be excluded from the outer Fileset
   */
  public static FilesetTraversalParams nestedTraversal(
      Label ownerLabel,
      Artifact artifact,
      PathFragment destPath,
      @Nullable Set<String> excludes) {
    return NestedTraversalParams.getNestedTraversal(ownerLabel, artifact, destPath, excludes);
  }

  private static ImmutableSortedSet<String> getOrderedExcludes(@Nullable Set<String> excludes) {
    // Order the set for the sake of deterministic fingerprinting.
    return excludes == null
        ? ImmutableSortedSet.of()
        : ImmutableSortedSet.copyOf(Ordering.natural(), excludes);
  }

  @AutoValue
  abstract static class DirectoryTraversalParams implements FilesetTraversalParams {
    @Override
    public Artifact getNestedArtifact() {
      return null;
    }

    @Memoized
    @Override
    public abstract int hashCode();

    @Memoized
    byte[] getFingerprint() {
      Fingerprint fp = new Fingerprint();
      fp.addPath(getDestPath());
      if (!getExcludedFiles().isEmpty()) {
        fp.addStrings(getExcludedFiles());
      }
      fp.addBytes(getDirectTraversal().getFingerprint());
      return fp.digestAndReset();
    }

    @Override
    public void fingerprint(Fingerprint fp) {
      fp.addBytes(getFingerprint());
    }

    static DirectoryTraversalParams getDirectoryTraversalParams(
        Label ownerLabelForErrorMessages,
        DirectTraversalRoot root,
        PathFragment destPath,
        @Nullable Set<String> excludes,
        PackageBoundaryMode pkgBoundaryMode,
        boolean strictFilesetOutput,
        boolean permitDirectories,
        boolean isGenerated) {
      DirectTraversal traversal =
          DirectTraversal.getDirectTraversal(
              root, pkgBoundaryMode, strictFilesetOutput, permitDirectories, isGenerated);
      return new AutoValue_FilesetTraversalParamsFactory_DirectoryTraversalParams(
          ownerLabelForErrorMessages, destPath, getOrderedExcludes(excludes), traversal);
    }
  }

  @AutoValue
  abstract static class NestedTraversalParams implements FilesetTraversalParams {
    @Override
    public DirectTraversal getDirectTraversal() {
      return null;
    }

    @Memoized
    @Override
    public abstract int hashCode();

    @Memoized
    protected byte[] getFingerprint() {
      Fingerprint fp = new Fingerprint();
      fp.addPath(getDestPath());
      if (!getExcludedFiles().isEmpty()) {
        fp.addStrings(getExcludedFiles());
      }
      fp.addPath(getNestedArtifact().getExecPath());
      return fp.digestAndReset();
    }

    @Override
    public void fingerprint(Fingerprint fp) {
      fp.addBytes(getFingerprint());
    }

    static NestedTraversalParams getNestedTraversal(
        Label ownerLabelForErrorMessages,
        Artifact nestedArtifact,
        PathFragment destPath,
        @Nullable Set<String> excludes) {
      return new AutoValue_FilesetTraversalParamsFactory_NestedTraversalParams(
          ownerLabelForErrorMessages, destPath, getOrderedExcludes(excludes), nestedArtifact);
    }
  }
}
