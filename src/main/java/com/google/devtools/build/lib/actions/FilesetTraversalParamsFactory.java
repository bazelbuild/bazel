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
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversalRoot;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.FilesetEntry.SymlinkBehavior;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Set;
import javax.annotation.Nullable;

/** Factory of {@link FilesetTraversalParams}. */
public final class FilesetTraversalParamsFactory {

  /**
   * Creates parameters for a recursive traversal request in a package.
   *
   * <p>"Recursive" means that a directory is traversed along with all of its subdirectories. Such
   * a traversal is created when FilesetEntry.files is unspecified.
   *
   * @param ownerLabel the rule that created this object
   * @param buildFile path of the BUILD file of the package to traverse
   * @param destPath path in the Fileset's output directory that will be the root of files found
   *     in this directory
   * @param excludes optional; set of files directly under this package's directory to exclude;
   *     files in subdirectories cannot be excluded
   * @param symlinkBehaviorMode what to do with symlinks
   * @param pkgBoundaryMode what to do when the traversal hits a subdirectory that is also a
   *     subpackage (contains a BUILD file)
   */
  public static FilesetTraversalParams recursiveTraversalOfPackage(Label ownerLabel,
      Artifact buildFile, PathFragment destPath, @Nullable Set<String> excludes,
      SymlinkBehavior symlinkBehaviorMode, PackageBoundaryMode pkgBoundaryMode) {
    Preconditions.checkState(buildFile.isSourceArtifact(), "%s", buildFile);
    return DirectoryTraversalParams.getDirectoryTraversalParams(ownerLabel,
        DirectTraversalRoot.forPackage(buildFile), true, destPath, excludes,
        symlinkBehaviorMode, pkgBoundaryMode, true, false);
  }

  /**
   * Creates parameters for a recursive traversal request in a directory.
   *
   * <p>"Recursive" means that a directory is traversed along with all of its subdirectories. Such
   * a traversal is created when FilesetEntry.files is unspecified.
   *
   * @param ownerLabel the rule that created this object
   * @param directoryToTraverse path of the directory to traverse
   * @param destPath path in the Fileset's output directory that will be the root of files found
   *     in this directory
   * @param excludes optional; set of files directly below this directory to exclude; files in
   *     subdirectories cannot be excluded
   * @param symlinkBehaviorMode what to do with symlinks
   * @param pkgBoundaryMode what to do when the traversal hits a subdirectory that is also a
   *     subpackage (contains a BUILD file)
   */
  public static FilesetTraversalParams recursiveTraversalOfDirectory(Label ownerLabel,
      Artifact directoryToTraverse, PathFragment destPath, @Nullable Set<String> excludes,
      SymlinkBehavior symlinkBehaviorMode, PackageBoundaryMode pkgBoundaryMode) {
    return DirectoryTraversalParams.getDirectoryTraversalParams(ownerLabel,
        DirectTraversalRoot.forFileOrDirectory(directoryToTraverse), false, destPath,
        excludes, symlinkBehaviorMode, pkgBoundaryMode, true,
        !directoryToTraverse.isSourceArtifact());
  }

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
   * @param symlinkBehaviorMode what to do with symlinks
   * @param pkgBoundaryMode what to do when the traversal hits a subdirectory that is also a
   *     subpackage (contains a BUILD file)
   */
  public static FilesetTraversalParams fileTraversal(Label ownerLabel, Artifact fileToTraverse,
      PathFragment destPath, SymlinkBehavior symlinkBehaviorMode,
      PackageBoundaryMode pkgBoundaryMode) {
    return DirectoryTraversalParams.getDirectoryTraversalParams(ownerLabel,
        DirectTraversalRoot.forFileOrDirectory(fileToTraverse), false, destPath, null,
        symlinkBehaviorMode, pkgBoundaryMode, false, !fileToTraverse.isSourceArtifact());
  }

  /**
   * Creates traversal request parameters for a FilesetEntry wrapping another Fileset. If possible,
   * the original {@code nested} is returned to avoid unnecessary object creation. In that case, the
   * {@code ownerLabelForErrorMessages} may be ignored. Since the wrapping traversal could not have
   * an error on its own, any error messages printed will still be correct.
   *
   * @param ownerLabel the rule that created this object
   * @param nested the list of traversal params that were used for the nested (inner) Fileset
   * @param destDir path in the Fileset's output directory that will be the root of files coming
   *     from the nested Fileset
   * @param excludes optional; set of files directly below (not in a subdirectory of) the nested
   *     Fileset that should be excluded from the outer Fileset
   */
  public static FilesetTraversalParams nestedTraversal(
      Label ownerLabel,
      ImmutableList<FilesetTraversalParams> nested,
      PathFragment destDir,
      @Nullable Set<String> excludes) {
    if (nested.size() == 1
        && destDir.segmentCount() == 0
        && (excludes == null || excludes.isEmpty())) {
      // Wrapping the traversal here would not lead to a different result: the output location is
      // the same and there are no additional excludes.
      return Iterables.getOnlyElement(nested);
    }
    // When srcdir is another Fileset, then files must be null so strip_prefix must also be null.
    return NestedTraversalParams.getNestedTraversal(ownerLabel, nested, destDir, excludes);
  }

  private static Set<String> getOrderedExcludes(@Nullable Set<String> excludes) {
    // Order the set for the sake of deterministic fingerprinting.
    return excludes == null
        ? ImmutableSet.of()
        : ImmutableSet.copyOf(Ordering.natural().immutableSortedCopy(excludes));
  }

  @AutoValue
  abstract static class DirectoryTraversalParams implements FilesetTraversalParams {
    @Override
    public ImmutableList<FilesetTraversalParams> getNestedTraversal() {
      return ImmutableList.of();
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
      fp.addBytes(getDirectTraversal().get().getFingerprint());
      return fp.digestAndReset();
    }

    @Override
    public void fingerprint(Fingerprint fp) {
      fp.addBytes(getFingerprint());
    }

    static DirectoryTraversalParams getDirectoryTraversalParams(Label ownerLabel,
        DirectTraversalRoot root,
        boolean isPackage,
        PathFragment destPath,
        @Nullable Set<String> excludes,
        SymlinkBehavior symlinkBehaviorMode,
        PackageBoundaryMode pkgBoundaryMode,
        boolean isRecursive,
        boolean isGenerated) {
      DirectTraversal traversal = DirectTraversal.getDirectTraversal(root, isPackage,
          symlinkBehaviorMode == SymlinkBehavior.DEREFERENCE, pkgBoundaryMode, isRecursive,
          isGenerated);
      return new AutoValue_FilesetTraversalParamsFactory_DirectoryTraversalParams(
          ownerLabel, destPath, getOrderedExcludes(excludes), Optional.of(traversal));
    }
  }

  @AutoValue
  abstract static class NestedTraversalParams implements FilesetTraversalParams {
    @Override
    public Optional<DirectTraversal> getDirectTraversal() {
      return Optional.absent();
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
      getNestedTraversal().forEach(nestedTraversal -> nestedTraversal.fingerprint(fp));
      return fp.digestAndReset();
    }

    @Override
    public void fingerprint(Fingerprint fp) {
      fp.addBytes(getFingerprint());
    }

    static NestedTraversalParams getNestedTraversal(
        Label ownerLabel,
        ImmutableList<FilesetTraversalParams> nested,
        PathFragment destDir,
        @Nullable Set<String> excludes) {
      return new AutoValue_FilesetTraversalParamsFactory_NestedTraversalParams(
          ownerLabel, destDir, getOrderedExcludes(excludes), nested);
    }
  }
}
