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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversalRoot;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.LinkSupplier;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.FilesetEntry.SymlinkBehavior;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
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
   * @param strictFilesetOutput whether Fileset assumes that output Artifacts are regular files.
   */
  public static FilesetTraversalParams recursiveTraversalOfPackage(Label ownerLabel,
      Artifact buildFile, PathFragment destPath, @Nullable Set<String> excludes,
      SymlinkBehavior symlinkBehaviorMode, PackageBoundaryMode pkgBoundaryMode,
      boolean strictFilesetOutput) {
    Preconditions.checkState(buildFile.isSourceArtifact(), "%s", buildFile);
    return DirectoryTraversalParams.getDirectoryTraversalParams(ownerLabel,
        DirectTraversalRoot.forPackage(buildFile), true, destPath, excludes,
        symlinkBehaviorMode, pkgBoundaryMode, strictFilesetOutput, true, false);
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
   * @param strictFilesetOutput whether Fileset assumes that output Artifacts are regular files.
   */
  public static FilesetTraversalParams recursiveTraversalOfDirectory(Label ownerLabel,
      Artifact directoryToTraverse, PathFragment destPath, @Nullable Set<String> excludes,
      SymlinkBehavior symlinkBehaviorMode, PackageBoundaryMode pkgBoundaryMode,
      boolean strictFilesetOutput) {
    return DirectoryTraversalParams.getDirectoryTraversalParams(ownerLabel,
        DirectTraversalRoot.forFileOrDirectory(directoryToTraverse), false, destPath,
        excludes, symlinkBehaviorMode, pkgBoundaryMode, strictFilesetOutput, true,
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
   * @param strictFilesetOutput whether Fileset assumes that output Artifacts are regular files.
   */
  public static FilesetTraversalParams fileTraversal(Label ownerLabel, Artifact fileToTraverse,
      PathFragment destPath, SymlinkBehavior symlinkBehaviorMode,
      PackageBoundaryMode pkgBoundaryMode, boolean strictFilesetOutput) {
    return DirectoryTraversalParams.getDirectoryTraversalParams(ownerLabel,
        DirectTraversalRoot.forFileOrDirectory(fileToTraverse), false, destPath, null,
        symlinkBehaviorMode, pkgBoundaryMode, strictFilesetOutput, false,
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

  /**
   * Creates traversal request parameters for a FilesetEntry returning a customized list of links.
   *
   * @param ownerLabel the rule that created this object
   * @param linkSupplier the {@link LinkSupplier} returning a custom list of links.
   * @param destPath path in the Fileset's output directory that will be the root of files coming
   *     from the nested Fileset
   * @param excludes optional; set of files directly below (not in a subdirectory of) the nested
   *     Fileset that should be excluded from the outer Fileset
   */
  public static FilesetTraversalParams knownTraversal(
      Label ownerLabel,
      LinkSupplier linkSupplier,
      PathFragment destPath,
      @Nullable Set<String> excludes) {
    return KnownLinksTraversalParams.getKnownLinksTraversal(ownerLabel, linkSupplier, destPath,
        excludes);
  }



  private static ImmutableSortedSet<String> getOrderedExcludes(@Nullable Set<String> excludes) {
    // Order the set for the sake of deterministic fingerprinting.
    return excludes == null
        ? ImmutableSortedSet.of()
        : ImmutableSortedSet.copyOf(Ordering.natural(), excludes);
  }

  @AutoCodec
  @AutoValue
  abstract static class DirectoryTraversalParams implements FilesetTraversalParams {
    @Override
    public Artifact getNestedArtifact() {
      return null;
    }

    @Override
    public LinkSupplier additionalLinks() {
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
        boolean strictFilesetOutput,
        boolean isRecursive,
        boolean isGenerated) {
      DirectTraversal traversal = DirectTraversal.getDirectTraversal(root, isPackage,
          symlinkBehaviorMode == SymlinkBehavior.DEREFERENCE, pkgBoundaryMode, strictFilesetOutput,
          isRecursive, isGenerated);
      return create(ownerLabel, destPath, getOrderedExcludes(excludes), Optional.of(traversal));
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static DirectoryTraversalParams create(
        Label ownerLabelForErrorMessages,
        PathFragment destPath,
        ImmutableSortedSet<String> excludedFiles,
        Optional<DirectTraversal> directTraversal) {
      return new AutoValue_FilesetTraversalParamsFactory_DirectoryTraversalParams(
          ownerLabelForErrorMessages, destPath, excludedFiles, directTraversal);
    }
  }

  @AutoCodec
  @AutoValue
  abstract static class NestedTraversalParams implements FilesetTraversalParams {
    @Override
    public Optional<DirectTraversal> getDirectTraversal() {
      return Optional.absent();
    }

    @Override
    public LinkSupplier additionalLinks() {
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
        Label ownerLabel,
        Artifact nestedArtifact,
        PathFragment destPath,
        @Nullable Set<String> excludes) {
      return create(ownerLabel, destPath, getOrderedExcludes(excludes), nestedArtifact);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static NestedTraversalParams create(
        Label ownerLabelForErrorMessages,
        PathFragment destPath,
        ImmutableSortedSet<String> excludedFiles,
        Artifact nestedArtifact) {
      return new AutoValue_FilesetTraversalParamsFactory_NestedTraversalParams(
          ownerLabelForErrorMessages, destPath, excludedFiles, nestedArtifact);
    }
  }

  @AutoCodec
  @AutoValue
  abstract static class KnownLinksTraversalParams implements FilesetTraversalParams {
    @Override
    public Optional<DirectTraversal> getDirectTraversal() {
      return Optional.absent();
    }

    @Override
    public Artifact getNestedArtifact() {
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
      additionalLinks().fingerprint(fp);
      return fp.digestAndReset();
    }

    @Override
    public void fingerprint(Fingerprint fp) {
      fp.addBytes(getFingerprint());
    }

    static KnownLinksTraversalParams getKnownLinksTraversal(
        Label ownerLabel,
        LinkSupplier additionalLinks,
        PathFragment destPath,
        @Nullable Set<String> excludes) {
      return create(ownerLabel, destPath, getOrderedExcludes(excludes), additionalLinks);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static KnownLinksTraversalParams create(
        Label ownerLabelForErrorMessages,
        PathFragment destPath,
        ImmutableSortedSet<String> excludedFiles,
        LinkSupplier additionalLinks) {
      return new AutoValue_FilesetTraversalParamsFactory_KnownLinksTraversalParams(
          ownerLabelForErrorMessages, destPath, excludedFiles, additionalLinks);
    }
  }
}
