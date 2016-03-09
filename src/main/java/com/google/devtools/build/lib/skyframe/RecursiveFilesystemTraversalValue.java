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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Objects;
import com.google.common.base.Optional;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalFunction.DanglingSymlinkException;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalFunction.FileType;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.regex.Pattern;

import javax.annotation.Nullable;

/**
 * Collection of files found while recursively traversing a path.
 *
 * <p>The path may refer to files, symlinks or directories that may or may not exist.
 *
 * <p>Traversing a file or a symlink results in a single {@link ResolvedFile} corresponding to the
 * file or symlink.
 *
 * <p>Traversing a directory results in a collection of {@link ResolvedFile}s for all files and
 * symlinks under it, and in all of its subdirectories. The {@link TraversalRequest} can specify
 * whether to traverse source subdirectories that are packages (have BUILD files in them).
 *
 * <p>Traversing a symlink that points to a directory is the same as traversing a normal directory.
 * The paths in the result will not be resolved; the files will be listed under the symlink, as if
 * it was the actual directory they reside in.
 *
 * <p>Editing a file that is part of this traversal, or adding or removing a file in a directory
 * that is part of this traversal, will invalidate this {@link SkyValue}. This also applies to
 * directories that are symlinked to.
 */
public final class RecursiveFilesystemTraversalValue implements SkyValue {
  static final RecursiveFilesystemTraversalValue EMPTY = new RecursiveFilesystemTraversalValue(
      Optional.<ResolvedFile>absent(),
      NestedSetBuilder.<ResolvedFile>emptySet(Order.STABLE_ORDER));

  /** The root of the traversal. May only be absent for the {@link #EMPTY} instance. */
  private final Optional<ResolvedFile> resolvedRoot;

  /** The transitive closure of {@link ResolvedFile}s. */
  private final NestedSet<ResolvedFile> resolvedPaths;

  private RecursiveFilesystemTraversalValue(Optional<ResolvedFile> resolvedRoot,
      NestedSet<ResolvedFile> resolvedPaths) {
    this.resolvedRoot = Preconditions.checkNotNull(resolvedRoot);
    this.resolvedPaths = Preconditions.checkNotNull(resolvedPaths);
  }

  static RecursiveFilesystemTraversalValue of(ResolvedFile resolvedRoot,
      NestedSet<ResolvedFile> resolvedPaths) {
    if (resolvedPaths.isEmpty()) {
      return EMPTY;
    } else {
      return new RecursiveFilesystemTraversalValue(Optional.of(resolvedRoot), resolvedPaths);
    }
  }

  static RecursiveFilesystemTraversalValue of(ResolvedFile singleMember) {
    return new RecursiveFilesystemTraversalValue(Optional.of(singleMember),
        NestedSetBuilder.<ResolvedFile>create(Order.STABLE_ORDER, singleMember));
  }

  /** Returns the root of the traversal; absent only for the {@link #EMPTY} instance. */
  public Optional<ResolvedFile> getResolvedRoot() {
    return resolvedRoot;
  }

  /**
   * Retrieves the set of {@link ResolvedFile}s that were found by this traversal.
   *
   * <p>The returned set may be empty if no files were found, or the ones found were to be
   * considered non-existent. Unless it's empty, the returned set always includes the
   * {@link #getResolvedRoot() resolved root}.
   *
   * <p>The returned set also includes symlinks. If a symlink points to a directory, its contents
   * are also included in this set, and their path will start with the symlink's path, just like on
   * a usual Unix file system.
   */
  public NestedSet<ResolvedFile> getTransitiveFiles() {
    return resolvedPaths;
  }

  public static SkyKey key(TraversalRequest traversal) {
    return SkyKey.create(SkyFunctions.RECURSIVE_FILESYSTEM_TRAVERSAL, traversal);
  }

  /** The parameters of a file or directory traversal. */
  public static final class TraversalRequest {

    /** The path to start the traversal from; may be a file, a directory or a symlink. */
    final RootedPath path;

    /**
     * Whether the path is in the output tree.
     *
     * <p>Such paths and all their subdirectories are assumed not to define packages, so package
     * lookup for them is skipped.
     */
    final boolean isGenerated;

    /** Whether traversal should descend into directories that are roots of subpackages. */
    final PackageBoundaryMode crossPkgBoundaries;

    /**
     * Whether to skip checking if the root (if it's a directory) contains a BUILD file.
     *
     * <p>Such directories are not considered to be packages when this flag is true. This needs to
     * be true in order to traverse directories of packages, but should be false for <i>their</i>
     * subdirectories.
     */
    final boolean skipTestingForSubpackage;

    /** A pattern that files must match to be included in this traversal (may be null.) */
    @Nullable
    final Pattern pattern;

    /** Information to be attached to any error messages that may be reported. */
    @Nullable final String errorInfo;

    public TraversalRequest(RootedPath path, boolean isRootGenerated,
        PackageBoundaryMode crossPkgBoundaries, boolean skipTestingForSubpackage,
        @Nullable String errorInfo, @Nullable Pattern pattern) {
      this.path = path;
      this.isGenerated = isRootGenerated;
      this.crossPkgBoundaries = crossPkgBoundaries;
      this.skipTestingForSubpackage = skipTestingForSubpackage;
      this.errorInfo = errorInfo;
      this.pattern = pattern;
    }

    private TraversalRequest duplicate(RootedPath newRoot, boolean newSkipTestingForSubpackage) {
      return new TraversalRequest(newRoot, isGenerated, crossPkgBoundaries,
          newSkipTestingForSubpackage, errorInfo, pattern);
    }

    /** Creates a new request to traverse a child element in the current directory (the root). */
    TraversalRequest forChildEntry(RootedPath newPath) {
      return duplicate(newPath, false);
    }

    /**
     * Creates a new request for a changed root.
     *
     * <p>This method can be used when a package is found out to be under a different root path than
     * originally assumed.
     */
    TraversalRequest forChangedRootPath(Path newRoot) {
      return duplicate(RootedPath.toRootedPath(newRoot, path.getRelativePath()),
          skipTestingForSubpackage);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof TraversalRequest)) {
        return false;
      }
      TraversalRequest o = (TraversalRequest) obj;
      return path.equals(o.path) && isGenerated == o.isGenerated
          && crossPkgBoundaries == o.crossPkgBoundaries
          && skipTestingForSubpackage == o.skipTestingForSubpackage
          && Objects.equal(pattern, o.pattern);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, isGenerated, crossPkgBoundaries, skipTestingForSubpackage,
          pattern);
    }

    @Override
    public String toString() {
      return String.format(
          "TraversalParams(root=%s, is_generated=%d, skip_testing_for_subpkg=%d,"
          + " pkg_boundaries=%s, pattern=%s)", path, isGenerated ? 1 : 0,
          skipTestingForSubpackage ? 1 : 0, crossPkgBoundaries,
          pattern == null ? "null" : pattern.pattern());
    }
  }

  private static final class Symlink {
    private final RootedPath linkName;
    private final PathFragment unresolvedLinkTarget;
    // The resolved link target is returned by ResolvedFile.getPath()

    private Symlink(RootedPath linkName, PathFragment unresolvedLinkTarget) {
      this.linkName = Preconditions.checkNotNull(linkName);
      this.unresolvedLinkTarget = Preconditions.checkNotNull(unresolvedLinkTarget);
    }

    PathFragment getNameInSymlinkTree() {
      return linkName.getRelativePath();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Symlink)) {
        return false;
      }
      Symlink o = (Symlink) obj;
      return linkName.equals(o.linkName) && unresolvedLinkTarget.equals(o.unresolvedLinkTarget);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(linkName, unresolvedLinkTarget);
    }

    @Override
    public String toString() {
      return String.format("Symlink(link_name=%s, unresolved_target=%s)",
          linkName, unresolvedLinkTarget);
    }
  }

  private static final class RegularFile implements ResolvedFile {
    private final RootedPath path;
    @Nullable private final FileStateValue metadata;

    /** C'tor for {@link #stripMetadataForTesting()}. */
    private RegularFile(RootedPath path) {
      this.path = Preconditions.checkNotNull(path);
      this.metadata = null;
    }

    RegularFile(RootedPath path, FileStateValue metadata) {
      this.path = Preconditions.checkNotNull(path);
      this.metadata = Preconditions.checkNotNull(metadata);
    }

    @Override
    public FileType getType() {
      return FileType.FILE;
    }

    @Override
    public RootedPath getPath() {
      return path;
    }

    @Override
    @Nullable
    public int getMetadataHash() {
      return metadata == null ? 0 : metadata.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof RegularFile)) {
        return false;
      }
      return this.path.equals(((RegularFile) obj).path)
          && Objects.equal(this.metadata, ((RegularFile) obj).metadata);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, metadata);
    }

    @Override
    public String toString() {
      return String.format("RegularFile(path=%s)", path);
    }

    @Override
    public ResolvedFile stripMetadataForTesting() {
      return new RegularFile(path);
    }

    @Override
    public PathFragment getNameInSymlinkTree() {
      return path.getRelativePath();
    }

    @Override
    public PathFragment getTargetInSymlinkTree(boolean followSymlinks) {
      return path.asPath().asFragment();
    }
  }

  private static final class Directory implements ResolvedFile {
    private final RootedPath path;

    Directory(RootedPath path) {
      this.path = Preconditions.checkNotNull(path);
    }

    @Override
    public FileType getType() {
      return FileType.DIRECTORY;
    }

    @Override
    public RootedPath getPath() {
      return path;
    }

    @Override
    public int getMetadataHash() {
      return FileStateValue.DIRECTORY_FILE_STATE_NODE.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Directory)) {
        return false;
      }
      return this.path.equals(((Directory) obj).path);
    }

    @Override
    public int hashCode() {
      return path.hashCode();
    }

    @Override
    public String toString() {
      return String.format("Directory(path=%s)", path);
    }

    @Override
    public ResolvedFile stripMetadataForTesting() {
      return this;
    }

    @Override
    public PathFragment getNameInSymlinkTree() {
      return path.getRelativePath();
    }

    @Override
    public PathFragment getTargetInSymlinkTree(boolean followSymlinks) {
      return path.asPath().asFragment();
    }
  }

  private static final class DanglingSymlink implements ResolvedFile {
    private final Symlink symlink;
    @Nullable private final FileStateValue metadata;

    private DanglingSymlink(Symlink symlink) {
      this.symlink = symlink;
      this.metadata = null;
    }

    private DanglingSymlink(RootedPath linkNamePath, PathFragment linkTargetPath) {
      this.symlink = new Symlink(linkNamePath, linkTargetPath);
      this.metadata = null;
    }

    DanglingSymlink(RootedPath linkNamePath, PathFragment linkTargetPath,
        FileStateValue metadata) {
      this.symlink = new Symlink(linkNamePath, linkTargetPath);
      this.metadata = Preconditions.checkNotNull(metadata);
    }

    @Override
    public FileType getType() {
      return FileType.DANGLING_SYMLINK;
    }

    @Override
    @Nullable
    public RootedPath getPath() {
      return null;
    }

    @Override
    @Nullable
    public int getMetadataHash() {
      return metadata == null ? 0 : metadata.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof DanglingSymlink)) {
        return false;
      }
      return Objects.equal(this.metadata, ((DanglingSymlink) obj).metadata)
          && this.symlink.equals(((DanglingSymlink) obj).symlink);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(metadata, symlink);
    }

    @Override
    public String toString() {
      return String.format("DanglingSymlink(%s)", symlink);
    }

    @Override
    public ResolvedFile stripMetadataForTesting() {
      return new DanglingSymlink(symlink);
    }

    @Override
    public PathFragment getNameInSymlinkTree() {
      return symlink.getNameInSymlinkTree();
    }

    @Override
    public PathFragment getTargetInSymlinkTree(boolean followSymlinks)
        throws DanglingSymlinkException {
      if (followSymlinks) {
        throw new DanglingSymlinkException(symlink.linkName.asPath().getPathString(),
            symlink.unresolvedLinkTarget.getPathString());
      } else {
        return symlink.unresolvedLinkTarget;
      }
    }
  }

  private static final class SymlinkToFile implements ResolvedFile {
    private final RootedPath path;
    @Nullable private final FileStateValue metadata;
    private final Symlink symlink;

    /** C'tor for {@link #stripMetadataForTesting()}. */
    private SymlinkToFile(RootedPath targetPath, Symlink symlink) {
      this.path = Preconditions.checkNotNull(targetPath);
      this.metadata = null;
      this.symlink = Preconditions.checkNotNull(symlink);
    }

    private SymlinkToFile(
        RootedPath targetPath, RootedPath linkNamePath, PathFragment linkTargetPath) {
      this.path = Preconditions.checkNotNull(targetPath);
      this.metadata = null;
      this.symlink = new Symlink(linkNamePath, linkTargetPath);
    }

    SymlinkToFile(RootedPath targetPath, RootedPath linkNamePath,
        PathFragment linkTargetPath, FileStateValue metadata) {
      this.path = Preconditions.checkNotNull(targetPath);
      this.metadata = Preconditions.checkNotNull(metadata);
      this.symlink = new Symlink(linkNamePath, linkTargetPath);
    }

    @Override
    public FileType getType() {
      return FileType.SYMLINK_TO_FILE;
    }

    @Override
    public RootedPath getPath() {
      return path;
    }

    @Override
    @Nullable
    public int getMetadataHash() {
      return metadata == null ? 0 : metadata.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof SymlinkToFile)) {
        return false;
      }
      return this.path.equals(((SymlinkToFile) obj).path)
          && Objects.equal(this.metadata, ((SymlinkToFile) obj).metadata)
          && this.symlink.equals(((SymlinkToFile) obj).symlink);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, metadata, symlink);
    }

    @Override
    public String toString() {
      return String.format("SymlinkToFile(target=%s, %s)", path, symlink);
    }

    @Override
    public ResolvedFile stripMetadataForTesting() {
      return new SymlinkToFile(path, symlink);
    }

    @Override
    public PathFragment getNameInSymlinkTree() {
      return symlink.getNameInSymlinkTree();
    }

    @Override
    public PathFragment getTargetInSymlinkTree(boolean followSymlinks) {
      return followSymlinks ? path.asPath().asFragment() : symlink.unresolvedLinkTarget;
    }
  }

  private static final class SymlinkToDirectory implements ResolvedFile {
    private final RootedPath path;
    @Nullable private final int metadataHash;
    private final Symlink symlink;

    /** C'tor for {@link #stripMetadataForTesting()}. */
    private SymlinkToDirectory(RootedPath targetPath, Symlink symlink) {
      this.path = Preconditions.checkNotNull(targetPath);
      this.metadataHash = 0;
      this.symlink = symlink;
    }

    private SymlinkToDirectory(
        RootedPath targetPath, RootedPath linkNamePath, PathFragment linkValue) {
      this.path = Preconditions.checkNotNull(targetPath);
      this.metadataHash = 0;
      this.symlink = new Symlink(linkNamePath, linkValue);
    }

    SymlinkToDirectory(RootedPath targetPath, RootedPath linkNamePath,
        PathFragment linkValue, int metadataHash) {
      this.path = Preconditions.checkNotNull(targetPath);
      this.metadataHash = metadataHash;
      this.symlink = new Symlink(linkNamePath, linkValue);
    }

    @Override
    public FileType getType() {
      return FileType.SYMLINK_TO_DIRECTORY;
    }

    @Override
    public RootedPath getPath() {
      return path;
    }

    @Override
    @Nullable
    public int getMetadataHash() {
      return metadataHash;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof SymlinkToDirectory)) {
        return false;
      }
      return this.path.equals(((SymlinkToDirectory) obj).path)
          && Objects.equal(this.metadataHash, ((SymlinkToDirectory) obj).metadataHash)
          && this.symlink.equals(((SymlinkToDirectory) obj).symlink);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, metadataHash, symlink);
    }

    @Override
    public String toString() {
      return String.format("SymlinkToDirectory(target=%s, %s)", path, symlink);
    }

    @Override
    public ResolvedFile stripMetadataForTesting() {
      return new SymlinkToDirectory(path, symlink);
    }

    @Override
    public PathFragment getNameInSymlinkTree() {
      return symlink.getNameInSymlinkTree();
    }

    @Override
    public PathFragment getTargetInSymlinkTree(boolean followSymlinks) {
      return followSymlinks ? path.asPath().asFragment() : symlink.unresolvedLinkTarget;
    }
  }

  static final class ResolvedFileFactory {
    private ResolvedFileFactory() {}

    public static ResolvedFile regularFile(RootedPath path, FileStateValue metadata) {
      return new RegularFile(path, metadata);
    }

    public static ResolvedFile directory(RootedPath path) {
      return new Directory(path);
    }

    public static ResolvedFile symlinkToFile(RootedPath targetPath, RootedPath linkNamePath,
        PathFragment linkTargetPath, FileStateValue metadata) {
      return new SymlinkToFile(targetPath, linkNamePath, linkTargetPath, metadata);
    }

    public static ResolvedFile symlinkToDirectory(RootedPath targetPath,
        RootedPath linkNamePath, PathFragment linkValue, int metadataHash) {
      return new SymlinkToDirectory(targetPath, linkNamePath, linkValue, metadataHash);
    }

    public static ResolvedFile danglingSymlink(RootedPath linkNamePath, PathFragment linkValue,
        FileStateValue metadata) {
      return new DanglingSymlink(linkNamePath, linkValue, metadata);
    }
  }

  @VisibleForTesting
  static final class ResolvedFileFactoryForTesting {
    private ResolvedFileFactoryForTesting() {}

    static ResolvedFile regularFileForTesting(RootedPath path) {
      return new RegularFile(path);
    }

    static ResolvedFile symlinkToFileForTesting(
        RootedPath targetPath, RootedPath linkNamePath, PathFragment linkTargetPath) {
      return new SymlinkToFile(targetPath, linkNamePath, linkTargetPath);
    }

    static ResolvedFile symlinkToDirectoryForTesting(
        RootedPath targetPath, RootedPath linkNamePath, PathFragment linkValue) {
      return new SymlinkToDirectory(targetPath, linkNamePath, linkValue);
    }

    public static ResolvedFile danglingSymlinkForTesting(
        RootedPath linkNamePath, PathFragment linkValue) {
      return new DanglingSymlink(linkNamePath, linkValue);
    }
  }

  /**
   * Path and type information about a single file or symlink.
   *
   * <p>The object stores things such as the absolute path of the file or symlink, its exact type
   * and, if it's a symlink, the resolved and unresolved link target paths.
   */
  public static interface ResolvedFile {
    /** Type of the entity under {@link #getPath()}. */
    FileType getType();

    /**
     * Path of the file, directory or resolved target of the symlink.
     *
     * <p>May only return null for dangling symlinks.
     */
    @Nullable
    RootedPath getPath();

    /**
     * Hash code of associated metadata.
     *
     * <p>This is usually some hash of the {@link FileStateValue} of the underlying filesystem
     * entity.
     *
     * <p>If tests stripped the metadata or the {@link ResolvedFile} was created by the
     * {@link ResolvedFileFactoryForTesting}, this method returns 0.
     */
    int getMetadataHash();

    /**
     * Returns the path of the Fileset-output symlink relative to the output directory.
     *
     * <p>The path should contain the FilesetEntry-specific destination directory (if any) and
     * should have necessary prefixes stripped (if any).
     */
    PathFragment getNameInSymlinkTree();

    /**
     * Returns the path of the symlink target.
     *
     * @throws DanglingSymlinkException if the target cannot be resolved because the symlink is
     *     dangling
     */
    PathFragment getTargetInSymlinkTree(boolean followSymlinks) throws DanglingSymlinkException;

    /**
     * Returns a copy of this object with the metadata stripped away.
     *
     * <p>This method should only be used by tests that wish to assert that this
     * {@link ResolvedFile} refers to the expected absolute path and has the expected type, without
     * asserting its actual contents (which the metadata is a function of).
     */
    @VisibleForTesting
    ResolvedFile stripMetadataForTesting();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RecursiveFilesystemTraversalValue)) {
      return false;
    }
    RecursiveFilesystemTraversalValue o = (RecursiveFilesystemTraversalValue) obj;
    return resolvedRoot.equals(o.resolvedRoot) && resolvedPaths.equals(o.resolvedPaths);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(resolvedRoot, resolvedPaths);
  }
}
