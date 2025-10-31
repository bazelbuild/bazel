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

import static com.google.common.base.Preconditions.checkNotNull;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Objects;
import com.google.devtools.build.lib.actions.HasDigest;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;
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
  static final RecursiveFilesystemTraversalValue EMPTY =
      new RecursiveFilesystemTraversalValue(
          Optional.empty(), NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  /** The root of the traversal. May only be absent for the {@link #EMPTY} instance. */
  private final Optional<ResolvedFile> resolvedRoot;

  /** The transitive closure of {@link ResolvedFile}s. */
  private final NestedSet<ResolvedFile> resolvedPaths;

  private RecursiveFilesystemTraversalValue(
      Optional<ResolvedFile> resolvedRoot, NestedSet<ResolvedFile> resolvedPaths) {
    this.resolvedRoot = checkNotNull(resolvedRoot);
    this.resolvedPaths = checkNotNull(resolvedPaths);
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
    return new RecursiveFilesystemTraversalValue(
        Optional.of(singleMember), NestedSetBuilder.create(Order.STABLE_ORDER, singleMember));
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

  /** Type information about the filesystem entry residing at a path. */
  enum FileType {
    /** A regular file. */
    FILE {
      @Override
      boolean isFile() {
        return true;
      }

      @Override
      boolean exists() {
        return true;
      }

      @Override
      public String toString() {
        return "<f>";
      }
    },
    /**
     * A symlink to a regular file.
     *
     * <p>The symlink may be direct (points to a non-symlink (here a file)) or it may be transitive
     * (points to a direct or transitive symlink).
     */
    SYMLINK_TO_FILE {
      @Override
      boolean isFile() {
        return true;
      }

      @Override
      boolean isSymlink() {
        return true;
      }

      @Override
      boolean exists() {
        return true;
      }

      @Override
      public String toString() {
        return "<lf>";
      }
    },
    /** A directory. */
    DIRECTORY {
      @Override
      boolean isDirectory() {
        return true;
      }

      @Override
      boolean exists() {
        return true;
      }

      @Override
      public String toString() {
        return "<d>";
      }
    },
    /**
     * A symlink to a directory.
     *
     * <p>The symlink may be direct (points to a non-symlink (here a directory)) or it may be
     * transitive (points to a direct or transitive symlink).
     */
    SYMLINK_TO_DIRECTORY {
      @Override
      boolean isDirectory() {
        return true;
      }

      @Override
      boolean isSymlink() {
        return true;
      }

      @Override
      boolean exists() {
        return true;
      }

      @Override
      public String toString() {
        return "<ld>";
      }
    },
    /** A dangling symlink, i.e. one whose target is known not to exist. */
    DANGLING_SYMLINK {
      @Override
      boolean isFile() {
        throw new UnsupportedOperationException();
      }

      @Override
      boolean isDirectory() {
        throw new UnsupportedOperationException();
      }

      @Override
      boolean isSymlink() {
        return true;
      }

      @Override
      public String toString() {
        return "<l?>";
      }
    },
    /** A path that does not exist or should be ignored. */
    NONEXISTENT {
      @Override
      public String toString() {
        return "<?>";
      }
    };

    boolean isFile() {
      return false;
    }

    boolean isDirectory() {
      return false;
    }

    boolean isSymlink() {
      return false;
    }

    boolean exists() {
      return false;
    }

    @Override
    public abstract String toString();
  }

  private static final class Symlink {
    private final RootedPath linkName;
    private final PathFragment unresolvedLinkTarget;
    // The resolved link target is returned by ResolvedFile.getPath()

    private Symlink(RootedPath linkName, PathFragment unresolvedLinkTarget) {
      this.linkName = checkNotNull(linkName);
      this.unresolvedLinkTarget = checkNotNull(unresolvedLinkTarget);
    }

    PathFragment getNameInSymlinkTree() {
      return linkName.getRootRelativePath();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Symlink o)) {
        return false;
      }
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
    private final HasDigest metadata;

    RegularFile(RootedPath path, HasDigest metadata) {
      this.path = checkNotNull(path);
      this.metadata = checkNotNull(metadata);
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
    public HasDigest getMetadata() {
      return metadata;
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
          && this.metadata.equals(((RegularFile) obj).metadata);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, metadata);
    }

    @Override
    public String toString() {
      return String.format("RegularFile(path=%s -- %s)", path, metadata);
    }

    @Override
    public PathFragment getNameInSymlinkTree() {
      return path.getRootRelativePath();
    }
  }

  private static final class Directory implements ResolvedFile {
    private final RootedPath path;

    Directory(RootedPath path) {
      this.path = checkNotNull(path);
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
    public HasDigest getMetadata() {
      return HasDigest.EMPTY;
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
    public PathFragment getNameInSymlinkTree() {
      return path.getRootRelativePath();
    }
  }

  private static final class DanglingSymlink implements ResolvedFile {
    private final Symlink symlink;
    private final HasDigest metadata;

    DanglingSymlink(RootedPath linkNamePath, PathFragment linkTargetPath, HasDigest metadata) {
      this.symlink = new Symlink(linkNamePath, linkTargetPath);
      this.metadata = checkNotNull(metadata);
    }

    @Override
    public FileType getType() {
      return FileType.DANGLING_SYMLINK;
    }

    @Override
    @Nullable
    public RootedPath getPath() {
      return symlink.linkName;
    }

    @Override
    public HasDigest getMetadata() {
      return metadata;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof DanglingSymlink)) {
        return false;
      }
      return this.metadata.equals(((DanglingSymlink) obj).metadata)
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
    public PathFragment getNameInSymlinkTree() {
      return symlink.getNameInSymlinkTree();
    }
  }

  private static final class SymlinkToFile implements ResolvedFile {
    private final RootedPath path;
    private final HasDigest metadata;
    private final Symlink symlink;

    SymlinkToFile(
        RootedPath targetPath,
        RootedPath linkNamePath,
        PathFragment linkTargetPath,
        HasDigest metadata) {
      this.path = checkNotNull(targetPath);
      this.metadata = checkNotNull(metadata);
      this.symlink = new Symlink(linkNamePath, linkTargetPath);
    }

    @Override
    public FileType getType() {
      return FileType.SYMLINK_TO_FILE;
    }

    @Override
    public RootedPath getPath() {
      return symlink.linkName;
    }

    @Override
    public HasDigest getMetadata() {
      return metadata;
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
          && this.metadata.equals(((SymlinkToFile) obj).metadata)
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
    public PathFragment getNameInSymlinkTree() {
      return symlink.getNameInSymlinkTree();
    }
  }

  private static final class SymlinkToDirectory implements ResolvedFile {
    private final RootedPath path;
    private final HasDigest metadata;
    private final Symlink symlink;

    SymlinkToDirectory(
        RootedPath targetPath,
        RootedPath linkNamePath,
        PathFragment linkValue,
        HasDigest metadata) {
      this.path = checkNotNull(targetPath);
      this.metadata = checkNotNull(metadata);
      this.symlink = new Symlink(linkNamePath, linkValue);
    }

    @Override
    public FileType getType() {
      return FileType.SYMLINK_TO_DIRECTORY;
    }

    @Override
    public RootedPath getPath() {
      return symlink.linkName;
    }

    @Override
    public HasDigest getMetadata() {
      return metadata;
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
          && this.metadata.equals(((SymlinkToDirectory) obj).metadata)
          && this.symlink.equals(((SymlinkToDirectory) obj).symlink);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(path, metadata, symlink);
    }

    @Override
    public String toString() {
      return String.format("SymlinkToDirectory(target=%s, %s)", path, symlink);
    }

    @Override
    public PathFragment getNameInSymlinkTree() {
      return symlink.getNameInSymlinkTree();
    }
  }

  static final class ResolvedFileFactory {
    private ResolvedFileFactory() {}

    public static ResolvedFile regularFile(RootedPath path, HasDigest metadata) {
      return new RegularFile(path, metadata);
    }

    public static ResolvedFile directory(RootedPath path) {
      return new Directory(path);
    }

    public static ResolvedFile symlinkToFile(
        RootedPath targetPath,
        RootedPath linkNamePath,
        PathFragment linkTargetPath,
        HasDigest metadata) {
      return new SymlinkToFile(targetPath, linkNamePath, linkTargetPath, metadata);
    }

    public static ResolvedFile symlinkToDirectory(
        RootedPath targetPath,
        RootedPath linkNamePath,
        PathFragment linkValue,
        HasDigest metadata) {
      return new SymlinkToDirectory(targetPath, linkNamePath, linkValue, metadata);
    }

    public static ResolvedFile danglingSymlink(RootedPath linkNamePath, PathFragment linkValue) {
      byte[] digest =
          DigestHashFunction.SHA256
              .getHashFunction()
              .hashString(linkValue.getPathString(), ISO_8859_1)
              .asBytes();
      // Ensure that the digest does not collide with that of a regular file.
      digest[0] ^= 1;
      return new DanglingSymlink(linkNamePath, linkValue, new HasDigest.ByteStringDigest(digest));
    }
  }

  /**
   * Path and type information about a single file or symlink.
   *
   * <p>The object stores things such as the absolute path of the file or symlink, its exact type
   * and, if it's a symlink, the resolved and unresolved link target paths.
   */
  public interface ResolvedFile {
    /** Type of the entity under {@link #getPath()}. */
    FileType getType();

    /** Path of the file, directory or symlink. */
    RootedPath getPath();

    /**
     * Return the best effort metadata about the target. Currently this will be a FileStateValue for
     * source targets. For generated targets we try to return a FileArtifactValue when possible, or
     * else this will be a Integer hashcode of the target.
     */
    HasDigest getMetadata();

    /**
     * Returns the path of the Fileset-output symlink relative to the output directory.
     *
     * <p>The path should contain the FilesetEntry-specific destination directory (if any) and
     * should have necessary prefixes stripped (if any).
     */
    PathFragment getNameInSymlinkTree();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RecursiveFilesystemTraversalValue o)) {
      return false;
    }
    return resolvedRoot.equals(o.resolvedRoot) && resolvedPaths.equals(o.resolvedPaths);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(resolvedRoot, resolvedPaths);
  }
}
