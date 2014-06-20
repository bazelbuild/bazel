// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A node that corresponds to a file in the source tree.
 *
 * <p>Note that the existence of a file node does not imply that the file exists on the filesystem.
 * File nodes for missing files will be created on purpose in order to facilitate incremental
 * builds in the case those files have reappeared.
 *
 * <p>This node will depend on the file node corresponding to its parent directory, and on the
 * target of the symlink, if it is a symlink.
 *
 * <p>This class contains the relevant metadata for a file, although not the contents.
 */
@Immutable
@ThreadSafe
final class FileNode implements Node {

  enum Type {
    FILE,
    DIRECTORY,
    SYMLINK,
    NONEXISTENT,
  }

  private final RootedPath rootedPath;
  private final Type type;
  private final boolean isDirectory;
  private final boolean isFile;
  private final RootedPath realRootedPath;
  private final long size;
  // Only needed for empty-file equality-checking. Otherwise is always -1.
  // TODO(bazel-team): Consider getting rid of this special case for empty files.
  private final long mtime;
  @Nullable private final PathFragment symlinkTarget;
  @Nullable private final FileMetadata metadata;

  private FileNode(RootedPath rootedPath, Type type, boolean isDirectory, boolean isFile,
      RootedPath realRootedPath, @Nullable FileMetadata metadata, PathFragment symlinkTarget,
      long size, long mtime) {
    Preconditions.checkState(type == Type.SYMLINK || ((type == Type.FILE) == (metadata != null)),
        rootedPath);
    Preconditions.checkState((type == Type.SYMLINK) == (symlinkTarget != null), rootedPath);
    Preconditions.checkState(!isDirectory || (type == Type.DIRECTORY || type == Type.SYMLINK),
        rootedPath);
    Preconditions.checkState(!isFile || (type == Type.FILE || type == Type.SYMLINK), rootedPath);
    Preconditions.checkState(isFile == (size >= 0), "%s %s", rootedPath, size);

    this.rootedPath = Preconditions.checkNotNull(rootedPath);
    this.type = Preconditions.checkNotNull(type, rootedPath);
    this.isDirectory = isDirectory;
    this.isFile = isFile;
    this.realRootedPath = Preconditions.checkNotNull(realRootedPath, rootedPath);
    this.metadata = metadata;
    this.symlinkTarget = symlinkTarget;
    this.size = size;
    // mtime is forced to be -1 so that we do not accidentally depend on it for non-empty files,
    // which should only be compared using digests.
    this.mtime = isFile && size == 0 ? mtime : -1;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof FileNode)) {
      return false;
    }

    if (this == other) {
      return true;
    }

    FileNode otherNode = (FileNode) other;
    return Objects.equals(rootedPath, otherNode.rootedPath)
        && type == otherNode.type
        && isDirectory == otherNode.isDirectory
        && size == otherNode.size
        && mtime == otherNode.mtime
        && Objects.equals(realRootedPath, otherNode.realRootedPath)
        && Objects.equals(metadata, otherNode.metadata)
        && Objects.equals(symlinkTarget, otherNode.symlinkTarget);
  }

  @Override
  public int hashCode() {
    return Objects.hash(rootedPath, type, isDirectory, realRootedPath, size, metadata,
        symlinkTarget, mtime);
  }

  @Override
  public String toString() {
    return type + " "
        + (type == Type.SYMLINK ? (isFile ? "file: " : "") + (isDirectory ? "directory: " : "") :
              "")
        + rootedPath + (!rootedPath.equals(realRootedPath) ? "=>" + realRootedPath : "") + ", "
        + (symlinkTarget != null ? "target: " + symlinkTarget + ", " : "")
        + (size >= 0 ? size + " bytes, " : "")
        + (metadata != null ? metadata : "(null metadata)");
  }

  boolean exists() {
    return type != Type.NONEXISTENT;
  }

  boolean isSymlink() {
    return type == Type.SYMLINK;
  }

  /**
   * Returns true if this node corresponds to a file or symlink to an existing file. If so, its
   * parent directory is guaranteed to exist.
   */
  public boolean isFile() {
    return isFile;
  }

  /**
   * Returns true if the file is a directory or a symlink to an existing directory. If so, its
   * parent directory is guaranteed to exist.
   */
  boolean isDirectory() {
    return isDirectory;
  }

  RootedPath rootedPath() {
    return rootedPath;
  }

  /**
   * Returns the real rooted path of the file, taking ancestor symlinks into account. For example,
   * the rooted path ['root']/['a/b'] is really ['root']/['c/b'] if 'a' is a symlink to 'b'. Note
   * that ancestor symlinks outside the root boundary are not taken into consideration.
   */
  RootedPath realRootedPath() {
    return realRootedPath;
  }

  long getSize() {
    Preconditions.checkState(isFile, rootedPath);
    return size;
  }

  @Nullable
  byte[] getDigest() {
    Preconditions.checkState(isFile, rootedPath);
    return metadata.digest;
  }

  @Nullable
  PathFragment getSymlinkTarget() {
    return symlinkTarget;
  }

  /**
   * Returns a key for building a file node for the given root-relative path.
   */
  @ThreadSafe
  static NodeKey key(RootedPath rootedPath) {
    return new NodeKey(NodeTypes.FILE, rootedPath);
  }

  @ThreadSafe
  static NodeKey key(Artifact artifact) {
    Path root = artifact.getRoot().getPath();
    return key(RootedPath.toRootedPath(root, artifact.getPath()));
  }

  /**
   * Create a FileNode.
   *
   * @param stat may be given if the status is already known.
   *             If so, must be derived from statIfFound(NOFOLLOW).
   * @param symlinkTarget may be given optionally.
   */
  @ThreadSafe
  static FileNode nodeForRootedPath(RootedPath rootedPath, RootedPath realRootedPath,
      @Nullable TimestampGranularityMonitor tsgm, @Nullable FileStatusWithDigest stat,
      @Nullable PathFragment symlinkTarget) throws IOException {
    Path path = rootedPath.asPath();
    // Stat, but don't throw an exception for the common case of a nonexistent file. This still
    // throws an IOException in case any other IO error is encountered.
    if (stat == null) {
      stat = FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.NOFOLLOW));
    }
    if (stat == null) {
      // Nonexistent file.
      return new FileNode(rootedPath, Type.NONEXISTENT, false, false, realRootedPath, null, null,
          -1, -1);
    }
    if (stat.isFile()) {
      return new FileNode(rootedPath, Type.FILE, false, true, realRootedPath,
          FileMetadata.fromPath(path, stat, tsgm), null, stat.getSize(),
          stat.getLastModifiedTime());
    } else if (stat.isDirectory()) {
      return new FileNode(rootedPath, Type.DIRECTORY, true, false, realRootedPath,
          null, null, -1, -1);
    } else if (stat.isSymbolicLink()) {
      FileStatusWithDigest symlinkStat =
          FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.FOLLOW));
      boolean isDirectory = symlinkStat != null && symlinkStat.isDirectory();
      boolean isFile = symlinkStat != null && symlinkStat.isFile();
      if (symlinkTarget == null) {
        symlinkTarget = path.readSymbolicLink();
      }
      return new FileNode(rootedPath, Type.SYMLINK, isDirectory, isFile, realRootedPath,
          isFile ? FileMetadata.fromPath(path, symlinkStat, tsgm) : null, symlinkTarget,
          isFile ? symlinkStat.getSize() : -1, isFile ? symlinkStat.getLastModifiedTime() : -1);
    } else {
      throw new IllegalStateException("stat" + stat + " of unknown type");
    }
  }

  /**
   * A union of (digest, mtime). We use digests only if a fast digest lookup is available from the
   * filesystem. If not, we fall back to mtime-based digests. This avoids the case where Blaze
   * must read all files involved in the build in order to check for modifications in the case
   * where fast digest lookups are not available.
   *
   * <p>For FileNodes to be consistent with change pruning, the FileNode value must depend
   * transitively on the contents of the file it represents. For symlinks, this is accomplished
   * by retrieving a digest or last modification time (mtime) while following symlinks.
   *
   * <p>A better alternative would be to allow the structure of the symlink graph to more directly
   * propagate changes to file contents.
   */
  @ThreadSafe
  private static class FileMetadata implements Serializable {
    @Nullable private final byte[] digest;
    @Nullable private final FileContentsProxy contentsProxy;

    private FileMetadata(byte[] digest, FileContentsProxy contentsProxy) {
      Preconditions.checkState((digest == null) != (contentsProxy == null));
      this.digest = digest;
      this.contentsProxy = contentsProxy;
    }

    /**
     * Create a FileMetadata instance corresponding to the given existing file.
     * @param stat must be of type "File". (Not a symlink).
     */
    public static FileMetadata fromPath(Path path, FileStatusWithDigest stat,
                                        @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      Preconditions.checkState(stat.isFile(), path);
      byte[] digest = stat.getDigest();
      if (digest == null) {
        digest = path.getFastDigest();
      }
      if (digest == null) {
        long mtime = stat.getLastModifiedTime();
        // Note that TimestampGranularityMonitor#notifyDependenceOnFileTime is a thread-safe method.
        if (tsgm != null) {
          tsgm.notifyDependenceOnFileTime(mtime);
        }
        return new FileMetadata(null, FileContentsProxy.create(mtime, stat.getNodeId()));
      } else {
        // We are careful here to avoid putting the node ID into FileMetadata if we already have a
        // digest. Arbitrary filesystems may do weird things with the node ID; a digest is more
        // robust.
        return new FileMetadata(digest, null);
      }
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof FileMetadata) {
        FileMetadata other = (FileMetadata) obj;
        return Arrays.equals(digest, other.digest) &&
            Objects.equals(contentsProxy, other.contentsProxy);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(Arrays.hashCode(digest), contentsProxy);
    }
  }
}
