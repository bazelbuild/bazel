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
import java.util.Arrays;
import java.util.Objects;

import javax.annotation.Nullable;

/**
 * Encapsulates the filesystem operations needed to get state for a path. This is at least a
 * 'lstat' to determine what type of file the path is.
 * <ul>
 *   <li> For a non-existent file, the non existence is noted.
 *   <li> For a symlink, the symlink target is noted.
 *   <li> For a directory, the existence is noted.
 *   <li> For a file, the existence is noted, along with metadata about the file (e.g.
 *        file digest). See {@link FileFileStateNode}.
 * <ul>
 *
 * <p>This class is an implementation detail of {@link FileNode}.
 */
abstract class FileStateNode implements Node {

  public static final FileStateNode DIRECTORY_FILE_STATE_NODE = DirectoryFileStateNode.INSTANCE;
  public static final FileStateNode NONEXISTENT_FILE_STATE_NODE =
      NonexistentFileStateNode.INSTANCE; 

  public enum Type {
    FILE,
    DIRECTORY,
    SYMLINK,
    NONEXISTENT,
  }

  protected FileStateNode() {
  }

  public static FileStateNode create(RootedPath rootedPath,
      @Nullable TimestampGranularityMonitor tsgm) throws InconsistentFilesystemException,
      IOException {
    return create(rootedPath, null, tsgm);
  }

  public static FileStateNode create(RootedPath rootedPath, @Nullable FileStatusWithDigest stat,
      @Nullable TimestampGranularityMonitor tsgm) throws InconsistentFilesystemException,
      IOException {
    Path path = rootedPath.asPath();
    // Stat, but don't throw an exception for the common case of a nonexistent file. This still
    // throws an IOException in case any other IO error is encountered.
    if (stat == null) {
      stat = FileStatusWithDigestAdapter.adapt(path.statIfFound(Symlinks.NOFOLLOW));
    }
    if (stat == null) {
      return NONEXISTENT_FILE_STATE_NODE;
    } else if (stat.isFile()) {
      return FileFileStateNode.fromPath(path, stat, tsgm);
    } else if (stat.isDirectory()) {
      return DIRECTORY_FILE_STATE_NODE;
    } else if (stat.isSymbolicLink()) {
      return new SymlinkFileStateNode(path.readSymbolicLink());
    }
    throw new InconsistentFilesystemException("according to stat, existing path " + path + " is "
        + "neither a file nor directory nor symlink.");
  }

  @ThreadSafe
  static NodeKey key(RootedPath rootedPath) {
    return new NodeKey(NodeTypes.FILE_STATE, rootedPath);
  }

  abstract Type getType();

  PathFragment getSymlinkTarget() {
    throw new IllegalStateException();
  }

  long getSize() {
    throw new IllegalStateException();
  }

  @Nullable
  byte[] getDigest() {
    throw new IllegalStateException();
  }

  /**
   * Implementation of {@link FileStateNode} for files that exist.
   * 
   * <p>A union of (digest, mtime). We use digests only if a fast digest lookup is available from
   * the filesystem. If not, we fall back to mtime-based digests. This avoids the case where Blaze
   * must read all files involved in the build in order to check for modifications in the case
   * where fast digest lookups are not available.
   */
  @ThreadSafe
  private static final class FileFileStateNode extends FileStateNode {
    private final long size;
    // Only needed for empty-file equality-checking. Otherwise is always -1.
    // TODO(bazel-team): Consider getting rid of this special case for empty files.
    private final long mtime;
    @Nullable private final byte[] digest;
    @Nullable private final FileContentsProxy contentsProxy;

    private FileFileStateNode(long size, long mtime, byte[] digest,
        FileContentsProxy contentsProxy) {
      Preconditions.checkState((digest == null) != (contentsProxy == null));
      this.size = size;
      // mtime is forced to be -1 so that we do not accidentally depend on it for non-empty files,
      // which should only be compared using digests.
      this.mtime = size == 0 ? mtime : -1;
      this.digest = digest;
      this.contentsProxy = contentsProxy;
    }

    /**
     * Create a FileFileStateNode instance corresponding to the given existing file.
     * @param stat must be of type "File". (Not a symlink).
     */
    public static FileFileStateNode fromPath(Path path, FileStatusWithDigest stat,
                                        @Nullable TimestampGranularityMonitor tsgm)
        throws InconsistentFilesystemException {
      Preconditions.checkState(stat.isFile(), path);
      try {
        byte[] digest = stat.getDigest();
        if (digest == null) {
          digest = path.getFastDigest();
        }
        if (digest == null) {
          long mtime = stat.getLastModifiedTime();
          // Note that TimestampGranularityMonitor#notifyDependenceOnFileTime is a thread-safe
          // method.
          if (tsgm != null) {
            tsgm.notifyDependenceOnFileTime(mtime);
          }
          return new FileFileStateNode(stat.getSize(), stat.getLastModifiedTime(), null,
              FileContentsProxy.create(mtime, stat.getNodeId()));
        } else {
          // We are careful here to avoid putting the node ID into FileMetadata if we already have
          // a digest. Arbitrary filesystems may do weird things with the node ID; a digest is more
          // robust.
          return new FileFileStateNode(stat.getSize(), stat.getLastModifiedTime(), digest, null);
        }
      } catch (IOException e) {
        String errorMessage = e.getMessage() != null
            ? "error '" + e.getMessage() + "'" : "an error";
        throw new InconsistentFilesystemException("'stat' said " + path + " is a file but then we "
            + "later encountered " + errorMessage + " which indicates that " + path + " no longer "
            + "exists. Did you delete it during the build?");
      }
    }

    @Override
    public Type getType() {
      return Type.FILE;
    }

    @Override
    public long getSize() {
      return size;
    }

    @Override
    @Nullable
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof FileFileStateNode) {
        FileFileStateNode other = (FileFileStateNode) obj;
        return size == other.size && mtime == other.mtime && Arrays.equals(digest, other.digest)
            && Objects.equals(contentsProxy, other.contentsProxy);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(size, mtime, Arrays.hashCode(digest), contentsProxy);
    }

    @Override
    public String toString() {
      return "[size: " + size + " " + (mtime != -1 ? "mtime: " + mtime : "")
          + (digest != null ? "digest: " + Arrays.toString(digest) : contentsProxy) + "]";
    }
  }

  /** Implementation of {@link FileStateNode} for directories that exist. */
  private static final class DirectoryFileStateNode extends FileStateNode {

    public static final DirectoryFileStateNode INSTANCE = new DirectoryFileStateNode();

    private DirectoryFileStateNode() {
    }

    @Override
    public Type getType() {
      return Type.DIRECTORY;
    }

    @Override
    public String toString() {
      return "directory";
    }
  }

  /** Implementation of {@link FileStateNode} for symlinks. */
  private static final class SymlinkFileStateNode extends FileStateNode {

    private final PathFragment symlinkTarget;

    private SymlinkFileStateNode(PathFragment symlinkTarget) {
      this.symlinkTarget = symlinkTarget;
    }

    @Override
    public Type getType() {
      return Type.SYMLINK;
    }

    @Override
    public PathFragment getSymlinkTarget() {
      return symlinkTarget;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof SymlinkFileStateNode)) {
        return false;
      }
      SymlinkFileStateNode other = (SymlinkFileStateNode) obj;
      return symlinkTarget.equals(other.symlinkTarget);
    }

    @Override
    public int hashCode() {
      return symlinkTarget.hashCode();
    }

    @Override
    public String toString() {
      return "symlink to " + symlinkTarget;
    }
  }

  /** Implementation of {@link FileStateNode} for nonexistent files. */
  private static final class NonexistentFileStateNode extends FileStateNode {

    public static final NonexistentFileStateNode INSTANCE = new NonexistentFileStateNode();

    private NonexistentFileStateNode() {
    }

    @Override
    public Type getType() {
      return Type.NONEXISTENT;
    }

    @Override
    public String toString() {
      return "nonexistent";
    }
  }
}
