// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

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
 *        file digest). See {@link RegularFileStateValue}.
 * <ul>
 *
 * <p>This class is an implementation detail of {@link FileValue} and should not be used outside of
 * {@link FileFunction}. Instead, {@link FileValue} should be used by consumers that care about
 * files.
 *
 * <p>All subclasses must implement {@link #equals} and {@link #hashCode} properly.
 */
@VisibleForTesting
public abstract class FileStateValue implements SkyValue {

  public static final DirectoryFileStateValue DIRECTORY_FILE_STATE_NODE =
      new DirectoryFileStateValue();
  public static final NonexistentFileStateValue NONEXISTENT_FILE_STATE_NODE =
      new NonexistentFileStateValue();

  /** Type of a path. */
  public enum Type {
    REGULAR_FILE,
    SPECIAL_FILE,
    DIRECTORY,
    SYMLINK,
    NONEXISTENT,
  }

  protected FileStateValue() {
  }

  public static FileStateValue create(RootedPath rootedPath,
      @Nullable TimestampGranularityMonitor tsgm) throws InconsistentFilesystemException,
      IOException {
    Path path = rootedPath.asPath();
    // Stat, but don't throw an exception for the common case of a nonexistent file. This still
    // throws an IOException in case any other IO error is encountered.
    FileStatus stat = path.statIfFound(Symlinks.NOFOLLOW);
    if (stat == null) {
      return NONEXISTENT_FILE_STATE_NODE;
    }
    return createWithStatNoFollow(rootedPath, FileStatusWithDigestAdapter.adapt(stat), tsgm);
  }

  static FileStateValue createWithStatNoFollow(RootedPath rootedPath,
      FileStatusWithDigest statNoFollow, @Nullable TimestampGranularityMonitor tsgm)
          throws InconsistentFilesystemException, IOException {
    Path path = rootedPath.asPath();
    if (statNoFollow.isFile()) {
      return statNoFollow.isSpecialFile()
          ? SpecialFileStateValue.fromStat(statNoFollow, tsgm)
          : RegularFileStateValue.fromPath(path, statNoFollow, tsgm);
    } else if (statNoFollow.isDirectory()) {
      return DIRECTORY_FILE_STATE_NODE;
    } else if (statNoFollow.isSymbolicLink()) {
      return new SymlinkFileStateValue(path.readSymbolicLinkUnchecked());
    }
    throw new InconsistentFilesystemException("according to stat, existing path " + path + " is "
        + "neither a file nor directory nor symlink.");
  }

  @VisibleForTesting
  @ThreadSafe
  public static SkyKey key(RootedPath rootedPath) {
    return new SkyKey(SkyFunctions.FILE_STATE, rootedPath);
  }

  public abstract Type getType();

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

  @Override
  public String toString() {
    return prettyPrint();
  }

  abstract String prettyPrint();

  /**
   * Implementation of {@link FileStateValue} for regular files that exist.
   *
   * <p>A union of (digest, mtime). We use digests only if a fast digest lookup is available from
   * the filesystem. If not, we fall back to mtime-based digests. This avoids the case where Blaze
   * must read all files involved in the build in order to check for modifications in the case
   * where fast digest lookups are not available.
   */
  @ThreadSafe
  public static final class RegularFileStateValue extends FileStateValue {
    private final long size;
    // Only needed for empty-file equality-checking. Otherwise is always -1.
    // TODO(bazel-team): Consider getting rid of this special case for empty files.
    private final long mtime;
    @Nullable private final byte[] digest;
    @Nullable private final FileContentsProxy contentsProxy;

    public RegularFileStateValue(long size, long mtime, byte[] digest,
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
     * Create a FileFileStateValue instance corresponding to the given existing file.
     * @param stat must be of type "File". (Not a symlink).
     */
    private static RegularFileStateValue fromPath(Path path, FileStatusWithDigest stat,
                                        @Nullable TimestampGranularityMonitor tsgm)
        throws InconsistentFilesystemException {
      Preconditions.checkState(stat.isFile(), path);

      try {
        byte[] digest = tryGetDigest(path, stat);
        if (digest == null) {
          long mtime = stat.getLastModifiedTime();
          // Note that TimestampGranularityMonitor#notifyDependenceOnFileTime is a thread-safe
          // method.
          if (tsgm != null) {
            tsgm.notifyDependenceOnFileTime(mtime);
          }
          return new RegularFileStateValue(stat.getSize(), stat.getLastModifiedTime(), null,
              FileContentsProxy.create(mtime, stat.getNodeId()));
        } else {
          // We are careful here to avoid putting the value ID into FileMetadata if we already have
          // a digest. Arbitrary filesystems may do weird things with the value ID; a digest is more
          // robust.
          return new RegularFileStateValue(stat.getSize(), stat.getLastModifiedTime(), digest, null);
        }
      } catch (IOException e) {
        String errorMessage = e.getMessage() != null
            ? "error '" + e.getMessage() + "'" : "an error";
        throw new InconsistentFilesystemException("'stat' said " + path + " is a file but then we "
            + "later encountered " + errorMessage + " which indicates that " + path + " is no "
            + "longer a file. Did you delete it during the build?");
      }
    }

    @Nullable
    private static byte[] tryGetDigest(Path path, FileStatusWithDigest stat) throws IOException {
      try {
        byte[] digest = stat.getDigest();
        return digest != null ? digest : path.getFastDigest();
      } catch (IOException ioe) {
        if (!path.isReadable()) {
          return null;
        }
        throw ioe;
      }
    }

    @Override
    public Type getType() {
      return Type.REGULAR_FILE;
    }

    @Override
    public long getSize() {
      return size;
    }

    public long getMtime() {
      return mtime;
    }

    @Override
    @Nullable
    public byte[] getDigest() {
      return digest;
    }

    public FileContentsProxy getContentsProxy() {
      return contentsProxy;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof RegularFileStateValue) {
        RegularFileStateValue other = (RegularFileStateValue) obj;
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
    public String prettyPrint() {
      String contents = digest != null
          ? String.format("digest of ", Arrays.toString(digest))
          : contentsProxy.prettyPrint();
      String extra = mtime != -1 ? String.format(" and mtime of %d", mtime) : "";
      return String.format("regular file with size of %d and %s%s", size, contents, extra);
    }
  }

  /** Implementation of {@link FileStateValue} for special files that exist. */
  public static final class SpecialFileStateValue extends FileStateValue {
    private final FileContentsProxy contentsProxy;

    public SpecialFileStateValue(FileContentsProxy contentsProxy) {
      this.contentsProxy = contentsProxy;
    }

    static SpecialFileStateValue fromStat(FileStatusWithDigest stat,
        @Nullable TimestampGranularityMonitor tsgm) throws IOException {
      long mtime = stat.getLastModifiedTime();
      // Note that TimestampGranularityMonitor#notifyDependenceOnFileTime is a thread-safe
      // method.
      if (tsgm != null) {
        tsgm.notifyDependenceOnFileTime(mtime);
      }
      return new SpecialFileStateValue(FileContentsProxy.create(mtime, stat.getNodeId()));
    }

    @Override
    public Type getType() {
      return Type.SPECIAL_FILE;
    }

    @Override
    long getSize() {
      return 0;
    }

    @Override
    @Nullable
    byte[] getDigest() {
      return null;
    }

    public FileContentsProxy getContentsProxy() {
      return contentsProxy;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj instanceof SpecialFileStateValue) {
        SpecialFileStateValue other = (SpecialFileStateValue) obj;
        return Objects.equals(contentsProxy, other.contentsProxy);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return contentsProxy.hashCode();
    }

    @Override
    public String prettyPrint() {
      return String.format("special file with %s", contentsProxy.prettyPrint());
    }
  }

  /** Implementation of {@link FileStateValue} for directories that exist. */
  public static final class DirectoryFileStateValue extends FileStateValue {

    private DirectoryFileStateValue() {
    }

    @Override
    public Type getType() {
      return Type.DIRECTORY;
    }

    @Override
    public String prettyPrint() {
      return "directory";
    }

    // This object is normally a singleton, but deserialization produces copies.
    @Override
    public boolean equals(Object obj) {
      return obj instanceof DirectoryFileStateValue;
    }

    @Override
    public int hashCode() {
      return 7654321;
    }
  }

  /** Implementation of {@link FileStateValue} for symlinks. */
  public static final class SymlinkFileStateValue extends FileStateValue {

    private final PathFragment symlinkTarget;

    public SymlinkFileStateValue(PathFragment symlinkTarget) {
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
      if (!(obj instanceof SymlinkFileStateValue)) {
        return false;
      }
      SymlinkFileStateValue other = (SymlinkFileStateValue) obj;
      return symlinkTarget.equals(other.symlinkTarget);
    }

    @Override
    public int hashCode() {
      return symlinkTarget.hashCode();
    }

    @Override
    public String prettyPrint() {
      return "symlink to " + symlinkTarget;
    }
  }

  /** Implementation of {@link FileStateValue} for nonexistent files. */
  public static final class NonexistentFileStateValue extends FileStateValue {

    private NonexistentFileStateValue() {
    }

    @Override
    public Type getType() {
      return Type.NONEXISTENT;
    }

    @Override
    public String prettyPrint() {
      return "nonexistent path";
    }

    // This object is normally a singleton, but deserialization produces copies.
    @Override
    public boolean equals(Object obj) {
      return obj instanceof NonexistentFileStateValue;
    }

    @Override
    public int hashCode() {
      return 8765432;
    }
  }
}
