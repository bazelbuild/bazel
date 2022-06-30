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
package com.google.devtools.build.lib.actions;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.XattrProvider;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Encapsulates the filesystem operations needed to get state for a path. This is equivalent to an
 * 'lstat' that does not follow symlinks to determine what type of file the path is.
 *
 * <ul>
 *   <li>For a non-existent file, the non-existence is noted.
 *   <li>For a symlink, the symlink target is noted.
 *   <li>For a directory, the existence is noted.
 *   <li>For a file, the existence is noted, along with metadata about the file (e.g. file digest).
 *       See {@link RegularFileStateValue}.
 * </ul>
 *
 * <p>This class is an implementation detail of {@link FileValue} and should not be used by {@link
 * com.google.devtools.build.skyframe.SkyFunction}s other than {@link
 * com.google.devtools.build.lib.skyframe.FileFunction}. Instead, {@link FileValue} should be used
 * by {@link com.google.devtools.build.skyframe.SkyFunction} consumers that care about files.
 *
 * <p>All subclasses must implement {@link #equals} and {@link #hashCode} properly.
 */
public abstract class FileStateValue implements HasDigest, SkyValue {
  @SerializationConstant
  public static final DirectoryFileStateValue DIRECTORY_FILE_STATE_NODE =
      new DirectoryFileStateValue();

  @SerializationConstant
  public static final NonexistentFileStateValue NONEXISTENT_FILE_STATE_NODE =
      new NonexistentFileStateValue();

  private FileStateValue() {}

  public static FileStateValue create(
      RootedPath rootedPath, SyscallCache syscallCache, @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    Path path = rootedPath.asPath();
    SyscallCache.DirentTypeWithSkip typeWithSkip = syscallCache.getType(path, Symlinks.NOFOLLOW);
    FileStatus stat = null;
    Dirent.Type type = null;
    if (typeWithSkip == SyscallCache.DirentTypeWithSkip.FILESYSTEM_OP_SKIPPED) {
      stat = syscallCache.statIfFound(path, Symlinks.NOFOLLOW);
      type = SyscallCache.statusToDirentType(stat);
    } else if (typeWithSkip != null) {
      type = typeWithSkip.getType();
    }
    if (type == null) {
      return NONEXISTENT_FILE_STATE_NODE;
    }
    switch (type) {
      case DIRECTORY:
        return DIRECTORY_FILE_STATE_NODE;
      case SYMLINK:
        return new SymlinkFileStateValue(PathFragment.create(path.readSymbolicLinkUnchecked()));
      case FILE:
      case UNKNOWN:
        if (stat == null) {
          stat = syscallCache.statIfFound(path, Symlinks.NOFOLLOW);
        }
        if (stat == null) {
          throw new InconsistentFilesystemException(
              "File " + rootedPath + " found in directory, but stat failed");
        }
        return createWithStatNoFollow(
            rootedPath,
            Preconditions.checkNotNull(FileStatusWithDigestAdapter.maybeAdapt(stat), rootedPath),
            /*digestWillBeInjected=*/ false,
            syscallCache,
            tsgm);
    }
    throw new AssertionError(type);
  }

  public static FileStateValue createWithStatNoFollow(
      RootedPath rootedPath,
      FileStatusWithDigest statNoFollow,
      boolean digestWillBeInjected,
      XattrProvider xattrProvider,
      @Nullable TimestampGranularityMonitor tsgm)
      throws IOException {
    Path path = rootedPath.asPath();
    if (statNoFollow.isFile()) {
      return statNoFollow.isSpecialFile()
          ? SpecialFileStateValue.fromStat(path.asFragment(), statNoFollow, tsgm)
          : RegularFileStateValue.fromPath(
              path, statNoFollow, digestWillBeInjected, xattrProvider, tsgm);
    } else if (statNoFollow.isDirectory()) {
      return DIRECTORY_FILE_STATE_NODE;
    } else if (statNoFollow.isSymbolicLink()) {
      return new SymlinkFileStateValue(PathFragment.create(path.readSymbolicLinkUnchecked()));
    }
    throw new InconsistentFilesystemException("according to stat, existing path " + path + " is "
        + "neither a file nor directory nor symlink.");
  }

  @ThreadSafe
  public static RootedPath key(RootedPath rootedPath) {
    // RootedPath is already the SkyKey we want; see FileStateKey. This method and that interface
    // are provided as readability aids.
    //
    // We used to weakly intern all key instances but no longer do so after concluding the data
    // structure overhead of the intern was a net negative wrt retained heap. The current approach
    // of interning nothing is instead a net positive (saved ~0.1% when implemented in Feb 2022).
    //
    // A specific call to #key is going to be for one of the following important use-cases:
    //   * FileFunction computing a specific FileValue (FV) node, declaring a Skyframe dep on a
    //     specific FileStateValue (FSV) node. There are two things to consider:
    //     * A specific FSV node will have exactly one rdep so there's no business-logic reason
    //       interning of the RootedPath here would be productive. One exception to this reasoning
    //       is symlinks: if paths `a` and `b` are both symlinks to `c` and Blaze needs to consider
    //       `a` and `b`, then it'll have nodes FV(a); FV(b); FSV(a); FSV(b); FSV(c) and forward
    //       edge sets FV(a)->{FSV(a); FSV(c)}; FV(b)->{FSV(b); FSV(c)}. So our current
    //       non-interning approach will mean we have different RootedPath instances for right
    //       endpoints of edges FV(a)->FSV(c); FV(b)->FSV(c). This is an acceptable inefficiency
    //       because this situation is not common and not worth optimizing for.
    //     * The Skyframe engine implementation effectively deduplicates the set of Skyframe deps of
    //       a node declared by a skyFunction.compute(k, env), across all Skyframe restarts. So
    //       there's no concern about a specific FV node causing many equivalent RootedPath
    //       instances to be retained, due to Skyframe restarts of FileFunction.
    //   * SkyQuery's RBuildFilesVisitor, creating a root node for a graph traversal for query
    //     evaluation. The set of RootedPaths used for these roots is both low in number and also
    //     deduped, so interning RootedPath here isn't productive. Also, these objects are not
    //     retained for long anyway so it's fine to have some garbage churn.
    //   * SkyframeExecutor, creating a root node for an invalidation traversal. Same reasoning as
    //     above.
    return rootedPath;
  }

  public abstract FileStateType getType();

  /** Returns the target of the symlink, or throws an exception if this is not a symlink. */
  public PathFragment getSymlinkTarget() {
    throw new IllegalStateException();
  }

  long getSize() {
    throw new IllegalStateException();
  }

  @Nullable
  public abstract FileContentsProxy getContentsProxy();

  @Nullable
  @Override
  public byte[] getDigest() {
    throw new IllegalStateException();
  }

  public abstract byte[] getValueFingerprint();

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
   * must read all files involved in the build in order to check for modifications in the case where
   * fast digest lookups are not available.
   */
  @ThreadSafe
  public static final class RegularFileStateValue extends FileStateValue {
    private final long size;
    @Nullable private final byte[] digest;
    @Nullable private final FileContentsProxy contentsProxy;

    @VisibleForTesting
    public RegularFileStateValue(long size, byte[] digest, FileContentsProxy contentsProxy) {
      Preconditions.checkState((digest == null) != (contentsProxy == null));
      this.size = size;
      this.digest = digest;
      this.contentsProxy = contentsProxy;
    }

    /**
     * Creates a FileFileStateValue instance corresponding to the given existing file.
     *
     * @param stat must be of type "File". (Not a symlink).
     */
    private static RegularFileStateValue fromPath(
        Path path,
        FileStatusWithDigest stat,
        boolean digestWillBeInjected,
        XattrProvider xattrProvider,
        @Nullable TimestampGranularityMonitor tsgm)
        throws InconsistentFilesystemException {
      Preconditions.checkState(stat.isFile(), path);

      try {
        // If the digest will be injected, we can skip calling getFastDigest, but we need to store a
        // contents proxy because if the digest is injected but is not available from the
        // filesystem, we will need the proxy to determine whether the file was modified.
        byte[] digest = digestWillBeInjected ? null : tryGetDigest(path, stat, xattrProvider);
        if (digest == null) {
          // Note that TimestampGranularityMonitor#notifyDependenceOnFileTime is a thread-safe
          // method.
          if (tsgm != null) {
            tsgm.notifyDependenceOnFileTime(path.asFragment(), stat.getLastChangeTime());
          }
          return new RegularFileStateValue(stat.getSize(), null, FileContentsProxy.create(stat));
        } else {
          // We are careful here to avoid putting the value ID into FileMetadata if we already have
          // a digest. Arbitrary filesystems may do weird things with the value ID; a digest is more
          // robust.
          return new RegularFileStateValue(stat.getSize(), digest, null);
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
    private static byte[] tryGetDigest(
        Path path, FileStatusWithDigest stat, XattrProvider xattrProvider) throws IOException {
      try {
        byte[] digest = stat.getDigest();
        return digest != null ? digest : xattrProvider.getFastDigest(path);
      } catch (IOException ioe) {
        if (!path.isReadable()) {
          return null;
        }
        throw ioe;
      }
    }

    @Override
    public FileStateType getType() {
      return FileStateType.REGULAR_FILE;
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
    public FileContentsProxy getContentsProxy() {
      return contentsProxy;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof RegularFileStateValue)) {
        return false;
      }
      RegularFileStateValue other = (RegularFileStateValue) obj;
      return size == other.size
          && Arrays.equals(digest, other.digest)
          && Objects.equals(contentsProxy, other.contentsProxy);
    }

    @Override
    public int hashCode() {
      return Objects.hash(size, Arrays.hashCode(digest), contentsProxy);
    }

    @Override
    public byte[] getValueFingerprint() {
      Fingerprint fp = new Fingerprint().addLong(size);
      if (digest != null) {
        fp.addBytes(digest);
      }
      if (contentsProxy != null) {
        contentsProxy.addToFingerprint(fp);
      }
      return fp.digestAndReset();
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("digest", digest)
          .add("size", size)
          .add("contentsProxy", contentsProxy).toString();
    }

    @Override
    public String prettyPrint() {
      String contents = digest != null
          ? String.format("digest of %s", Arrays.toString(digest))
          : contentsProxy.prettyPrint();
      return String.format("regular file with size of %d and %s", size, contents);
    }
  }

  /** Implementation of {@link FileStateValue} for special files that exist. */
  @VisibleForTesting
  public static final class SpecialFileStateValue extends FileStateValue {
    private final FileContentsProxy contentsProxy;

    @VisibleForTesting
    public SpecialFileStateValue(FileContentsProxy contentsProxy) {
      this.contentsProxy = Preconditions.checkNotNull(contentsProxy);
    }

    private static SpecialFileStateValue fromStat(
        PathFragment path, FileStatus stat, @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      // Note that TimestampGranularityMonitor#notifyDependenceOnFileTime is a thread-safe method.
      if (tsgm != null) {
        tsgm.notifyDependenceOnFileTime(path, stat.getLastChangeTime());
      }
      return new SpecialFileStateValue(FileContentsProxy.create(stat));
    }

    @Override
    public FileStateType getType() {
      return FileStateType.SPECIAL_FILE;
    }

    @Override
    long getSize() {
      return 0;
    }

    @Override
    @Nullable
    public byte[] getDigest() {
      return null;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      return contentsProxy;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof SpecialFileStateValue)) {
        return false;
      }
      SpecialFileStateValue other = (SpecialFileStateValue) obj;
      return contentsProxy.equals(other.contentsProxy);
    }

    @Override
    public int hashCode() {
      return contentsProxy.hashCode();
    }

    @Override
    public byte[] getValueFingerprint() {
      Fingerprint fp = new Fingerprint();
      contentsProxy.addToFingerprint(fp);
      return fp.digestAndReset();
    }

    @Override
    public String prettyPrint() {
      return String.format("special file with %s", contentsProxy.prettyPrint());
    }
  }

  /** Implementation of {@link FileStateValue} for directories that exist. */
  public static final class DirectoryFileStateValue extends FileStateValue {
    private static final byte[] FINGERPRINT = "DirectoryFileStateValue".getBytes(UTF_8);

    private DirectoryFileStateValue() {}

    @Override
    public FileStateType getType() {
      return FileStateType.DIRECTORY;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
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

    @Override
    public byte[] getValueFingerprint() {
      return FINGERPRINT;
    }
  }

  /** Implementation of {@link FileStateValue} for symlinks. */
  @VisibleForTesting
  public static final class SymlinkFileStateValue extends FileStateValue {

    private final PathFragment symlinkTarget;

    @VisibleForTesting
    public SymlinkFileStateValue(PathFragment symlinkTarget) {
      this.symlinkTarget = symlinkTarget;
    }

    @Override
    public FileStateType getType() {
      return FileStateType.SYMLINK;
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
    public FileContentsProxy getContentsProxy() {
      return null;
    }

    @Override
    public byte[] getValueFingerprint() {
      return new Fingerprint().addPath(symlinkTarget).digestAndReset();
    }

    @Override
    public String prettyPrint() {
      return "symlink to " + symlinkTarget;
    }
  }

  /** Implementation of {@link FileStateValue} for nonexistent files. */
  private static final class NonexistentFileStateValue extends FileStateValue {
    private static final byte[] FINGERPRINT = "NonexistentFileStateValue".getBytes(UTF_8);

    private NonexistentFileStateValue() {}

    @Override
    public FileStateType getType() {
      return FileStateType.NONEXISTENT;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String prettyPrint() {
      return "nonexistent path";
    }

    // This object is normally a singleton, but deserialization produces copies.
    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      return obj instanceof NonexistentFileStateValue;
    }

    @Override
    public int hashCode() {
      return 8765432;
    }

    @Override
    public byte[] getValueFingerprint() {
      return FINGERPRINT;
    }
  }
}
