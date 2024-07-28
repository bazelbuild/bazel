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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.actions.FileValue.RegularFileValue;
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
 * <p>The common case for {@link FileValue} is {@link RegularFileValue} (i.e. the path's real path
 * is itself, and it's an existing file). As a memory optimization for this common case, we have
 * {@link FileStateValue} be a {@link RegularFileValue} so that we don't need a wrapper object for
 * the value of the corresponding {@link FileValue} node.
 *
 * <p>All subclasses must implement {@link #equals} and {@link #hashCode} properly.
 */
public abstract class FileStateValue extends RegularFileValue implements HasDigest {
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
        return new SymlinkFileStateValue(path.readSymbolicLinkUnchecked());
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
            checkNotNull(FileStatusWithDigestAdapter.maybeAdapt(stat), rootedPath),
            /* digestWillBeInjected= */ false,
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
          : createRegularFileStateValueFromPath(
              path, statNoFollow, digestWillBeInjected, xattrProvider, tsgm);
    } else if (statNoFollow.isDirectory()) {
      return DIRECTORY_FILE_STATE_NODE;
    } else if (statNoFollow.isSymbolicLink()) {
      return new SymlinkFileStateValue(path.readSymbolicLinkUnchecked());
    }
    throw new InconsistentFilesystemException("according to stat, existing path " + path + " is "
        + "neither a file nor directory nor symlink.");
  }

  /**
   * Creates a {@link FileStateValue} instance corresponding to the given existing file.
   *
   * <p>We use digests only if a fast digest lookup is available from the filesystem. If not, we
   * fall back to mtime-based digests. This avoids the case where Blaze must read all files involved
   * in the build in order to check for modifications in the case where fast digest lookups are not
   * available.
   *
   * @param stat must be of type "File". (Not a symlink).
   */
  private static FileStateValue createRegularFileStateValueFromPath(
      Path path,
      FileStatusWithDigest stat,
      boolean digestWillBeInjected,
      XattrProvider xattrProvider,
      @Nullable TimestampGranularityMonitor tsgm)
      throws InconsistentFilesystemException {
    checkState(stat.isFile(), path);

    try {
      // If the digest will be injected, we can skip calling getFastDigest, but we need to store a
      // contents proxy because if the digest is injected but is not available from the filesystem,
      // we will need the proxy to determine whether the file was modified.
      byte[] digest = digestWillBeInjected ? null : tryGetDigest(path, stat, xattrProvider);
      if (digest == null) {
        // Note that TimestampGranularityMonitor#notifyDependenceOnFileTime is a thread-safe method.
        if (tsgm != null) {
          tsgm.notifyDependenceOnFileTime(path.asFragment(), stat.getLastChangeTime());
        }
        return new RegularFileStateValueWithContentsProxy(
            stat.getSize(), FileContentsProxy.create(stat));
      } else {
        // We are careful here to avoid putting the value ID into FileMetadata if we already have a
        // digest. Arbitrary filesystems may do weird things with the value ID; a digest is more
        // robust.
        return new RegularFileStateValueWithDigest(stat.getSize(), digest);
      }
    } catch (IOException e) {
      String errorMessage = e.getMessage() != null ? "error '" + e.getMessage() + "'" : "an error";
      throw new InconsistentFilesystemException(
          "'stat' said "
              + path
              + " is a file but then we "
              + "later encountered "
              + errorMessage
              + " which indicates that "
              + path
              + " is no "
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

  @ThreadSafe
  public static RootedPath key(RootedPath rootedPath) {
    // RootedPath is already the SkyKey we want; see FileStateKey. This method and that interface
    // are provided as readability aids.
    return rootedPath;
  }

  @Override
  public FileStateValue realFileStateValue() {
    return this;
  }

  public abstract FileStateType getType();

  /** Returns the target of the symlink, or throws an exception if this is not a symlink. */
  public PathFragment getSymlinkTarget() {
    throw new IllegalStateException();
  }

  @Override
  public long getSize() {
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
   * Implementation of {@link FileStateValue} for regular files when a {@link #digest} is provided.
   */
  public static final class RegularFileStateValueWithDigest extends FileStateValue {
    private final long size;
    private final byte[] digest;

    @VisibleForTesting
    public RegularFileStateValueWithDigest(long size, byte[] digest) {
      this.size = size;
      this.digest = checkNotNull(digest);
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
    public byte[] getDigest() {
      return digest;
    }

    @Override
    @Nullable
    public FileContentsProxy getContentsProxy() {
      return null;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof RegularFileStateValueWithDigest other)) {
        return false;
      }
      return size == other.size && Arrays.equals(digest, other.digest);
    }

    @Override
    public int hashCode() {
      return Objects.hash(size, Arrays.hashCode(digest));
    }

    @Override
    public byte[] getValueFingerprint() {
      Fingerprint fp = new Fingerprint().addLong(size);
      fp.addBytes(digest);
      return fp.digestAndReset();
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("digest", digest).add("size", size).toString();
    }

    @Override
    public String prettyPrint() {
      String contents = String.format("digest of %s", Arrays.toString(digest));
      return String.format("regular file with size of %d and %s", size, contents);
    }
  }

  /**
   * Implementation of {@link FileStateValue} for regular files when {@link FileContentsProxy} is
   * provided.
   *
   * <p>{@link #contentsProxy} is used to determine whether the file was modified.
   */
  public static final class RegularFileStateValueWithContentsProxy extends FileStateValue {
    private final long size;
    private final FileContentsProxy contentsProxy;

    @VisibleForTesting
    public RegularFileStateValueWithContentsProxy(long size, FileContentsProxy contentsProxy) {
      this.size = size;
      this.contentsProxy = checkNotNull(contentsProxy);
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
      if (!(obj instanceof RegularFileStateValueWithContentsProxy other)) {
        return false;
      }
      return size == other.size && Objects.equals(contentsProxy, other.contentsProxy);
    }

    @Override
    public int hashCode() {
      return Objects.hash(size, contentsProxy);
    }

    @Override
    public byte[] getValueFingerprint() {
      Fingerprint fp = new Fingerprint().addLong(size);
      contentsProxy.addToFingerprint(fp);
      return fp.digestAndReset();
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("size", size)
          .add("contentsProxy", contentsProxy)
          .toString();
    }

    @Override
    public String prettyPrint() {
      return String.format(
          "regular file with size of %d and %s", size, contentsProxy.prettyPrint());
    }
  }

  /** Implementation of {@link FileStateValue} for special files that exist. */
  @VisibleForTesting
  public static final class SpecialFileStateValue extends FileStateValue {
    private final FileContentsProxy contentsProxy;

    @VisibleForTesting
    public SpecialFileStateValue(FileContentsProxy contentsProxy) {
      this.contentsProxy = checkNotNull(contentsProxy);
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
    public long getSize() {
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
      if (!(obj instanceof SpecialFileStateValue other)) {
        return false;
      }
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
      if (!(obj instanceof SymlinkFileStateValue other)) {
        return false;
      }
      return symlinkTarget.equals(other.symlinkTarget);
    }

    @Override
    public int hashCode() {
      return symlinkTarget.hashCode();
    }

    @Nullable
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
