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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.BigIntegerFingerprint;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DefaultHashFunctionNotSetException;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A value that represents a file for the purposes of up-to-dateness checks of actions.
 *
 * <p>It always stands for an actual file. In particular, tree artifacts and middlemen do not have a
 * corresponding {@link FileArtifactValue}. However, the file is not necessarily present in the file
 * system; this happens when intermediate build outputs are not downloaded (and maybe when an input
 * artifact of an action is missing?)
 *
 * <p>It makes its main appearance in {@code ActionExecutionValue.artifactData}. It has two main
 * uses:
 *
 * <ul>
 *   <li>This is how dependent actions get hold of the output metadata of their generated inputs.
 *   <li>This is how {@code FileSystemValueChecker} figures out which actions need to be invalidated
 *       (just propagating the invalidation up from leaf nodes is not enough, because the output
 *       tree may have been changed while Blaze was not looking)
 * </ul>
 */
@Immutable
@ThreadSafe
public abstract class FileArtifactValue implements SkyValue, HasDigest {
  /**
   * The type of the underlying file system object. If it is a regular file, then it is guaranteed
   * to have a digest. Otherwise it does not have a digest.
   */
  public abstract FileStateType getType();

  /**
   * Returns a digest of the content of the underlying file system object; must always return a
   * non-null value for instances of type {@link FileStateType#REGULAR_FILE} that are owned by an
   * {@code ActionExecutionValue}.
   *
   * <p>All instances of this interface must either have a digest or return a last-modified time.
   * Clients should prefer using the digest for content identification (e.g., for caching), and only
   * fall back to the last-modified time if no digest is available.
   *
   * <p>The return value is owned by this object and must not be modified.
   */
  @Override
  public abstract byte[] getDigest();

  /** Returns the file's size, or 0 if the underlying file system object is not a file. */
  // TODO(ulfjack): Throw an exception if it's not a file.
  public abstract long getSize();

  /**
   * Returns the last modified time; see the documentation of {@link #getDigest} for when this can
   * and should be called.
   */
  public abstract long getModifiedTime();

  // TODO(lberki): This is only used by FileArtifactValue itself. It seems possible to remove this.
  public abstract FileContentsProxy getContentsProxy();

  @Nullable
  public BigInteger getValueFingerprint() {
    byte[] digest = getDigest();
    if (digest != null) {
      return new BigIntegerFingerprint().addDigestedBytes(digest).getFingerprint();
    }
    // TODO(janakr): return fingerprint in other cases: symlink, directory.
    return null;
  }

  /**
   * Index used to resolve remote files.
   *
   * <p>0 indicates that no such information is available which can mean that it's either a local
   * file, empty, or an omitted output.
   */
  public int getLocationIndex() {
    return 0;
  }

  /** Returns {@code true} if this is a special marker as opposed to a representing a real file. */
  public boolean isMarkerValue() {
    return this instanceof Singleton;
  }

  /** Returns {@code true} if the file only exists remotely. */
  public boolean isRemote() {
    return false;
  }

  /**
   * Provides a best-effort determination whether the file was changed since the digest was
   * computed. This method performs file system I/O, so may be expensive. It's primarily intended to
   * avoid storing bad cache entries in an action cache. It should return true if there is a chance
   * that the file was modified since the digest was computed. Better not upload if we are not sure
   * that the cache entry is reliable.
   */
  // TODO(lberki): This is very similar to couldBeModifiedSince(). Check if we can unify these.
  public abstract boolean wasModifiedSinceDigest(Path path) throws IOException;

  /**
   * Returns whether the two {@link FileArtifactValue} instances could be considered the same for
   * purposes of action invalidation.
   */
  // TODO(lberki): This is very similar to wasModifiedSinceDigest(). Check if we can unify these.
  public boolean couldBeModifiedSince(FileArtifactValue lastKnown) {
    if (this instanceof Singleton || lastKnown instanceof Singleton) {
      return true;
    }

    if (getType() != lastKnown.getType()) {
      return true;
    }

    if (getDigest() != null && lastKnown.getDigest() != null) {
      // If we know the digests, we can tell with certainty whether the file has changed.
      return !Arrays.equals(getDigest(), lastKnown.getDigest()) || getSize() != lastKnown.getSize();
    } else {
      // If not, we assume by default that the file has changed, but individual implementations
      // might know better. For example, regular local files can be compared by ctime or mtime.
      return couldBeModifiedByMetadata(lastKnown);
    }
  }

  protected boolean couldBeModifiedByMetadata(FileArtifactValue lastKnown) {
    return true;
  }

  /**
   * Marker interface for singleton implementations of this class.
   *
   * <p>Needed for a correct implementation of {@code equals}.
   */
  interface Singleton {}

  @AutoCodec public static final FileArtifactValue DEFAULT_MIDDLEMAN = new SingletonMarkerValue();
  /** Data that marks that a file is not present on the filesystem. */
  @AutoCodec public static final FileArtifactValue MISSING_FILE_MARKER = new SingletonMarkerValue();
  /**
   * Represents an omitted file -- we are aware of it but it doesn't exist. All access methods are
   * unsupported.
   */
  @AutoCodec public static final FileArtifactValue OMITTED_FILE_MARKER = new OmittedFileValue();

  public static FileArtifactValue createForSourceArtifact(Artifact artifact, FileValue fileValue)
      throws IOException {
    Preconditions.checkState(artifact.isSourceArtifact());
    Preconditions.checkState(!artifact.isConstantMetadata());
    boolean isFile = fileValue.isFile();
    return create(
        artifact.getPath(),
        isFile,
        isFile ? fileValue.getSize() : 0,
        isFile ? fileValue.realFileStateValue().getContentsProxy() : null,
        isFile ? fileValue.getDigest() : null,
        /* isShareable=*/ true);
  }

  public static FileArtifactValue createFromInjectedDigest(
      FileArtifactValue metadata, @Nullable byte[] digest, boolean isShareable) {
    return createForNormalFile(
        digest, metadata.getContentsProxy(), metadata.getSize(), isShareable);
  }

  @VisibleForTesting
  public static FileArtifactValue createForTesting(Artifact artifact) throws IOException {
    Path path = artifact.getPath();
    boolean isShareable = !artifact.isConstantMetadata();
    // Caution: there's a race condition between stating the file and computing the
    // digest. We need to stat first, since we're using the stat to detect changes.
    // We follow symlinks here to be consistent with getDigest.
    return createFromStat(path, path.stat(Symlinks.FOLLOW), isShareable);
  }

  @VisibleForTesting
  public static FileArtifactValue createForTesting(Path path) throws IOException {
    /*isShareable=*/
    // Caution: there's a race condition between stating the file and computing the
    // digest. We need to stat first, since we're using the stat to detect changes.
    // We follow symlinks here to be consistent with getDigest.
    return createFromStat(path, path.stat(Symlinks.FOLLOW), true);
  }

  public static FileArtifactValue createFromStat(Path path, FileStatus stat, boolean isShareable)
      throws IOException {
    return create(
        path, stat.isFile(), stat.getSize(), FileContentsProxy.create(stat), null, isShareable);
  }

  private static FileArtifactValue create(
      Path path,
      boolean isFile,
      long size,
      FileContentsProxy proxy,
      @Nullable byte[] digest,
      boolean isShareable)
      throws IOException {
    if (!isFile) {
      // In this case, we need to store the mtime because the action cache uses mtime for
      // directories to determine if this artifact has changed. We want this code path to go away
      // somehow.
      return new DirectoryArtifactValue(path.getLastModifiedTime());
    }
    if (digest == null) {
      digest = DigestUtils.getDigestOrFail(path, size);
    }
    Preconditions.checkState(digest != null, path);
    return createForNormalFile(digest, proxy, size, isShareable);
  }

  public static FileArtifactValue createForVirtualActionInput(byte[] digest, long size) {
    return new RegularFileArtifactValue(digest, /*proxy=*/ null, size);
  }

  public static FileArtifactValue createForUnresolvedSymlink(PathFragment symlinkTarget) {
    DigestHashFunction digestHashFunction;

    try {
      digestHashFunction = DigestHashFunction.getDefault();
    } catch (DefaultHashFunctionNotSetException e) {
      throw new IllegalStateException(e);
    }

    byte[] digest =
        digestHashFunction
            .getHashFunction()
            .hashString(symlinkTarget.getPathString(), StandardCharsets.ISO_8859_1)
            .asBytes();

    // We need to be able to tell the difference between a symlink and a file containing the same
    // text. So we transform the digest a bit. This works because if one wants to craft a file with
    // the same digest as a symlink, one would need to mount a preimage attack on the digest
    // function (this would be different if we tweaked the data before applying the hash function)
    digest[0] = (byte) (digest[0] ^ 0xff);

    return new UnresolvedSymlinkArtifactValue(digest);
  }

  @VisibleForTesting
  public static FileArtifactValue createForNormalFile(
      byte[] digest, @Nullable FileContentsProxy proxy, long size, boolean isShareable) {
    return isShareable
        ? new RegularFileArtifactValue(digest, proxy, size)
        : new UnshareableRegularFileArtifactValue(digest, proxy, size);
  }

  /**
   * Create a FileArtifactValue using the {@link Path} and size. FileArtifactValue#create will
   * handle getting the digest using the Path and size values.
   */
  public static FileArtifactValue createForNormalFileUsingPath(Path path, long size)
      throws IOException {
    return create(path, true, size, null, null, true);
  }

  public static FileArtifactValue createForDirectoryWithHash(byte[] digest) {
    return new HashedDirectoryArtifactValue(digest);
  }

  public static FileArtifactValue createForDirectoryWithMtime(long mtime) {
    return new DirectoryArtifactValue(mtime);
  }

  /**
   * Creates a FileArtifactValue used as a 'proxy' input for other ArtifactValues. These are used in
   * {@link ActionCacheChecker}.
   */
  public static FileArtifactValue createProxy(byte[] digest) {
    Preconditions.checkNotNull(digest);
    return createForNormalFile(digest, /*proxy=*/ null, /*size=*/ 0, /*isShareable=*/ true);
  }

  private static String bytesToString(byte[] bytes) {
    return "0x" + BaseEncoding.base16().omitPadding().encode(bytes);
  }

  private static final class DirectoryArtifactValue extends FileArtifactValue {
    private final long mtime;

    private DirectoryArtifactValue(long mtime) {
      this.mtime = mtime;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof DirectoryArtifactValue)) {
        return false;
      }

      DirectoryArtifactValue that = (DirectoryArtifactValue) o;
      return mtime == that.mtime;
    }

    @Override
    public int hashCode() {
      return Long.hashCode(mtime);
    }

    @Override
    public FileStateType getType() {
      return FileStateType.DIRECTORY;
    }

    @Nullable
    @Override
    public byte[] getDigest() {
      return null;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public BigInteger getValueFingerprint() {
      BigIntegerFingerprint fp = new BigIntegerFingerprint();
      fp.addString(getClass().getCanonicalName());
      fp.addLong(mtime);
      return fp.getFingerprint();
    }

    @Override
    public long getModifiedTime() {
      return mtime;
    }

    @Override
    public long getSize() {
      return 0;
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) throws IOException {
      return false;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("mtime", mtime).toString();
    }
  }

  private static final class HashedDirectoryArtifactValue extends FileArtifactValue {

    private final byte[] digest;

    private HashedDirectoryArtifactValue(byte[] digest) {
      this.digest = digest;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof HashedDirectoryArtifactValue)) {
        return false;
      }

      HashedDirectoryArtifactValue that = (HashedDirectoryArtifactValue) o;
      return Arrays.equals(digest, that.digest);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(digest);
    }

    @Override
    public FileStateType getType() {
      return FileStateType.DIRECTORY;
    }

    @Nullable
    @Override
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getModifiedTime() {
      return 0;
    }

    @Override
    public long getSize() {
      return 0;
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      // TODO(ulfjack): Ideally, we'd attempt to detect intra-build modifications here. I'm
      // consciously deferring work here as this code will most likely change again, and we're
      // already doing better than before by detecting inter-build modifications.
      return false;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("digest", digest).toString();
    }
  }

  private static class RegularFileArtifactValue extends FileArtifactValue {

    private final byte[] digest;
    @Nullable private final FileContentsProxy proxy;
    private final long size;

    private RegularFileArtifactValue(
        @Nullable byte[] digest, @Nullable FileContentsProxy proxy, long size) {
      this.digest = digest;
      this.proxy = proxy;
      this.size = size;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof RegularFileArtifactValue)) {
        return false;
      }

      RegularFileArtifactValue that = (RegularFileArtifactValue) o;
      return Arrays.equals(digest, that.digest)
          && Objects.equals(proxy, that.proxy)
          && size == that.size
          && dataIsShareable() == that.dataIsShareable();
    }

    @Override
    public int hashCode() {
      return Objects.hash(Arrays.hashCode(digest), proxy, size, dataIsShareable());
    }

    @Override
    public FileStateType getType() {
      return FileStateType.REGULAR_FILE;
    }

    @Override
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      return proxy;
    }

    @Override
    public long getSize() {
      return size;
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) throws IOException {
      if (proxy == null) {
        return false;
      }
      FileStatus stat = path.statIfFound(Symlinks.FOLLOW);
      return stat == null || !stat.isFile() || !proxy.equals(FileContentsProxy.create(stat));
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException(
          "regular file's mtime should never be called. (" + this + ")");
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add(
              "digest",
              digest == null ? "(null)" : BaseEncoding.base16().lowerCase().encode(digest))
          .add("size", size)
          .add("proxy", proxy)
          .toString();
    }

    @Override
    protected boolean couldBeModifiedByMetadata(FileArtifactValue o) {
      if (!(o instanceof RegularFileArtifactValue)) {
        return true;
      }

      RegularFileArtifactValue lastKnown = (RegularFileArtifactValue) o;
      return size != lastKnown.size || !Objects.equals(proxy, lastKnown.proxy);
    }
  }

  private static final class UnshareableRegularFileArtifactValue extends RegularFileArtifactValue {
    private UnshareableRegularFileArtifactValue(
        byte[] digest, @Nullable FileContentsProxy proxy, long size) {
      super(digest, proxy, size);
    }

    @Override
    public boolean dataIsShareable() {
      return false;
    }
  }

  /** Metadata for remotely stored files. */
  public static final class RemoteFileArtifactValue extends FileArtifactValue {
    private final byte[] digest;
    private final long size;
    private final int locationIndex;

    public RemoteFileArtifactValue(byte[] digest, long size, int locationIndex) {
      this.digest = digest;
      this.size = size;
      this.locationIndex = locationIndex;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof RemoteFileArtifactValue)) {
        return false;
      }

      RemoteFileArtifactValue that = (RemoteFileArtifactValue) o;
      return Arrays.equals(digest, that.digest)
          && size == that.size
          && locationIndex == that.locationIndex;
    }

    @Override
    public int hashCode() {
      return Objects.hash(Arrays.hashCode(digest), size, locationIndex, dataIsShareable());
    }

    @Override
    public FileStateType getType() {
      return FileStateType.REGULAR_FILE;
    }

    @Override
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
      return size;
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException(
          "RemoteFileArifactValue doesn't support getModifiedTime");
    }

    @Override
    public int getLocationIndex() {
      return locationIndex;
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      return false;
    }

    @Override
    public boolean isRemote() {
      return true;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("digest", bytesToString(digest))
          .add("size", size)
          .add("locationIndex", locationIndex)
          .toString();
    }
  }

  /** A {@link FileArtifactValue} representing a symlink that is not to be resolved. */
  public static final class UnresolvedSymlinkArtifactValue extends FileArtifactValue {
    private final byte[] digest;

    private UnresolvedSymlinkArtifactValue(byte[] digest) {
      this.digest = digest;
    }

    @Override
    public FileStateType getType() {
      return FileStateType.SYMLINK;
    }

    @Override
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public long getSize() {
      return 0;
    }

    @Override
    public long getModifiedTime() {
      throw new IllegalStateException();
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new IllegalStateException();
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      // We could store an mtime but I have no clue where to get one from createFromMetadata
      return true;
    }
  }

  /** File stored inline in metadata. */
  public static class InlineFileArtifactValue extends FileArtifactValue {
    private final byte[] data;
    private final byte[] digest;

    private InlineFileArtifactValue(byte[] data, byte[] digest) {
      this.data = Preconditions.checkNotNull(data);
      this.digest = Preconditions.checkNotNull(digest);
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof InlineFileArtifactValue)) {
        return false;
      }

      InlineFileArtifactValue that = (InlineFileArtifactValue) o;
      return Arrays.equals(digest, that.digest) && dataIsShareable() == that.dataIsShareable();
    }

    @Override
    public int hashCode() {
      return Objects.hash(Arrays.hashCode(digest), dataIsShareable());
    }

    private InlineFileArtifactValue(byte[] bytes) {
      this(
          bytes,
          DigestHashFunction.getDefaultUnchecked().getHashFunction().hashBytes(bytes).asBytes());
    }

    public static InlineFileArtifactValue create(byte[] bytes, boolean shareable) {
      return shareable
          ? new InlineFileArtifactValue(bytes)
          : new UnshareableInlineFileArtifactValue(bytes);
    }

    public ByteArrayInputStream getInputStream() {
      return new ByteArrayInputStream(data);
    }

    @Override
    public FileStateType getType() {
      return FileStateType.REGULAR_FILE;
    }

    @Override
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
      return data.length;
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      throw new UnsupportedOperationException();
    }
  }

  private static final class UnshareableInlineFileArtifactValue extends InlineFileArtifactValue {
    UnshareableInlineFileArtifactValue(byte[] bytes) {
      super(bytes);
    }

    @Override
    public boolean dataIsShareable() {
      return false;
    }
  }

  /**
   * Used to resolve source symlinks when diskless.
   *
   * <p>When {@link com.google.devtools.build.lib.skyframe.ActionFileSystem} creates symlinks, it
   * relies on metadata ({@link FileArtifactValue}) to resolve the actual underlying data. In the
   * case of remote or inline files, this information is self-contained. However, in the case of
   * source files, the path is required to resolve the content.
   */
  public static final class SourceFileArtifactValue extends FileArtifactValue {
    private final PathFragment execPath;
    private final byte[] digest;
    private final long size;

    public SourceFileArtifactValue(PathFragment execPath, byte[] digest, long size) {
      this.execPath = Preconditions.checkNotNull(execPath);
      this.digest = Preconditions.checkNotNull(digest);
      this.size = size;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof SourceFileArtifactValue)) {
        return false;
      }

      SourceFileArtifactValue that = (SourceFileArtifactValue) o;
      return Objects.equals(execPath, that.execPath)
          && Arrays.equals(digest, that.digest)
          && size == that.size;
    }

    @Override
    public int hashCode() {
      return Objects.hash(execPath, Arrays.hashCode(digest), size);
    }

    public PathFragment getExecPath() {
      return execPath;
    }

    @Override
    public FileStateType getType() {
      return FileStateType.REGULAR_FILE;
    }

    @Override
    public byte[] getDigest() {
      return digest;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
      return size;
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      throw new UnsupportedOperationException();
    }
  }

  private static final class SingletonMarkerValue extends FileArtifactValue implements Singleton {
    @Override
    public FileStateType getType() {
      return FileStateType.NONEXISTENT;
    }

    @Nullable
    @Override
    public byte[] getDigest() {
      return null;
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
      return 0;
    }

    @Override
    public long getModifiedTime() {
      return 0;
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      return false;
    }

    @Nullable
    @Override
    public BigInteger getValueFingerprint() {
      return BigInteger.TEN;
    }

    @Override
    public String toString() {
      return "singleton marker artifact value (" + hashCode() + ")";
    }
  }

  private static final class OmittedFileValue extends FileArtifactValue implements Singleton {
    @Override
    public FileStateType getType() {
      return FileStateType.NONEXISTENT;
    }

    @Override
    public byte[] getDigest() {
      throw new UnsupportedOperationException();
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getSize() {
      throw new UnsupportedOperationException();
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) throws IOException {
      return false;
    }

    @Override
    public String toString() {
      return "OMITTED_FILE_MARKER";
    }
  }
}
