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
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Stores the actual metadata data of a file. We have the following cases:
 *
 * <ul>
 * <li> an ordinary file, in which case we would expect to see a digest and size;
 * <li> a directory, in which case we would expect to see an mtime;
 * <li> an intentionally omitted file which the build system is aware of but doesn't actually exist,
 *     where all access methods are unsupported;
 * <li> a "middleman marker" object, which has a null digest, 0 size, and mtime of 0.
 * <li> The "self data" of a TreeArtifact, where we would expect to see a digest representing the
 *     artifact's contents, and a size of 0.
 * </ul>
 */
// TODO(janakr): make this an interface once JDK8 allows us to have static methods on interfaces.
@Immutable @ThreadSafe
public abstract class FileArtifactValue implements SkyValue, Metadata {
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
    public long getSize() {
      return 0;
    }

    @Override
    public long getModifiedTime() {
      return 0;
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) throws IOException {
      return false;
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

  @AutoCodec static final FileArtifactValue DEFAULT_MIDDLEMAN = new SingletonMarkerValue();
  /** Data that marks that a file is not present on the filesystem. */
  @VisibleForTesting @AutoCodec
  public static final FileArtifactValue MISSING_FILE_MARKER = new SingletonMarkerValue();

  /**
   * Represents an omitted file -- we are aware of it but it doesn't exist. All access methods are
   * unsupported.
   */
  @AutoCodec static final FileArtifactValue OMITTED_FILE_MARKER = new OmittedFileValue();

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static final class DirectoryArtifactValue extends FileArtifactValue {
    private final long mtime;

    @AutoCodec.VisibleForSerialization
    DirectoryArtifactValue(long mtime) {
      this.mtime = mtime;
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

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static final class RegularFileArtifactValue extends FileArtifactValue {
    private final byte[] digest;
    @Nullable private final FileContentsProxy proxy;
    private final long size;

    @AutoCodec.VisibleForSerialization
    RegularFileArtifactValue(byte[] digest, @Nullable FileContentsProxy proxy, long size) {
      this.digest = Preconditions.checkNotNull(digest);
      this.proxy = proxy;
      this.size = size;
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
          .add("digest", BaseEncoding.base16().lowerCase().encode(digest))
          .add("size", size)
          .add("proxy", proxy).toString();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof RegularFileArtifactValue)) {
        return false;
      }
      RegularFileArtifactValue r = (RegularFileArtifactValue) o;
      return Arrays.equals(digest, r.digest) && Objects.equals(proxy, r.proxy) && size == r.size;
    }

    @Override
    public int hashCode() {
      return (proxy != null ? 127 * proxy.hashCode() : 0)
          + 37 * Long.hashCode(getSize()) + Arrays.hashCode(getDigest());
    }
  }

  static final class RemoteFileArtifactValue extends FileArtifactValue {
    private final byte[] digest;
    private final long size;
    private final long modifiedTime;
    private final int locationIndex;

    RemoteFileArtifactValue(byte[] digest, long size, long modifiedTime, int locationIndex) {
      this.digest = digest;
      this.size = size;
      this.modifiedTime = modifiedTime;
      this.locationIndex = locationIndex;
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
    public long getSize() {
      return size;
    }

    @Override
    public long getModifiedTime() {
      return modifiedTime;
    }

    @Override
    public int getLocationIndex() {
      return locationIndex;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof RemoteFileArtifactValue)) {
        return false;
      }
      RemoteFileArtifactValue r = (RemoteFileArtifactValue) o;
      return Arrays.equals(digest, r.digest)
          && size == r.size
          && modifiedTime == r.modifiedTime
          && locationIndex == r.locationIndex;
    }

    @Override
    public int hashCode() {
      return Objects.hash(Arrays.hashCode(digest), size, modifiedTime, locationIndex);
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      throw new UnsupportedOperationException();
    }
  }

  static FileArtifactValue create(Artifact artifact, FileValue fileValue) throws IOException {
    boolean isFile = fileValue.isFile();
    FileContentsProxy proxy = getProxyFromFileStateValue(fileValue.realFileStateValue());
    return create(artifact.getPath(), isFile, isFile ? fileValue.getSize() : 0, proxy,
        isFile ? fileValue.getDigest() : null);
  }

  static FileArtifactValue create(
      Artifact artifact, FileValue fileValue, @Nullable byte[] injectedDigest) throws IOException {
    boolean isFile = fileValue.isFile();
    FileContentsProxy proxy = getProxyFromFileStateValue(fileValue.realFileStateValue());
    return create(artifact.getPath(), isFile, isFile ? fileValue.getSize() : 0, proxy,
        injectedDigest);
  }

  @VisibleForTesting
  public static FileArtifactValue create(Artifact artifact) throws IOException {
    return create(artifact.getPath());
  }

  @VisibleForTesting
  public static FileArtifactValue create(Path path) throws IOException {
    // Caution: there's a race condition between stating the file and computing the
    // digest. We need to stat first, since we're using the stat to detect changes.
    // We follow symlinks here to be consistent with getDigest.
    FileStatus stat = path.stat(Symlinks.FOLLOW);
    return create(path, stat.isFile(), stat.getSize(), FileContentsProxy.create(stat), null);
  }

  private static FileArtifactValue create(
      Path path, boolean isFile, long size, FileContentsProxy proxy, @Nullable byte[] digest)
          throws IOException {
    if (!isFile) {
      // In this case, we need to store the mtime because the action cache uses mtime for
      // directories to determine if this artifact has changed. We want this code path to go away
      // somehow (maybe by implementing FileSet in Skyframe).
      return new DirectoryArtifactValue(path.getLastModifiedTime());
    }
    if (digest == null) {
      digest = DigestUtils.getDigestOrFail(path, size);
    }
    Preconditions.checkState(digest != null, path);
    return new RegularFileArtifactValue(digest, proxy, size);
  }

  public static FileArtifactValue createForVirtualActionInput(byte[] digest, long size) {
    return new RegularFileArtifactValue(digest, /*proxy=*/ null, size);
  }

  public static FileArtifactValue createNormalFile(
      byte[] digest, @Nullable FileContentsProxy proxy, long size) {
    return new RegularFileArtifactValue(digest, proxy, size);
  }

  static FileArtifactValue createNormalFile(FileValue fileValue) {
    FileContentsProxy proxy = getProxyFromFileStateValue(fileValue.realFileStateValue());
    return new RegularFileArtifactValue(fileValue.getDigest(), proxy, fileValue.getSize());
  }

  private static FileContentsProxy getProxyFromFileStateValue(FileStateValue value) {
    if (value instanceof FileStateValue.RegularFileStateValue) {
      return ((FileStateValue.RegularFileStateValue) value).getContentsProxy();
    } else if (value instanceof FileStateValue.SpecialFileStateValue) {
      return ((FileStateValue.SpecialFileStateValue) value).getContentsProxy();
    }
    return null;
  }

  @VisibleForTesting
  public static FileArtifactValue createNormalFile(byte[] digest, long size) {
    return createNormalFile(digest, /*proxy=*/null, size);
  }

  public static FileArtifactValue createDirectory(long mtime) {
    return new DirectoryArtifactValue(mtime);
  }

  /**
   * Creates a FileArtifactValue used as a 'proxy' input for other ArtifactValues.
   * These are used in {@link com.google.devtools.build.lib.actions.ActionCacheChecker}.
   */
  static FileArtifactValue createProxy(byte[] digest) {
    Preconditions.checkNotNull(digest);
    return createNormalFile(digest, /*proxy=*/ null, /*size=*/ 0);
  }

  @Override
  public abstract FileStateType getType();

  @Nullable
  @Override
  public abstract byte[] getDigest();

  @Override
  public abstract long getSize();

  @Override
  public abstract long getModifiedTime();

  /**
   * Provides a best-effort determination whether the file was changed since the digest was
   * computed. This method performs file system I/O, so may be expensive. It's primarily intended to
   * avoid storing bad cache entries in an action cache. It should return true if there is a chance
   * that the file was modified since the digest was computed. Better not upload if we are not sure
   * that the cache entry is reliable.
   */
  public abstract boolean wasModifiedSinceDigest(Path path) throws IOException;

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Metadata)) {
      return false;
    }
    if ((this instanceof Singleton) || (o instanceof Singleton)) {
      return false;
    }
    Metadata m = (Metadata) o;
    if (getType() != m.getType()) {
      return false;
    }
    if (getDigest() != null) {
      return Arrays.equals(getDigest(), m.getDigest()) && getSize() == m.getSize();
    } else {
      return getModifiedTime() == m.getModifiedTime();
    }
  }

  @Override
  public int hashCode() {
    if (this instanceof Singleton) {
      return System.identityHashCode(this);
    }
    // Hash digest by content, not reference.
    if (getDigest() != null) {
      return 37 * Long.hashCode(getSize()) + Arrays.hashCode(getDigest());
    } else {
      return Long.hashCode(getModifiedTime());
    }
  }
}
