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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Arrays;
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
    public boolean isFile() {
      return false;
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
    public boolean isFile() {
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
    public String toString() {
      return "OMITTED_FILE_MARKER";
    }
  }

  static final FileArtifactValue DEFAULT_MIDDLEMAN = new SingletonMarkerValue();
  /** Data that marks that a file is not present on the filesystem. */
  @VisibleForTesting
  public static final FileArtifactValue MISSING_FILE_MARKER = new SingletonMarkerValue();

  /**
   * Represents an omitted file -- we are aware of it but it doesn't exist. All access methods are
   * unsupported.
   */
  static final FileArtifactValue OMITTED_FILE_MARKER = new OmittedFileValue();

  private static final class DirectoryArtifactValue extends FileArtifactValue {
    private final long mtime;

    private DirectoryArtifactValue(long mtime) {
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
    public boolean isFile() {
      return false;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("mtime", mtime).toString();
    }
  }

  private static final class RegularFileArtifactValue extends FileArtifactValue {
    private final byte[] digest;
    private final long size;

    private RegularFileArtifactValue(byte[] digest, long size) {
      this.digest = Preconditions.checkNotNull(digest);
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
    public boolean isFile() {
      return true;
    }

    @Override
    public long getSize() {
      return size;
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException(
          "regular file's mtime should never be called. (" + this + ")");
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("digest", digest).add("size", size).toString();
    }
  }

  @VisibleForTesting
  public static FileArtifactValue create(Artifact artifact) throws IOException {
    Path path = artifact.getPath();
    FileStatus stat = path.stat();
    boolean isFile = stat.isFile();
    return create(artifact, isFile, isFile ? stat.getSize() : 0, null);
  }

  static FileArtifactValue create(Artifact artifact, FileValue fileValue) throws IOException {
    boolean isFile = fileValue.isFile();
    return create(artifact, isFile, isFile ? fileValue.getSize() : 0,
        isFile ? fileValue.getDigest() : null);
  }

  static FileArtifactValue create(Artifact artifact, boolean isFile, long size,
      @Nullable byte[] digest) throws IOException {
    if (isFile && digest == null) {
      digest = DigestUtils.getDigestOrFail(artifact.getPath(), size);
    }
    if (!isFile) {
      // In this case, we need to store the mtime because the action cache uses mtime for
      // directories to determine if this artifact has changed. We want this code path to go away
      // somehow (maybe by implementing FileSet in Skyframe).
      return new DirectoryArtifactValue(artifact.getPath().getLastModifiedTime());
    }
    Preconditions.checkState(digest != null, artifact);
    return createNormalFile(digest, size);
  }

  public static FileArtifactValue createNormalFile(byte[] digest, long size) {
    return new RegularFileArtifactValue(digest, size);
  }

  static FileArtifactValue createNormalFile(FileValue fileValue) {
    return new RegularFileArtifactValue(fileValue.getDigest(), fileValue.getSize());
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
    return createNormalFile(digest, /*size=*/ 0);
  }

  @Override
  public abstract boolean isFile();

  @Nullable
  @Override
  public abstract byte[] getDigest();

  @Override
  public abstract long getSize();

  @Override
  public abstract long getModifiedTime();

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
    if (isFile()) {
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
    if (isFile()) {
      return 37 * Long.hashCode(getSize()) + Arrays.hashCode(getDigest());
    } else {
      return Long.hashCode(getModifiedTime());
    }
  }
}
