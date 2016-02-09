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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Arrays;

import javax.annotation.Nullable;

/**
 * Stores the actual metadata data of a file. We have the following cases:
 * <ul><li>
 *   an ordinary file, in which case we would expect to see a digest and size;
 * </li><li>
 *   a directory, in which case we would expect to see an mtime;
 * </li><li>
 *   an empty file corresponding to an Artifact, where we would expect to see a size (=0), mtime,
 *   and digest;
 * </li><li>
 *   an intentionally omitted file which the build system is aware of but doesn't actually exist,
 *   where all access methods are unsupported;
 * </li><li>
 *   The "self data" of a middleman artifact or TreeArtifact, where we would expect to see a digest
 *   representing the artifact's contents, and a size of 1.
 * </li></ul>
 */
public class FileArtifactValue extends ArtifactValue {
  /** Data for Middleman artifacts that did not have data specified. */
  static final FileArtifactValue DEFAULT_MIDDLEMAN = new FileArtifactValue(null, 0, 0);
  /** Data that marks that a file is not present on the filesystem. */
  @VisibleForTesting
  public static final FileArtifactValue MISSING_FILE_MARKER = new FileArtifactValue(null, 1, 0) {
    @Override
    public boolean exists() {
      return false;
    }
  };

  /**
   * Represents an omitted file -- we are aware of it but it doesn't exist. All access methods
   * are unsupported.
   */
  static final FileArtifactValue OMITTED_FILE_MARKER = new FileArtifactValue(null, 2, 0) {
    @Override public byte[] getDigest() { throw new UnsupportedOperationException(); }
    @Override public boolean isFile() { throw new UnsupportedOperationException(); }
    @Override public long getSize() { throw new UnsupportedOperationException(); }
    @Override public long getModifiedTime() { throw new UnsupportedOperationException(); }
    @Override public boolean equals(Object o) { return this == o; }
    @Override public int hashCode() { return System.identityHashCode(this); }
    @Override public String toString() { return "OMITTED_FILE_MARKER"; }
  };

  @Nullable private final byte[] digest;
  private final long mtime;
  private final long size;

  private FileArtifactValue(byte[] digest, long size) {
    this.digest = Preconditions.checkNotNull(digest, size);
    this.size = size;
    this.mtime = -1;
  }

  // Only used by empty files (non-null digest) and directories (null digest).
  private FileArtifactValue(byte[] digest, long mtime, long size) {
    Preconditions.checkState(mtime >= 0, "mtime must be non-negative: %s %s", mtime, size);
    Preconditions.checkState(size == 0, "size must be zero: %s %s", mtime, size);
    this.digest = digest;
    this.size = size;
    this.mtime = mtime;
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
    if (!DigestUtils.useFileDigest(isFile, size)) {
      // In this case, we need to store the mtime because the action cache uses mtime to determine
      // if this artifact has changed. This is currently true for empty files and directories. We
      // do not optimize for this code path (by storing the mtime in a FileValue) because we do not
      // like it and may remove this special-casing for empty files in the future. We want this code
      // path to go away somehow too for directories (maybe by implementing FileSet
      // in Skyframe)
      return new FileArtifactValue(digest, artifact.getPath().getLastModifiedTime(), size);
    }
    Preconditions.checkState(digest != null, artifact);
    return new FileArtifactValue(digest, size);
  }

  /** Returns a FileArtifactValue with the given digest, even for empty files (size = 0). */
  static FileArtifactValue createWithDigest(Path path, byte[] digest, long size)
      throws IOException {
    // Eventually, we want to migrate everything away from using mtimes instead of digests.
    // But right now, some cases always use digests (TreeArtifacts) and some don't.
    // So we have different constructors.
    if (digest == null) {
      digest = DigestUtils.getDigestOrFail(path, size);
    }
    return new FileArtifactValue(digest, size);
  }

  /**
   * Creates a FileArtifactValue used as a 'proxy' input for other ArtifactValues.
   * These are used in {@link com.google.devtools.build.lib.actions.ActionCacheChecker}.
   */
  static FileArtifactValue createProxy(byte[] digest) {
    Preconditions.checkNotNull(digest);
    // The Middleman artifact values have size 1 because we want their digests to be used. This hack
    // can be removed once empty files are digested.
    return new FileArtifactValue(digest, /*size=*/1);
  }

  @Nullable
  public byte[] getDigest() {
    return digest;
  }

  /** @return true if this is a file or a symlink to an existing file */
  boolean isFile() {
    return digest != null;
  }

  /** Gets the size of the file. Directories have size 0. */
  public long getSize() {
    return size;
  }

  /**
   * Gets last modified time of file. Should only be called if {@link DigestUtils#useFileDigest} was
   * false for this artifact -- namely, either it is a directory or an empty file. Note that since
   * we store directory sizes as 0, all files for which this method can be called have size 0.
   */
  long getModifiedTime() {
    Preconditions.checkState(size == 0, "%s %s %s", digest, mtime, size);
    return mtime;
  }

  public boolean exists() {
    return true;
  }

  @Override
  public int hashCode() {
    // Hash digest by content, not reference. Note that digest is the only array in this array.
    return Arrays.deepHashCode(new Object[] {size, mtime, digest});
  }

  /**
   * Two FileArtifactValues will only compare equal if they have the same content. This differs
   * from the {@code Metadata#equivalence} method, which allows for comparison using mtime if
   * one object does not have a digest available.
   */
  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof FileArtifactValue)) {
      return false;
    }
    FileArtifactValue that = (FileArtifactValue) other;
    return this.mtime == that.mtime && this.size == that.size
        && Arrays.equals(this.digest, that.digest);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(FileArtifactValue.class)
        .add("digest", digest)
        .add("mtime", mtime)
        .add("size", size).toString();
  }
}
