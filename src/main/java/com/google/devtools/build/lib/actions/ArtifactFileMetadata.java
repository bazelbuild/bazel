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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A value that corresponds to the metadata for an artifact, which may be a file or directory or
 * symlink or non-existent file, fully accounting for symlinks (e.g. proper dependencies on ancestor
 * symlinks so as to be incrementally correct).
 *
 * <p>Note that the existence of this object does not imply that the file exists on the filesystem.
 * Values for missing files may be created on purpose in order to facilitate incremental builds in
 * the case those files have reappeared.
 *
 * <p>Very similar to {@link FileValue}, but contains strictly less information: does not have a
 * {@link com.google.devtools.build.lib.vfs.RootedPath}, since execution never needs to access the
 * filesystem via this object.
 */
@Immutable
@ThreadSafe
public abstract class ArtifactFileMetadata {
  /**
   * Exists to accommodate the control flow of {@link
   * com.google.devtools.build.lib.skyframe.ActionMetadataHandler#getMetadata}.
   *
   * <p>{@link com.google.devtools.build.lib.skyframe.ActionMetadataHandler#getMetadata} always
   * checks {@link com.google.devtools.build.lib.skyframe.ActionMetadataHandler#outputArtifactData}
   * before checking {@link
   * com.google.devtools.build.lib.skyframe.ActionMetadataHandler#additionalOutputData} so some
   * placeholder value is needed to allow an injected {@link FileArtifactValue} to be returned.
   *
   * <p>Similarly, {@link
   * com.google.devtools.build.lib.skyframe.ActionExecutionValue#getAllFileValues} replaces this
   * placeholder with metadata from {@link
   * com.google.devtools.build.lib.skyframe.ActionExecutionValue#additionalOutputData}.
   */
  @AutoCodec public static final ArtifactFileMetadata PLACEHOLDER = new PlaceholderFileValue();

  // No implementations outside this class.
  private ArtifactFileMetadata() {}

  public boolean exists() {
    return realFileStateValue().getType() != FileStateType.NONEXISTENT;
  }

  /** Returns true if the original path is a symlink; the target path can never be a symlink. */
  public boolean isSymlink() {
    return false;
  }

  /**
   * Returns true if this value corresponds to a file or symlink to an existing regular or special
   * file. If so, its parent directory is guaranteed to exist.
   */
  public boolean isFile() {
    return realFileStateValue().getType() == FileStateType.REGULAR_FILE
        || realFileStateValue().getType() == FileStateType.SPECIAL_FILE;
  }

  /**
   * Returns true if this value corresponds to a file or symlink to an existing special file. If so,
   * its parent directory is guaranteed to exist.
   */
  public boolean isSpecialFile() {
    return realFileStateValue().getType() == FileStateType.SPECIAL_FILE;
  }

  /**
   * Returns true if the file is a directory or a symlink to an existing directory. If so, its
   * parent directory is guaranteed to exist.
   */
  public boolean isDirectory() {
    return realFileStateValue().getType() == FileStateType.DIRECTORY;
  }

  public abstract FileStateValue realFileStateValue();

  public long getSize() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getSize();
  }

  @Nullable
  public byte[] getDigest() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getDigest();
  }

  public static ArtifactFileMetadata value(
      PathFragment pathFragment,
      FileStateValue fileStateValue,
      PathFragment realPathFragment,
      FileStateValue realFileStateValue) {
    Preconditions.checkState(pathFragment.isAbsolute(), pathFragment);
    Preconditions.checkState(realPathFragment.isAbsolute(), realPathFragment);
    if (pathFragment.equals(realPathFragment)) {
      Preconditions.checkState(
          fileStateValue.getType() != FileStateType.SYMLINK,
          "path: %s, fileStateValue: %s, realPath: %s, realFileStateValue: %s",
          pathFragment,
          fileStateValue,
          realPathFragment,
          realFileStateValue);
      return new Regular(pathFragment, fileStateValue);
    } else {
      if (fileStateValue.getType() == FileStateType.SYMLINK) {
        return new Symlink(realPathFragment, realFileStateValue, fileStateValue.getSymlinkTarget());
      } else {
        return new DifferentRealPath(realPathFragment, realFileStateValue);
      }
    }
  }

  /**
   * Implementation of {@link ArtifactFileMetadata} for files whose fully resolved path is the same
   * as the requested path. For example, this is the case for the path "foo/bar/baz" if neither
   * 'foo' nor 'foo/bar' nor 'foo/bar/baz' are symlinks.
   */
  private static final class Regular extends ArtifactFileMetadata {
    private final PathFragment realPath;
    private final FileStateValue fileStateValue;

    Regular(PathFragment realPath, FileStateValue fileStateValue) {
      this.realPath = Preconditions.checkNotNull(realPath);
      this.fileStateValue = Preconditions.checkNotNull(fileStateValue);
    }

    @Override
    public FileStateValue realFileStateValue() {
      return fileStateValue;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != Regular.class) {
        return false;
      }
      Regular other = (Regular) obj;
      return realPath.equals(other.realPath) && fileStateValue.equals(other.fileStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realPath, fileStateValue);
    }

    @Override
    public String toString() {
      return realPath + ", " + fileStateValue;
    }
  }

  /**
   * Base class for {@link ArtifactFileMetadata}s for files whose fully resolved path is different
   * than the requested path. For example, this is the case for the path "foo/bar/baz" if at least
   * one of 'foo', 'foo/bar', or 'foo/bar/baz' is a symlink.
   */
  private static class DifferentRealPath extends ArtifactFileMetadata {
    protected final PathFragment realPath;
    protected final FileStateValue realFileStateValue;

    DifferentRealPath(PathFragment realPath, FileStateValue realFileStateValue) {
      this.realPath = Preconditions.checkNotNull(realPath);
      this.realFileStateValue = Preconditions.checkNotNull(realFileStateValue);
    }

    @Override
    public FileStateValue realFileStateValue() {
      return realFileStateValue;
    }

    @SuppressWarnings("EqualsGetClass") // Only subclass should never be equal to this class.
    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != DifferentRealPath.class) {
        return false;
      }
      DifferentRealPath other = (DifferentRealPath) obj;
      return realPath.equals(other.realPath) && realFileStateValue.equals(other.realFileStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realPath, realFileStateValue);
    }

    @Override
    public String toString() {
      return realPath + ", " + realFileStateValue + " (symlink ancestor)";
    }
  }

  /** Implementation of {@link ArtifactFileMetadata} for files that are symlinks. */
  private static final class Symlink extends DifferentRealPath {
    private final PathFragment linkTarget;

    private Symlink(
        PathFragment realPath, FileStateValue realFileStateValue, PathFragment linkTarget) {
      super(realPath, realFileStateValue);
      this.linkTarget = linkTarget;
    }

    @Override
    public boolean isSymlink() {
      return true;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != Symlink.class) {
        return false;
      }
      Symlink other = (Symlink) obj;
      return realPath.equals(other.realPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && linkTarget.equals(other.linkTarget);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realPath, realFileStateValue, linkTarget, Boolean.TRUE);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink (real_path=%s, real_state=%s, link_value=%s)",
          realPath, realFileStateValue, linkTarget);
    }
  }

  private static final class PlaceholderFileValue extends ArtifactFileMetadata {
    private PlaceholderFileValue() {}
    @Override
    public FileStateValue realFileStateValue() {
      throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
      return "PlaceholderFileValue:Singleton";
    }
  }
}
