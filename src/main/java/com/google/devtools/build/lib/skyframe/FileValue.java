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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.FileStateValue.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A value that corresponds to a file (or directory or symlink or non-existent file), fully
 * accounting for symlinks (e.g. proper dependencies on ancestor symlinks so as to be incrementally
 * correct). Anything in Skyframe that cares about the fully resolved path of a file (e.g. anything
 * that cares about the contents of a file) should have a dependency on the corresponding
 * {@link FileValue}.
 *
 * <p>
 * Note that the existence of a file value does not imply that the file exists on the filesystem.
 * File values for missing files will be created on purpose in order to facilitate incremental
 * builds in the case those files have reappeared.
 *
 * <p>
 * This class contains the relevant metadata for a file, although not the contents. Note that
 * since a FileValue doesn't store its corresponding SkyKey, it's possible for the FileValues for
 * two different paths to be the same.
 *
 * <p>
 * This should not be used for build outputs; use {@link ArtifactValue} for those.
 */
@Immutable
@ThreadSafe
public abstract class FileValue implements SkyValue {

  public boolean exists() {
    return realFileStateValue().getType() != Type.NONEXISTENT;
  }

  public boolean isSymlink() {
    return false;
  }

  /**
   * Returns true if this value corresponds to a file or symlink to an existing regular or special
   * file. If so, its parent directory is guaranteed to exist.
   */
  public boolean isFile() {
    return realFileStateValue().getType() == Type.REGULAR_FILE
        || realFileStateValue().getType() == Type.SPECIAL_FILE;
  }

  /**
   * Returns true if this value corresponds to a file or symlink to an existing special file. If so,
   * its parent directory is guaranteed to exist.
   */
  public boolean isSpecialFile() {
    return realFileStateValue().getType() == Type.SPECIAL_FILE;
  }

  /**
   * Returns true if the file is a directory or a symlink to an existing directory. If so, its
   * parent directory is guaranteed to exist.
   */
  public boolean isDirectory() {
    return realFileStateValue().getType() == Type.DIRECTORY;
  }

  /**
   * Returns the real rooted path of the file, taking ancestor symlinks into account. For example,
   * the rooted path ['root']/['a/b'] is really ['root']/['c/b'] if 'a' is a symlink to 'b'. Note
   * that ancestor symlinks outside the root boundary are not taken into consideration.
   */
  public abstract RootedPath realRootedPath();

  public abstract FileStateValue realFileStateValue();

  /**
   * Returns the unresolved link target if {@link #isSymlink()}.
   *
   * <p>This is useful if the caller wants to, for example, duplicate a relative symlink. An actual
   * example could be a build rule that copies a set of input files to the output directory, but
   * upon encountering symbolic links it can decide between copying or following them.
   */
  PathFragment getUnresolvedLinkTarget() {
    throw new IllegalStateException(this.toString());
  }

  long getSize() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getSize();
  }

  @Nullable
  byte[] getDigest() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getDigest();
  }

  /**
   * Returns a key for building a file value for the given root-relative path.
   */
  @ThreadSafe
  public static SkyKey key(RootedPath rootedPath) {
    return SkyKey.create(SkyFunctions.FILE, rootedPath);
  }

  /**
   * Only intended to be used by {@link FileFunction}. Should not be used for symlink cycles.
   */
  static FileValue value(RootedPath rootedPath, FileStateValue fileStateValue,
                         RootedPath realRootedPath, FileStateValue realFileStateValue) {
    if (rootedPath.equals(realRootedPath)) {
      Preconditions.checkState(fileStateValue.getType() != FileStateValue.Type.SYMLINK,
          "rootedPath: %s, fileStateValue: %s, realRootedPath: %s, realFileStateValue: %s",
          rootedPath, fileStateValue, realRootedPath, realFileStateValue);
      return new RegularFileValue(rootedPath, fileStateValue);
    } else {
      if (fileStateValue.getType() == FileStateValue.Type.SYMLINK) {
        return new SymlinkFileValue(realRootedPath, realFileStateValue,
            fileStateValue.getSymlinkTarget());
      } else {
        return new DifferentRealPathFileValue(
            realRootedPath, realFileStateValue);
      }
    }
  }

  /**
   * Implementation of {@link FileValue} for files whose fully resolved path is the same as the
   * requested path. For example, this is the case for the path "foo/bar/baz" if neither 'foo' nor
   * 'foo/bar' nor 'foo/bar/baz' are symlinks.
   */
  public static final class RegularFileValue extends FileValue {

    private final RootedPath rootedPath;
    private final FileStateValue fileStateValue;

    public RegularFileValue(RootedPath rootedPath, FileStateValue fileState) {
      this.rootedPath = Preconditions.checkNotNull(rootedPath);
      this.fileStateValue = Preconditions.checkNotNull(fileState);
    }

    @Override
    public RootedPath realRootedPath() {
      return rootedPath;
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
      if (obj.getClass() != RegularFileValue.class) {
        return false;
      }
      RegularFileValue other = (RegularFileValue) obj;
      return rootedPath.equals(other.rootedPath) && fileStateValue.equals(other.fileStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(rootedPath, fileStateValue);
    }

    @Override
    public String toString() {
      return rootedPath + ", " + fileStateValue;
    }
  }

  /**
   * Base class for {@link FileValue}s for files whose fully resolved path is different than the
   * requested path. For example, this is the case for the path "foo/bar/baz" if at least one of
   * 'foo', 'foo/bar', or 'foo/bar/baz' is a symlink.
   */
  public static class DifferentRealPathFileValue extends FileValue {

    protected final RootedPath realRootedPath;
    protected final FileStateValue realFileStateValue;

    public DifferentRealPathFileValue(RootedPath realRootedPath,
                                      FileStateValue realFileStateValue) {
      this.realRootedPath = Preconditions.checkNotNull(realRootedPath);
      this.realFileStateValue = Preconditions.checkNotNull(realFileStateValue);
    }

    @Override
    public RootedPath realRootedPath() {
      return realRootedPath;
    }

    @Override
    public FileStateValue realFileStateValue() {
      return realFileStateValue;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != DifferentRealPathFileValue.class) {
        return false;
      }
      DifferentRealPathFileValue other = (DifferentRealPathFileValue) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realRootedPath, realFileStateValue);
    }

    @Override
    public String toString() {
      return realRootedPath + ", " + realFileStateValue + " (symlink ancestor)";
    }
  }

  /** Implementation of {@link FileValue} for files that are symlinks. */
  public static final class SymlinkFileValue extends DifferentRealPathFileValue {
    private final PathFragment linkValue;

    public SymlinkFileValue(RootedPath realRootedPath, FileStateValue realFileState,
                            PathFragment linkTarget) {
      super(realRootedPath, realFileState);
      this.linkValue = linkTarget;
    }

    @Override
    public boolean isSymlink() {
      return true;
    }

    @Override
    public PathFragment getUnresolvedLinkTarget() {
      return linkValue;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != SymlinkFileValue.class) {
        return false;
      }
      SymlinkFileValue other = (SymlinkFileValue) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && linkValue.equals(other.linkValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          realRootedPath, realFileStateValue, linkValue, Boolean.TRUE);
    }

    @Override
    public String toString() {
      return String.format("symlink (real_path=%s, real_state=%s, link_value=%s)",
          realRootedPath, realFileStateValue, linkValue);
    }
  }
}
