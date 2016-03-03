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
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

/**
 * A value that represents the list of files in a given directory under a given package path root.
 * Anything in Skyframe that cares about the contents of a directory should have a dependency
 * on the corresponding {@link DirectoryListingValue}.
 *
 * <p>This value only depends on the FileValue corresponding to the directory. In particular, note
 * that it does not depend on any of its child entries.
 *
 * <p>Note that symlinks in dirents are <b>not</b> expanded. Dependents of the value are responsible
 * for expanding the symlink entries by referring to FileValues that correspond to the symlinks.
 * This is a little onerous, but correct: we do not need to reread the directory when a symlink
 * inside it changes, therefore this value should not be invalidated in that case.
 */
@Immutable
@ThreadSafe
public abstract class DirectoryListingValue implements SkyValue {

  /**
   * Returns the directory entries for this directory, in a stable order.
   *
   * <p>Symlinks are not expanded.
   */
  public Dirents getDirents() {
    return getDirectoryListingStateValue().getDirents();
  }

  public abstract DirectoryListingStateValue getDirectoryListingStateValue();

  /**
   * Returns a {@link SkyKey} for getting the directory entries of the given directory. The
   * given path is assumed to be an existing directory (e.g. via {@link FileValue#isDirectory} or
   * from a directory listing on its parent directory).
   */
  @ThreadSafe
  static SkyKey key(RootedPath directoryUnderRoot) {
    return SkyKey.create(SkyFunctions.DIRECTORY_LISTING, directoryUnderRoot);
  }

  static DirectoryListingValue value(RootedPath dirRootedPath, FileValue dirFileValue,
      DirectoryListingStateValue realDirectoryListingStateValue) {
    return dirFileValue.realRootedPath().equals(dirRootedPath)
        ? new RegularDirectoryListingValue(realDirectoryListingStateValue)
        : new DifferentRealPathDirectoryListingValue(dirFileValue.realRootedPath(),
            realDirectoryListingStateValue);
  }

  /** Normal {@link DirectoryListingValue}. */
  @ThreadSafe
  public static final class RegularDirectoryListingValue extends DirectoryListingValue {

    private final DirectoryListingStateValue directoryListingStateValue;

    public RegularDirectoryListingValue(DirectoryListingStateValue directoryListingStateValue) {
      this.directoryListingStateValue = directoryListingStateValue;
    }

    @Override
    public DirectoryListingStateValue getDirectoryListingStateValue() {
      return directoryListingStateValue;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof RegularDirectoryListingValue)) {
        return false;
      }
      RegularDirectoryListingValue other = (RegularDirectoryListingValue) obj;
      return directoryListingStateValue.equals(other.directoryListingStateValue);
    }

    @Override
    public int hashCode() {
      return directoryListingStateValue.hashCode();
    }
  }

  /** A {@link DirectoryListingValue} with a different root. */
  @ThreadSafe
  public static final class DifferentRealPathDirectoryListingValue extends DirectoryListingValue {

    private final RootedPath realDirRootedPath;
    private final DirectoryListingStateValue directoryListingStateValue;

    public DifferentRealPathDirectoryListingValue(RootedPath realDirRootedPath,
        DirectoryListingStateValue directoryListingStateValue) {
      this.realDirRootedPath = realDirRootedPath;
      this.directoryListingStateValue = directoryListingStateValue;
    }

    public RootedPath getRealDirRootedPath() {
      return realDirRootedPath;
    }

    @Override
    public DirectoryListingStateValue getDirectoryListingStateValue() {
      return directoryListingStateValue;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof DifferentRealPathDirectoryListingValue)) {
        return false;
      }
      DifferentRealPathDirectoryListingValue other = (DifferentRealPathDirectoryListingValue) obj;
      return realDirRootedPath.equals(other.realDirRootedPath)
          && directoryListingStateValue.equals(other.directoryListingStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realDirRootedPath, directoryListingStateValue);
    }
  }
}
