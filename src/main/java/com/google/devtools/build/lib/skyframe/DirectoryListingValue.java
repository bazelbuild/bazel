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

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value that represents the dirents (name and type of child entries) in a given directory under a
 * given package path root, fully accounting for symlinks in the directory's path. Anything in
 * Skyframe that cares about the contents of a directory should have a dependency on the
 * corresponding {@link DirectoryListingValue}.
 *
 * <p>Note that dirents that are themselves symlinks are <b>not</b> resolved. Consumers of such a
 * dirent are responsible for resolving the symlink entry via an appropriate {@link FileValue}.
 * This is a little onerous, but correct: we do not need to reread the directory when a symlink
 * inside it changes (or, more generally, when the *contents* of a dirent changes), therefore the
 * {@link DirectoryListingValue} value should not be invalidated in that case.
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
   * Returns a {@link Key} for getting the directory entries of the given directory. The given path
   * is assumed to be an existing directory (e.g. via {@link FileValue#isDirectory} or from a
   * directory listing on its parent directory).
   */
  @ThreadSafe
  public static Key key(RootedPath directoryUnderRoot) {
    return Key.create(directoryUnderRoot);
  }

  /** Key type for DirectoryListingValue. */
  @VisibleForSerialization
  @AutoCodec
  public static class Key extends AbstractSkyKey<RootedPath> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RootedPath arg) {
      super(arg);
    }

    private static Key create(RootedPath arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.DIRECTORY_LISTING;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
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
      if (!(obj instanceof RegularDirectoryListingValue other)) {
        return false;
      }
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
      if (!(obj instanceof DifferentRealPathDirectoryListingValue other)) {
        return false;
      }
      return realDirRootedPath.equals(other.realDirRootedPath)
          && directoryListingStateValue.equals(other.directoryListingStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realDirRootedPath, directoryListingStateValue);
    }
  }
}
