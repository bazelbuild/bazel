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

import com.google.common.base.MoreObjects;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/**
 * Encapsulates the filesystem operations needed to get the directory entries of a directory.
 *
 * <p>This class is an implementation detail of {@link DirectoryListingValue}.
 */
@VisibleForSerialization
public final class DirectoryListingStateValue implements SkyValue {

  private final CompactSortedDirents compactSortedDirents;

  private DirectoryListingStateValue(Collection<Dirent> dirents) {
    this.compactSortedDirents = CompactSortedDirents.create(dirents);
  }

  @AutoCodec.Instantiator
  public static DirectoryListingStateValue create(Collection<Dirent> dirents) {
    return new DirectoryListingStateValue(dirents);
  }

  @ThreadSafe
  public static Key key(RootedPath rootedPath) {
    return Key.create(rootedPath);
  }

  /** Key type for DirectoryListingStateValue. */
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
      return SkyFunctions.DIRECTORY_LISTING_STATE;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  /**
   * Returns the directory entries for this directory, in a stable order.
   *
   * <p>Symlinks are not expanded.
   */
  public Dirents getDirents() {
    return compactSortedDirents;
  }

  @Override
  public int hashCode() {
    return compactSortedDirents.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof DirectoryListingStateValue other)) {
      return false;
    }
    return compactSortedDirents.equals(other.compactSortedDirents);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("dirents", Iterables.toString(getDirents()))
        .toString();
  }

}
