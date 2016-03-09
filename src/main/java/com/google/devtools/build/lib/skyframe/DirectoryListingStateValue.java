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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Dirent.Type;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Objects;

import javax.annotation.Nullable;

/**
 * Encapsulates the filesystem operations needed to get the directory entries of a directory.
 *
 * <p>This class is an implementation detail of {@link DirectoryListingValue}.
 */
public final class DirectoryListingStateValue implements SkyValue {

  private final CompactSortedDirents compactSortedDirents;

  private DirectoryListingStateValue(Collection<Dirent> dirents) {
    this.compactSortedDirents = CompactSortedDirents.create(dirents);
  }

  public static DirectoryListingStateValue create(Collection<Dirent> dirents) {
    return new DirectoryListingStateValue(dirents);
  }

  public static DirectoryListingStateValue create(RootedPath dirRootedPath) throws IOException {
    Collection<Dirent> dirents = dirRootedPath.asPath().readdir(Symlinks.NOFOLLOW);
    return create(dirents);
  }

  @ThreadSafe
  public static SkyKey key(RootedPath rootedPath) {
    return SkyKey.create(SkyFunctions.DIRECTORY_LISTING_STATE, rootedPath);
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
    if (!(obj instanceof DirectoryListingStateValue)) {
      return false;
    }
    DirectoryListingStateValue other = (DirectoryListingStateValue) obj;
    return compactSortedDirents.equals(other.compactSortedDirents);
  }

  /** A space-efficient, sorted, immutable dirent structure. */
  private static class CompactSortedDirents implements Dirents, Serializable {

    private final String[] names;
    private final BitSet packedTypes;

    private CompactSortedDirents(String[] names, BitSet packedTypes) {
      this.names = names;
      this.packedTypes = packedTypes;
    }

    public static CompactSortedDirents create(Collection<Dirent> dirents) {
      final Dirent[] direntArray = dirents.toArray(new Dirent[dirents.size()]);
      Integer[] indices = new Integer[dirents.size()];
      for (int i = 0; i < dirents.size(); i++) {
        indices[i] = i;
      }
      Arrays.sort(indices,
          new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
              return direntArray[o1].getName().compareTo(direntArray[o2].getName());
            }
          });
      String[] names = new String[dirents.size()];
      BitSet packedTypes = new BitSet(dirents.size() * 2);
      for (int i = 0; i < dirents.size(); i++) {
        Dirent dirent = direntArray[indices[i]];
        names[i] = dirent.getName();
        packType(packedTypes, dirent.getType(), i);
      }
      return new CompactSortedDirents(names, packedTypes);
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof CompactSortedDirents)) {
        return false;
      }
      if (this == obj) {
        return true;
      }
      CompactSortedDirents other = (CompactSortedDirents) obj;
      return Arrays.equals(names,  other.names) && packedTypes.equals(other.packedTypes);
    }

    @Override
    public int hashCode() {
      return Objects.hash(Arrays.hashCode(names), packedTypes);
    }

    @Override
    @Nullable
    public Dirent maybeGetDirent(String baseName) {
      int pos = Arrays.binarySearch(names, baseName);
      return pos < 0 ? null : direntAt(pos);
    }

    @Override
    public Iterator<Dirent> iterator() {
      return new Iterator<Dirent>() {

        private int i = 0;

        @Override
        public boolean hasNext() {
          return i < size();
        }

        @Override
        public Dirent next() {
          return direntAt(i++);
        }

        @Override
        public void remove() {
          throw new UnsupportedOperationException();
        }
      };
    }

    @Override
    public int size() {
      return names.length;
    }

    /** Returns the type of the ith dirent. */
    private Dirent.Type unpackType(int i) {
      int start = i * 2;
      boolean upper = packedTypes.get(start);
      boolean lower = packedTypes.get(start + 1);
      if (!upper && !lower) {
        return Type.FILE;
      } else if (!upper && lower){
        return Type.DIRECTORY;
      } else if (upper && !lower) {
        return Type.SYMLINK;
      } else {
        return Type.UNKNOWN;
      }
    }

    /** Sets the type of the ith dirent. */
    private static void packType(BitSet bitSet, Dirent.Type type, int i) {
      int start = i * 2;
      switch (type) {
        case FILE:
          pack(bitSet, start, false, false);
          break;
        case DIRECTORY:
          pack(bitSet, start, false, true);
          break;
        case SYMLINK:
          pack(bitSet, start, true, false);
          break;
        case UNKNOWN:
          pack(bitSet, start, true, true);
          break;
        default:
          throw new IllegalStateException("Unknown dirent type: " + type);
      }
    }

    private static void pack(BitSet bitSet, int start, boolean upper, boolean lower) {
      bitSet.set(start, upper);
      bitSet.set(start + 1, lower);
    }

    private Dirent direntAt(int i) {
      Preconditions.checkState(i >= 0 && i < size(), "i: %s, size: %s", i, size());
      return new Dirent(names[i], unpackType(i));
    }
  }
}
