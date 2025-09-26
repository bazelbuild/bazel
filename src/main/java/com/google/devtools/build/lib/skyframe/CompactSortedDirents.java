// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.Dirent;
import java.util.AbstractCollection;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Objects;
import javax.annotation.Nullable;

/** A space-efficient, sorted, immutable dirent structure. */
final class CompactSortedDirents extends AbstractCollection<Dirent> implements Dirents {

  private final String[] names;
  private final BitSet packedTypes;

  private CompactSortedDirents(String[] names, BitSet packedTypes) {
    this.names = names;
    this.packedTypes = packedTypes;
  }

  static CompactSortedDirents create(Collection<Dirent> dirents) {
    final Dirent[] direntArray = dirents.toArray(Dirent[]::new);
    Integer[] indices = new Integer[dirents.size()];
    for (int i = 0; i < dirents.size(); i++) {
      indices[i] = i;
    }
    Arrays.sort(indices, Comparator.comparing(o -> direntArray[o]));
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
    if (!(obj instanceof CompactSortedDirents other)) {
      return false;
    }
    if (this == obj) {
      return true;
    }
    return Arrays.equals(names, other.names) && packedTypes.equals(other.packedTypes);
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
    return new Iterator<>() {

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
      return Dirent.Type.FILE;
    } else if (!upper && lower) {
      return Dirent.Type.DIRECTORY;
    } else if (upper && !lower) {
      return Dirent.Type.SYMLINK;
    } else {
      return Dirent.Type.UNKNOWN;
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
