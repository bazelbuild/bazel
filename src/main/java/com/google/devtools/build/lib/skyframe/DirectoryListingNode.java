// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Dirent.Type;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Objects;

/**
 * A node that represents the list of files in a given directory under a given package path root.
 *
 * <p>The node contains dirents corresponding to entries in the directory.
 *
 * <p>This node only depends on the FileNode corresponding to the directory. In particular, note
 * that it does not depend on any of its child entries.
 *
 * <p>Note that symlinks in dirents are <b>not</b> expanded. Dependents of the node are responsible
 * for expanding the symlink entries by referring to FileNodes that correspond to the symlinks.
 * This is a little onerous, but correct: we do not need to reread the directory when a symlink
 * inside it changes, therefore this node should not be invalidated in that case.
 */
@Immutable
@ThreadSafe
final class DirectoryListingNode implements Node {

  private final RootedPath rootedPath;
  private final CompactSortedDirents compactSortedDirents;

  /** Constructs a DirectoryListingNode with the given dirents. */
  public DirectoryListingNode(RootedPath directoryUnderRoot, Collection<Dirent> dirents) {
    this.rootedPath = Preconditions.checkNotNull(directoryUnderRoot);
    this.compactSortedDirents = CompactSortedDirents.create(dirents);
  }

  /** Constructs a DirectoryListingNode with the same dirents as the given node. */
  public DirectoryListingNode(RootedPath directoryUnderRoot, DirectoryListingNode other) {
    this.rootedPath = Preconditions.checkNotNull(directoryUnderRoot);
    this.compactSortedDirents = other.compactSortedDirents;
  }

  /**
   * Returns the path of the directory that this node represents.
   */
  public RootedPath getRootedPath() {
    return rootedPath;
  }

  /**
   * Returns the directory entries for this directory, in a stable order.
   *
   * <p>Symlinks are not expanded.
   */
  public Iterable<Dirent> getDirents() {
    return compactSortedDirents;
  }

  /**
   * Returns a {@link NodeKey} for getting the directory entries of the given directory. The
   * given path is assumed to be an existing directory (e.g. via {@link FileNode#isDirectory} or
   * from a directory listing on its parent directory).
   */
  @ThreadSafe
  static NodeKey key(RootedPath directoryUnderRoot) {
    return new NodeKey(NodeTypes.DIRECTORY_LISTING, directoryUnderRoot);
  }

  @ThreadSafe
  static DirectoryListingNode nodeForRootedPath(RootedPath directoryUnderRoot)
      throws IOException {
    Collection<Dirent> dirents = directoryUnderRoot.asPath().readdir(Symlinks.NOFOLLOW);
    return new DirectoryListingNode(directoryUnderRoot, dirents);
  }

  @Override
  public int hashCode() {
    return Objects.hash(rootedPath, compactSortedDirents);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof DirectoryListingNode)) {
      return false;
    }
    DirectoryListingNode other = (DirectoryListingNode) obj;
    if (!rootedPath.equals(other.rootedPath)) {
      return false;
    }
    if (!compactSortedDirents.equals(other.compactSortedDirents)) {
      return false;
    }
    return true;
  }

  /** A space-efficient, sorted, immutable dirent structure. */
  private static class CompactSortedDirents implements Iterable<Dirent>, Serializable {

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

    private int size() {
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
