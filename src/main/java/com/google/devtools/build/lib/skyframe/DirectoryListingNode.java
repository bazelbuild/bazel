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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

import java.util.Objects;

/**
 * A node that represents the list of files in a given directory under a given package path root.
 * Anything in Skyframe that cares about the contents of a directory should have a dependency
 * on the corresponding {@link DirectoryListingNode}.
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
abstract class DirectoryListingNode implements Node {

  /**
   * Returns the directory entries for this directory, in a stable order.
   *
   * <p>Symlinks are not expanded.
   */
  public abstract Iterable<Dirent> getDirents();

  /**
   * Returns a {@link NodeKey} for getting the directory entries of the given directory. The
   * given path is assumed to be an existing directory (e.g. via {@link FileNode#isDirectory} or
   * from a directory listing on its parent directory).
   */
  @ThreadSafe
  static NodeKey key(RootedPath directoryUnderRoot) {
    return new NodeKey(NodeTypes.DIRECTORY_LISTING, directoryUnderRoot);
  }

  static DirectoryListingNode node(RootedPath dirRootedPath, FileNode dirFileNode,
      DirectoryListingStateNode realDirectoryListingStateNode) {
    return dirFileNode.realRootedPath().equals(dirRootedPath)
        ? new RegularDirectoryListingNode(realDirectoryListingStateNode)
        : new DifferentRealPathDirectoryListingNode(dirFileNode.realRootedPath(),
            realDirectoryListingStateNode);
  }

  @ThreadSafe
  private static final class RegularDirectoryListingNode extends DirectoryListingNode {

    private final DirectoryListingStateNode directoryListingStateNode;

    private RegularDirectoryListingNode(DirectoryListingStateNode directoryListingStateNode) {
      this.directoryListingStateNode = directoryListingStateNode;
    }

    @Override
    public Iterable<Dirent> getDirents() {
      return directoryListingStateNode.getDirents();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof RegularDirectoryListingNode)) {
        return false;
      }
      RegularDirectoryListingNode other = (RegularDirectoryListingNode) obj;
      return directoryListingStateNode.equals(other.directoryListingStateNode);
    }

    @Override
    public int hashCode() {
      return directoryListingStateNode.hashCode();
    }
  }

  @ThreadSafe
  private static final class DifferentRealPathDirectoryListingNode extends DirectoryListingNode {

    private final RootedPath realDirRootedPath;
    private final DirectoryListingStateNode directoryListingStateNode;

    private DifferentRealPathDirectoryListingNode(RootedPath realDirRootedPath,
        DirectoryListingStateNode directoryListingStateNode) {
      this.realDirRootedPath = realDirRootedPath;
      this.directoryListingStateNode = directoryListingStateNode;
    }

    @Override
    public Iterable<Dirent> getDirents() {
      return directoryListingStateNode.getDirents();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof DifferentRealPathDirectoryListingNode)) {
        return false;
      }
      DifferentRealPathDirectoryListingNode other = (DifferentRealPathDirectoryListingNode) obj;
      return realDirRootedPath.equals(other.realDirRootedPath)
          && directoryListingStateNode.equals(other.directoryListingStateNode);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realDirRootedPath, directoryListingStateNode);
    }
  }
}
