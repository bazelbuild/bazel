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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.FileStateNode.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A node that corresponds to a file (or directory or symlink or non-existent file), fully
 * accounting for symlinks (e.g. proper dependencies on ancestor symlinks so as to be incrementally
 * correct). Anything in Skyframe that cares about the fully resolved path of a file (e.g. anything
 * that cares about the contents of a file) should have a dependency on the corresponding
 * {@link FileNode}.
 *
 * <p>
 * Note that the existence of a file node does not imply that the file exists on the filesystem.
 * File nodes for missing files will be created on purpose in order to facilitate incremental builds
 * in the case those files have reappeared.
 *
 * <p>
 * This class contains the relevant metadata for a file, although not the contents. Note that
 * since a FileNode doesn't store its corresponding NodeKey, it's possible for the FileNodes for
 * two different paths to be the same.
 */
@Immutable
@ThreadSafe
abstract class FileNode implements Node {

  boolean exists() {
    return realFileStateNode().getType() != Type.NONEXISTENT;
  }

  boolean isSymlink() {
    return false;
  }

  /**
   * Returns true if this node corresponds to a file or symlink to an existing file. If so, its
   * parent directory is guaranteed to exist.
   */
  public boolean isFile() {
    return realFileStateNode().getType() == Type.FILE;
  }

  /**
   * Returns true if the file is a directory or a symlink to an existing directory. If so, its
   * parent directory is guaranteed to exist.
   */
  boolean isDirectory() {
    return realFileStateNode().getType() == Type.DIRECTORY;
  }

  /**
   * Returns the real rooted path of the file, taking ancestor symlinks into account. For example,
   * the rooted path ['root']/['a/b'] is really ['root']/['c/b'] if 'a' is a symlink to 'b'. Note
   * that ancestor symlinks outside the root boundary are not taken into consideration.
   */
  abstract RootedPath realRootedPath();

  abstract FileStateNode realFileStateNode();

  long getSize() {
    Preconditions.checkState(isFile(), this);
    return realFileStateNode().getSize();
  }

  @Nullable
  byte[] getDigest() {
    Preconditions.checkState(isFile(), this);
    return realFileStateNode().getDigest();
  }

  /**
   * Returns a key for building a file node for the given root-relative path.
   */
  @ThreadSafe
  static NodeKey key(RootedPath rootedPath) {
    return new NodeKey(NodeTypes.FILE, rootedPath);
  }

  @ThreadSafe
  static NodeKey key(Artifact artifact) {
    Path root = artifact.getRoot().getPath();
    return key(RootedPath.toRootedPath(root, artifact.getPath()));
  }

  /**
   * Only intended to be used by {@link FileNodeBuilder}. Should not be used for symlink cycles.
   */
  static FileNode node(RootedPath rootedPath, FileStateNode fileStateNode,
      RootedPath realRootedPath, FileStateNode realFileStateNode) {
    if (rootedPath.equals(realRootedPath)) {
      Preconditions.checkState(fileStateNode.getType() != FileStateNode.Type.SYMLINK,
          "rootedPath: %s, fileStateNode: %s, realRootedPath: %s, realFileStateNode: %s",
          rootedPath, fileStateNode, realRootedPath, realFileStateNode);
      return new RegularFileNode(rootedPath, fileStateNode);
    } else {
      if (fileStateNode.getType() == FileStateNode.Type.SYMLINK) {
        return new SymlinkFileNode(realRootedPath, realFileStateNode);
      } else {
        return new DifferentRealPathFileNode(realRootedPath, realFileStateNode);
      }
    }
  }

  /**
   * Implementation of {@link FileNode} for files whose fully resolved path is the same as the
   * requested path. For example, this is the case for the path "foo/bar/baz" if neither 'foo' nor
   * 'foo/bar' nor 'foo/bar/baz' are symlinks.
   */
  private static final class RegularFileNode extends FileNode {

    private final RootedPath rootedPath;
    private final FileStateNode fileStateNode;

    private RegularFileNode(RootedPath rootedPath, FileStateNode fileState) {
      this.rootedPath = rootedPath;
      this.fileStateNode = fileState;
    }

    @Override
    RootedPath realRootedPath() {
      return rootedPath;
    }

    @Override
    FileStateNode realFileStateNode() {
      return fileStateNode;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != RegularFileNode.class) {
        return false;
      }
      RegularFileNode other = (RegularFileNode) obj;
      return rootedPath.equals(other.rootedPath) && fileStateNode.equals(other.fileStateNode);
    }

    @Override
    public int hashCode() {
      return Objects.hash(rootedPath, fileStateNode);
    }

    @Override
    public String toString() {
      return rootedPath + "with state " + fileStateNode
          + " (neither this file nor any ancestor is a symlink)";
    }
  }

  /**
   * Base class for {@link FileNode}s for files whose fully resolved path is different than the
   * requested path. For example, this is the case for the path "foo/bar/baz" if at least one of
   * 'foo', 'foo/bar', or 'foo/bar/baz' is a symlink.
   */
  private static class DifferentRealPathFileNode extends FileNode {

    protected final RootedPath realRootedPath;
    protected final FileStateNode realFileStateNode;

    private DifferentRealPathFileNode(RootedPath realRootedPath, FileStateNode realFileStateNode) {
      this.realRootedPath = realRootedPath;
      this.realFileStateNode = realFileStateNode;
    }

    @Override
    RootedPath realRootedPath() {
      return realRootedPath;
    }

    @Override
    FileStateNode realFileStateNode() {
      return realFileStateNode;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != DifferentRealPathFileNode.class) {
        return false;
      }
      DifferentRealPathFileNode other = (DifferentRealPathFileNode) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateNode.equals(other.realFileStateNode);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realRootedPath, realFileStateNode);
    }

    @Override
    public String toString() {
      return realRootedPath + "with state " + realFileStateNode
          + " (at least one ancestory is a symlink)";
    }
  }

  /** Implementation of {@link FileNode} for files that are symlinks. */
  private static final class SymlinkFileNode extends DifferentRealPathFileNode {

    private SymlinkFileNode(RootedPath realRootedPath, FileStateNode realFileState) {
      super(realRootedPath, realFileState);
    }

    @Override
    boolean isSymlink() {
      return true;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (obj.getClass() != SymlinkFileNode.class) {
        return false;
      }
      SymlinkFileNode other = (SymlinkFileNode) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateNode.equals(other.realFileStateNode);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realRootedPath, realFileStateNode, Boolean.TRUE);
    }

    @Override
    public String toString() {
      return "symlink to " + realRootedPath + "with state " + realFileStateNode;
    }
  }
}
