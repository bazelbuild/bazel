// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.merkletree;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedSet;

/**
 * Intermediate tree representation of a list of lexicographically sorted list of files. Each node
 * in the tree represents either a directory or file.
 */
final class DirectoryTree {

  interface Visitor {
    void visitDirectory(PathFragment dirname, List<FileNode> files, List<SymlinkNode> symlinks,
        List<DirectoryNode> dirs);
  }

  abstract static class Node implements Comparable<Node> {
    private final String pathSegment;

    Node(String pathSegment) {
      this.pathSegment = Preconditions.checkNotNull(pathSegment);
    }

    String getPathSegment() {
      return pathSegment;
    }

    @Override
    public int compareTo(Node other) {
      return pathSegment.compareTo(other.pathSegment);
    }

    @Override
    public int hashCode() {
      return pathSegment.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof Node) {
        Node other = (Node) o;
        return Objects.equals(pathSegment, other.pathSegment);
      }
      return false;
    }
  }

  static class FileNode extends Node {
    private final Path path;
    private final ByteString data;
    private final Digest digest;
    private final boolean isExecutable;

    /**
     * Create a FileNode with its executable bit set.
     *
     * <p>We always treat files as executable since Bazel will `chmod 555` on the output files of an
     * action within ActionMetadataHandler#getMetadata after action execution if no metadata was
     * injected. We can't use real executable bit of the file until this behaviour is changed. See
     * https://github.com/bazelbuild/bazel/issues/13262 for more details.
     */
    static FileNode createExecutable(String pathSegment, Path path, Digest digest) {
      return new FileNode(pathSegment, path, digest, /* isExecutable= */ true);
    }

    static FileNode createExecutable(String pathSegment, ByteString data, Digest digest) {
      return new FileNode(pathSegment, data, digest, /* isExecutable= */ true);
    }

    private FileNode(String pathSegment, Path path, Digest digest, boolean isExecutable) {
      super(pathSegment);
      this.path = Preconditions.checkNotNull(path, "path");
      this.data = null;
      this.digest = Preconditions.checkNotNull(digest, "digest");
      this.isExecutable = isExecutable;
    }

    private FileNode(String pathSegment, ByteString data, Digest digest, boolean isExecutable) {
      super(pathSegment);
      this.path = null;
      this.data = Preconditions.checkNotNull(data, "data");
      this.digest = Preconditions.checkNotNull(digest, "digest");
      this.isExecutable = isExecutable;
    }

    Digest getDigest() {
      return digest;
    }

    Path getPath() {
      return path;
    }

    ByteString getBytes() {
      return data;
    }

    public boolean isExecutable() {
      return isExecutable;
    }

    @Override
    public int hashCode() {
      return Objects.hash(super.hashCode(), path, data, digest, isExecutable);
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof FileNode) {
        FileNode other = (FileNode) o;
        return super.equals(other)
            && Objects.equals(path, other.path)
            && Objects.equals(data, other.data)
            && Objects.equals(digest, other.digest)
            && isExecutable == other.isExecutable;
      }
      return false;
    }

    @Override
    public String toString() {
      return String.format("%s (hash: %s, size: %d)",
          getPathSegment(), digest.getHash(), digest.getSizeBytes());
    }
  }

  static class SymlinkNode extends Node {
    private final String target;

    SymlinkNode(String pathSegment, String target) {
      super(pathSegment);
      this.target = target;
    }

    public String getTarget() {
      return target;
    }

    @Override
    public int hashCode() {
      return Objects.hash(super.hashCode(), target);
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof SymlinkNode) {
        SymlinkNode other = (SymlinkNode) o;
        return super.equals(other) && Objects.equals(target, other.target);
      }
      return false;
    }

    @Override
    public String toString() {
      return String.format("%s --> %s", getPathSegment(), getTarget());
    }
  }

  static class DirectoryNode extends Node {
    private final SortedSet<Node> children = Sets.newTreeSet();

    DirectoryNode(String pathSegment) {
      super(pathSegment);
    }

    boolean addChild(Node child) {
      return children.add(Preconditions.checkNotNull(child, "child"));
    }

    @Override
    public int hashCode() {
      return Objects.hash(super.hashCode(), children.hashCode());
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof DirectoryNode) {
        DirectoryNode other = (DirectoryNode) o;
        return super.equals(other) && Objects.equals(children, other.children);
      }
      return false;
    }
  }

  private final Map<PathFragment, DirectoryNode> tree;
  private final int numFiles;

  DirectoryTree(Map<PathFragment, DirectoryNode> tree, int numFiles) {
    Preconditions.checkState(numFiles >= 0, "numFiles must gte 0");
    this.tree = Preconditions.checkNotNull(tree, "tree");
    this.numFiles = numFiles;
  }

  int numDirectories() {
    return tree.size();
  }

  int numFiles() {
    return numFiles;
  }

  boolean isEmpty() {
    return tree.isEmpty();
  }

  /**
   * Traverses the {@link DirectoryTree} in a depth first search manner. The children are visited in
   * lexographical order.
   */
  void visit(Visitor visitor) {
    Preconditions.checkNotNull(visitor, "visitor");
    visit(visitor, PathFragment.EMPTY_FRAGMENT);
  }

  private void visit(Visitor visitor, PathFragment dirname) {
    DirectoryNode dir = tree.get(dirname);
    if (dir == null) {
      return;
    }

    List<FileNode> files = new ArrayList<>(dir.children.size());
    List<SymlinkNode> symlinks = new ArrayList<>();
    List<DirectoryNode> dirs = new ArrayList<>();
    for (Node child : dir.children) {
      if (child instanceof FileNode) {
        files.add((FileNode) child);
      } else if (child instanceof SymlinkNode) {
        symlinks.add((SymlinkNode) child);
      } else if (child instanceof DirectoryNode) {
        dirs.add((DirectoryNode) child);
        visit(visitor, dirname.getRelative(child.pathSegment));
      } else {
        throw new IllegalStateException(
            String.format("Node type '%s' is not supported", child.getClass().getSimpleName()));
      }
    }
    visitor.visitDirectory(dirname, files, symlinks, dirs);
  }

  @Override
  public String toString() {
    Map<PathFragment, StringBuilder> m = new HashMap<>();
    visit(
        (dirname, files, symlinks, dirs) -> {
          int depth = dirname.segmentCount() - 1;
          StringBuilder sb = new StringBuilder();

          if (!dirname.equals(PathFragment.EMPTY_FRAGMENT)) {
            sb.append(" ".repeat(2 * depth));
            sb.append(dirname.getBaseName());
            sb.append("\n");
          }
          for (Node fileOrSymlink : Iterables.concat(files, symlinks)) {
            sb.append(" ".repeat(2 * (depth + 1)));
            sb.append(fileOrSymlink);
            sb.append("\n");
          }
          if (!dirs.isEmpty()) {
            for (DirectoryNode dir : dirs) {
              sb.append(m.remove(dirname.getRelative(dir.getPathSegment())));
            }
          }
          m.put(dirname, sb);
        });
    return m.get(PathFragment.EMPTY_FRAGMENT).toString();
  }

  @Override
  public int hashCode() {
    return tree.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof DirectoryTree)) {
      return false;
    }
    DirectoryTree other = (DirectoryTree) o;
    return tree.equals(other.tree);
  }
}
