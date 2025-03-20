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
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * Intermediate tree representation of a list of lexicographically sorted list of files. Each node
 * in the tree represents either a directory or file.
 */
final class DirectoryTree {

  interface Visitor {

    void visitDirectory(
        PathFragment dirname,
        SortedSet<FileNode> files,
        SortedSet<SymlinkNode> symlinks,
        SortedSet<DirectoryNode> dirs);
  }

  abstract static sealed class Node implements Comparable<Node> {
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
      if (o instanceof Node other) {
        return Objects.equals(pathSegment, other.pathSegment);
      }
      return false;
    }
  }

  static sealed class FileNode extends Node {
    private final Object /* Path | VirtualActionInput */ data;
    private final Digest digest;

    static FileNode create(String pathSegment, Path path, Digest digest) {
      return create(pathSegment, path, digest, /* toolInput= */ false);
    }

    static FileNode create(String pathSegment, Path path, Digest digest, boolean toolInput) {
      if (toolInput) {
        return new ToolInputFileNode(pathSegment, path, digest);
      } else {
        return new FileNode(pathSegment, path, digest);
      }
    }

    static FileNode create(
        String pathSegment,
        VirtualActionInput virtualActionInput,
        Digest digest,
        boolean toolInput) {
      if (toolInput) {
        return new ToolInputFileNode(pathSegment, virtualActionInput, digest);
      } else {
        return new FileNode(pathSegment, virtualActionInput, digest);
      }
    }

    private FileNode(
        String pathSegment, Object /* Path | VirtualActionInput */ data, Digest digest) {
      super(pathSegment);
      this.data = data;
      this.digest = digest;
    }

    Digest getDigest() {
      return digest;
    }

    Path getPath() {
      return data instanceof Path path ? path : null;
    }

    VirtualActionInput getVirtualActionInput() {
      return data instanceof VirtualActionInput virtualActionInput ? virtualActionInput : null;
    }

    boolean isToolInput() {
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(super.hashCode(), data, digest, isToolInput());
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof FileNode other) {
        return super.equals(other)
            && Objects.equals(data, other.data)
            && Objects.equals(digest, other.digest)
            && isToolInput() == other.isToolInput();
      }
      return false;
    }

    @Override
    public String toString() {
      return String.format(
          "%s (hash: %s, size: %d)", getPathSegment(), digest.getHash(), digest.getSizeBytes());
    }

    static final class ToolInputFileNode extends FileNode {
      ToolInputFileNode(
          String pathSegment, Object /* Path | VirtualActionInput */ data, Digest digest) {
        super(pathSegment, data, digest);
      }

      @Override
      boolean isToolInput() {
        return true;
      }
    }
  }

  static final class SymlinkNode extends Node {
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
      if (o instanceof SymlinkNode other) {
        return super.equals(other) && Objects.equals(target, other.target);
      }
      return false;
    }

    @Override
    public String toString() {
      return String.format("%s --> %s", getPathSegment(), getTarget());
    }
  }

  static final class DirectoryNode extends Node {

    private final SortedSet<FileNode> files = new TreeSet<>();
    private final SortedSet<SymlinkNode> symlinks = new TreeSet<>();
    private final SortedSet<DirectoryNode> subdirs = new TreeSet<>();

    DirectoryNode(String pathSegment) {
      super(pathSegment);
    }

    @CanIgnoreReturnValue
    boolean addChild(FileNode file) {
      return files.add(Preconditions.checkNotNull(file, "file"));
    }

    @CanIgnoreReturnValue
    boolean addChild(SymlinkNode symlink) {
      return symlinks.add(Preconditions.checkNotNull(symlink, "symlink"));
    }

    @CanIgnoreReturnValue
    boolean addChild(DirectoryNode subdir) {
      return subdirs.add(Preconditions.checkNotNull(subdir, "subdir"));
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          super.hashCode(), files.hashCode(), symlinks.hashCode(), subdirs.hashCode());
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof DirectoryNode other) {
        return super.equals(other)
            && Objects.equals(files, other.files)
            && Objects.equals(symlinks, other.symlinks)
            && Objects.equals(subdirs, other.subdirs);
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

    for (DirectoryNode subdir : dir.subdirs) {
      visit(visitor, dirname.getRelative(subdir.getPathSegment()));
    }
    visitor.visitDirectory(dirname, dir.files, dir.symlinks, dir.subdirs);
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
    if (!(o instanceof DirectoryTree other)) {
      return false;
    }
    return tree.equals(other.tree);
  }
}
