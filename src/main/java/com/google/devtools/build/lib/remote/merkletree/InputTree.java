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
import com.google.common.base.Strings;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Intermediate tree representation of a list of lexicographically sorted list of files. Each node
 * in the tree represents either a directory or file.
 */
class InputTree {

  interface Visitor {
    void visitDirectory(PathFragment dirname, List<FileNode> files, List<DirectoryNode> dirs);
  }

  abstract static class Node {
    private final String pathSegment;

    Node(String pathSegment) {
      this.pathSegment = Preconditions.checkNotNull(pathSegment);
    }

    String getPathSegment() {
      return pathSegment;
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
    private final ActionInput input;
    private final Digest digest;

    FileNode(String pathSegment, ActionInput input, Digest digest) {
      super(pathSegment);
      this.input = Preconditions.checkNotNull(input, "input");
      this.digest = Preconditions.checkNotNull(digest, "digest");
    }

    Digest getDigest() {
      return digest;
    }

    ActionInput getActionInput() {
      return input;
    }

    @Override
    public int hashCode() {
      return Objects.hash(super.hashCode(), input, digest);
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof FileNode) {
        FileNode other = (FileNode) o;
        return super.equals(other)
            && Objects.equals(input, other.input)
            && Objects.equals(digest, other.digest);
      }
      return false;
    }
  }

  static class DirectoryNode extends Node {
    private final List<Node> children = new ArrayList<>();

    DirectoryNode(String pathSegment) {
      super(pathSegment);
    }

    void addChild(Node child) {
      children.add(Preconditions.checkNotNull(child, "child"));
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

  private InputTree(Map<PathFragment, DirectoryNode> tree, int numFiles) {
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
   * Traverses the {@link InputTree} in a depth first search manner. The children are visited in
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
    List<DirectoryNode> dirs = new ArrayList<>();
    for (Node child : dir.children) {
      if (child instanceof FileNode) {
        files.add((FileNode) child);
      } else if (child instanceof DirectoryNode) {
        dirs.add((DirectoryNode) child);
        visit(visitor, dirname.getRelative(child.pathSegment));
      } else {
        throw new IllegalStateException(
            String.format("Node type '%s' is not supported", child.getClass().getSimpleName()));
      }
    }
    visitor.visitDirectory(dirname, files, dirs);
  }

  @Override
  public String toString() {
    Map<PathFragment, StringBuilder> m = new HashMap<>();
    visit(
        (dirname, files, dirs) -> {
          int depth = dirname.segmentCount() - 1;
          StringBuilder sb = new StringBuilder();

          if (!dirname.equals(PathFragment.EMPTY_FRAGMENT)) {
            sb.append(Strings.repeat("  ", depth));
            sb.append(dirname.getBaseName());
            sb.append("\n");
          }
          if (!files.isEmpty()) {
            for (FileNode file : files) {
              sb.append(Strings.repeat("  ", depth + 1));
              sb.append(formatFile(file));
              sb.append("\n");
            }
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
    if (!(o instanceof InputTree)) {
      return false;
    }
    InputTree other = (InputTree) o;
    return tree.equals(other.tree);
  }

  private static String formatFile(FileNode file) {
    return String.format(
        "%s (hash: %s, size: %d)",
        file.getPathSegment(), file.digest.getHash(), file.digest.getSizeBytes());
  }

  static InputTree build(
      SortedMap<PathFragment, ActionInput> inputs,
      MetadataProvider metadataProvider,
      Path execRoot,
      DigestUtil digestUtil)
      throws IOException {
    Map<PathFragment, DirectoryNode> tree = new HashMap<>();
    int numFiles = build(inputs, metadataProvider, execRoot, digestUtil, tree);
    return new InputTree(tree, numFiles);
  }

  private static int build(
      SortedMap<PathFragment, ActionInput> inputs,
      MetadataProvider metadataProvider,
      Path execRoot,
      DigestUtil digestUtil,
      Map<PathFragment, DirectoryNode> tree)
      throws IOException {
    if (inputs.isEmpty()) {
      return 0;
    }

    PathFragment dirname = null;
    DirectoryNode dir = null;
    int numFiles = inputs.size();
    for (Map.Entry<PathFragment, ActionInput> e : inputs.entrySet()) {
      PathFragment path = e.getKey();
      ActionInput input = e.getValue();
      if (dirname == null || !path.getParentDirectory().equals(dirname)) {
        dirname = path.getParentDirectory();
        dir = tree.get(dirname);
        if (dir == null) {
          dir = new DirectoryNode(dirname.getBaseName());
          tree.put(dirname, dir);
          createParentDirectoriesIfNotExist(dirname, dir, tree);
        }
      }

      if (input instanceof VirtualActionInput) {
        Digest d = digestUtil.compute((VirtualActionInput) input);
        dir.addChild(new FileNode(path.getBaseName(), input, d));
        continue;
      }

      FileArtifactValue metadata =
          Preconditions.checkNotNull(
              metadataProvider.getMetadata(input),
              "missing metadata for '%s'",
              input.getExecPathString());
      switch (metadata.getType()) {
        case REGULAR_FILE:
          Digest d = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
          dir.addChild(new FileNode(path.getBaseName(), input, d));
          break;

        case DIRECTORY:
          SortedMap<PathFragment, ActionInput> directoryInputs = explodeDirectory(path, execRoot);
          numFiles += build(directoryInputs, metadataProvider, execRoot, digestUtil, tree);
          break;

        case SYMLINK:
          throw new IllegalStateException(
              String.format(
                  "Encountered symlink input '%s', but all"
                      + " symlinks should have been resolved by SkyFrame. This is a bug.",
                  path));

        case SPECIAL_FILE:
          throw new IOException(
              String.format(
                  "The '%s' is a special input which is not supported"
                      + " by remote caching and execution.",
                  path));

        case NONEXISTENT:
          throw new IOException(String.format("The file type of '%s' is not supported.", path));
      }
    }
    return numFiles;
  }

  private static SortedMap<PathFragment, ActionInput> explodeDirectory(
      PathFragment dirname, Path execRoot) throws IOException {
    SortedMap<PathFragment, ActionInput> inputs = new TreeMap<>();
    explodeDirectory(dirname, inputs, execRoot);
    return inputs;
  }

  private static void explodeDirectory(
      PathFragment dirname, SortedMap<PathFragment, ActionInput> inputs, Path execRoot)
      throws IOException {
    Collection<Dirent> entries = execRoot.getRelative(dirname).readdir(Symlinks.FOLLOW);
    for (Dirent entry : entries) {
      String basename = entry.getName();
      PathFragment path = dirname.getChild(basename);
      switch (entry.getType()) {
        case FILE:
          inputs.put(path, ActionInputHelper.fromPath(path));
          break;

        case DIRECTORY:
          explodeDirectory(path, inputs, execRoot);
          break;

        case SYMLINK:
          throw new IllegalStateException(
              String.format(
                  "Encountered symlink input '%s', but all"
                      + " symlinks should have been resolved by readdir. This is a bug.",
                  path));

        case UNKNOWN:
          throw new IOException(String.format("The file type of '%s' is not supported.", path));
      }
    }
  }

  private static void createParentDirectoriesIfNotExist(
      PathFragment dirname, DirectoryNode dir, Map<PathFragment, DirectoryNode> tree) {
    PathFragment parentDirname = dirname.getParentDirectory();
    DirectoryNode prevDir = dir;
    while (parentDirname != null) {
      DirectoryNode parentDir = tree.get(parentDirname);
      if (parentDir != null) {
        parentDir.addChild(prevDir);
        break;
      }

      parentDir = new DirectoryNode(parentDirname.getBaseName());
      parentDir.addChild(prevDir);
      tree.put(parentDirname, parentDir);

      parentDirname = parentDirname.getParentDirectory();
      prevDir = parentDir;
    }
  }
}
