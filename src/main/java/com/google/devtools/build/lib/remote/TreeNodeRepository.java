// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.common.collect.Iterables;
import com.google.common.collect.TreeTraverser;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.RemoteProtocol.FileNode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A factory and repository for {@link TreeNode} objects. Provides directory structure traversals,
 * computing and caching Merkle hashes on all objects.
 */
@ThreadSafe
public final class TreeNodeRepository extends TreeTraverser<TreeNodeRepository.TreeNode> {
  /**
   * A single node in a hierarchical directory structure. Leaves are the Artifacts, although we only
   * use the ActionInput interface. We assume that the objects used for the ActionInputs are unique
   * (same data corresponds to a canonical object in memory).
   */
  @Immutable
  @ThreadSafe
  public static final class TreeNode {

    private final int hashCode;
    private final ImmutableList<ChildEntry> childEntries; // no need to make it a map thus far.
    @Nullable private final ActionInput actionInput; // Null iff this is a directory.

    /** A pair of path segment, TreeNode. */
    @Immutable
    public static final class ChildEntry {
      private final String segment;
      private final TreeNode child;

      public ChildEntry(String segment, TreeNode child) {
        this.segment = segment;
        this.child = child;
      }

      public TreeNode getChild() {
        return child;
      }

      public String getSegment() {
        return segment;
      }

      @Override
      @SuppressWarnings("ReferenceEquality")
      public boolean equals(Object o) {
        if (o == this) {
          return true;
        }
        if (!(o instanceof ChildEntry)) {
          return false;
        }
        ChildEntry other = (ChildEntry) o;
        // Pointer comparisons only, because both the Path segments and the TreeNodes are interned.
        return other.segment == segment && other.child == child;
      }

      @Override
      public int hashCode() {
        return Objects.hash(segment, child);
      }
    }

    // Should only be called by the TreeNodeRepository.
    private TreeNode(Iterable<ChildEntry> childEntries) {
      this.actionInput = null;
      this.childEntries = ImmutableList.copyOf(childEntries);
      hashCode = Arrays.hashCode(this.childEntries.toArray());
    }

    // Should only be called by the TreeNodeRepository.
    private TreeNode(ActionInput actionInput) {
      this.actionInput = actionInput;
      this.childEntries = ImmutableList.of();
      hashCode = actionInput.hashCode(); // This will ensure efficient interning of TreeNodes as
      // long as all ActionInputs either implement data-based hashCode or are interned themselves.
    }

    public ActionInput getActionInput() {
      return actionInput;
    }

    public ImmutableList<ChildEntry> getChildEntries() {
      return childEntries;
    }

    public boolean isLeaf() {
      return actionInput != null;
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof TreeNode)) {
        return false;
      }
      TreeNode otherNode = (TreeNode) o;
      // Full comparison of ActionInputs. If pointers are different, will compare paths.
      return Objects.equals(otherNode.actionInput, actionInput)
          && childEntries.equals(otherNode.childEntries);
    }

    private String toDebugStringAtLevel(int level) {
      char[] prefix = new char[level];
      Arrays.fill(prefix, ' ');
      StringBuilder sb = new StringBuilder();

      if (isLeaf()) {
        sb.append('\n');
        sb.append(prefix);
        sb.append("leaf: ");
        sb.append(actionInput);
      } else {
        for (ChildEntry entry : childEntries) {
          sb.append('\n');
          sb.append(prefix);
          sb.append(entry.segment);
          sb.append(entry.child.toDebugStringAtLevel(level + 1));
        }
      }
      return sb.toString();
    }

    public String toDebugString() {
      return toDebugStringAtLevel(0);
    }
  }

  // Keep only one canonical instance of every TreeNode in the repository.
  private final Interner<TreeNode> interner = Interners.newWeakInterner();
  // Merkle hashes are computed and cached by the repository, therefore execRoot must
  // be part of the state.
  private final Path execRoot;
  private final Map<ActionInput, ContentDigest> fileContentsDigestCache = new HashMap<>();
  private final Map<ContentDigest, ActionInput> digestFileContentsCache = new HashMap<>();
  private final Map<TreeNode, ContentDigest> treeNodeDigestCache = new HashMap<>();
  private final Map<ContentDigest, TreeNode> digestTreeNodeCache = new HashMap<>();
  private final Map<TreeNode, FileNode> fileNodeCache = new HashMap<>();

  public TreeNodeRepository(Path execRoot) {
    this.execRoot = execRoot;
  }

  @Override
  public Iterable<TreeNode> children(TreeNode node) {
    return Iterables.transform(
        node.getChildEntries(),
        new Function<TreeNode.ChildEntry, TreeNode>() {
          @Override
          public TreeNode apply(TreeNode.ChildEntry entry) {
            return entry.getChild();
          }
        });
  }

  /** Traverse the directory structure in order (pre-order tree traversal). */
  public Iterable<TreeNode> descendants(TreeNode node) {
    return preOrderTraversal(node);
  }

  /**
   * Traverse the directory structure in order (pre-order tree traversal), return only the leaves.
   */
  public Iterable<TreeNode> leaves(TreeNode node) {
    return Iterables.filter(
        descendants(node),
        new Predicate<TreeNode>() {
          @Override
          public boolean apply(TreeNode node) {
            return node.isLeaf();
          }
        });
  }

  /**
   * This function is a temporary and highly inefficient hack! It builds the tree from a ready list
   * of input files. TODO(olaola): switch to creating and maintaining the TreeNodeRepository based
   * on the build graph structure.
   */
  public TreeNode buildFromActionInputs(Iterable<ActionInput> actionInputs) {
    TreeMap<PathFragment, ActionInput> sortedMap = new TreeMap<>();
    for (ActionInput input : actionInputs) {
      sortedMap.put(new PathFragment(input.getExecPathString()), input);
    }
    ImmutableList.Builder<ImmutableList<String>> segments = ImmutableList.builder();
    for (PathFragment path : sortedMap.keySet()) {
      segments.add(path.getSegments());
    }
    ImmutableList<ActionInput> inputs = ImmutableList.copyOf(sortedMap.values());
    return buildParentNode(inputs, segments.build(), 0, inputs.size(), 0);
  }

  @SuppressWarnings("ReferenceEquality") // Segments are interned.
  private TreeNode buildParentNode(
      ImmutableList<ActionInput> inputs,
      ImmutableList<ImmutableList<String>> segments,
      int inputsStart,
      int inputsEnd,
      int segmentIndex) {
    if (segmentIndex == segments.get(inputsStart).size()) {
      // Leaf node reached. Must be unique.
      Preconditions.checkArgument(
          inputsStart == inputsEnd - 1, "Encountered two inputs with the same path.");
      // TODO: check that the actionInput is a single file!
      return interner.intern(new TreeNode(inputs.get(inputsStart)));
    }
    ArrayList<TreeNode.ChildEntry> entries = new ArrayList<>();
    String segment = segments.get(inputsStart).get(segmentIndex);
    for (int inputIndex = inputsStart; inputIndex < inputsEnd; ++inputIndex) {
      if (inputIndex + 1 == inputsEnd
          || segment != segments.get(inputIndex + 1).get(segmentIndex)) {
        entries.add(
            new TreeNode.ChildEntry(
                segment,
                buildParentNode(inputs, segments, inputsStart, inputIndex + 1, segmentIndex + 1)));
        if (inputIndex + 1 < inputsEnd) {
          inputsStart = inputIndex + 1;
          segment = segments.get(inputsStart).get(segmentIndex);
        }
      }
    }
    return interner.intern(new TreeNode(entries));
  }

  private synchronized ContentDigest getOrComputeActionInputDigest(ActionInput actionInput)
      throws IOException {
    ContentDigest digest = fileContentsDigestCache.get(actionInput);
    if (digest == null) {
      digest = ContentDigests.computeDigest(execRoot.getRelative(actionInput.getExecPathString()));
      fileContentsDigestCache.put(actionInput, digest);
      digestFileContentsCache.put(digest, actionInput);
    }
    return digest;
  }

  private synchronized FileNode getOrComputeFileNode(TreeNode node) throws IOException {
    // Assumes all child digests have already been computed!
    FileNode fileNode = fileNodeCache.get(node);
    if (fileNode == null) {
      FileNode.Builder b = FileNode.newBuilder();
      if (node.isLeaf()) {
        ContentDigest fileDigest = fileContentsDigestCache.get(node.getActionInput());
        Preconditions.checkState(fileDigest != null);
        b.getFileMetadataBuilder()
            .setDigest(fileDigest)
            .setExecutable(
                execRoot.getRelative(node.getActionInput().getExecPathString()).isExecutable());
      } else {
        for (TreeNode.ChildEntry entry : node.getChildEntries()) {
          ContentDigest childDigest = treeNodeDigestCache.get(entry.getChild());
          Preconditions.checkState(childDigest != null);
          b.addChildBuilder().setPath(entry.getSegment()).setDigest(childDigest);
        }
      }
      fileNode = b.build();
      fileNodeCache.put(node, fileNode);
      ContentDigest digest = ContentDigests.computeDigest(fileNode);
      treeNodeDigestCache.put(node, digest);
      digestTreeNodeCache.put(digest, node);
    }
    return fileNode;
  }

  // Recursively traverses the tree, expanding and computing Merkle digests for nodes for which
  // they have not yet been computed and cached.
  public void computeMerkleDigests(TreeNode root) throws IOException {
    synchronized (this) {
      if (fileNodeCache.get(root) != null) {
        // Strong assumption: the cache is valid, i.e. parent present implies children present.
        return;
      }
    }
    if (root.isLeaf()) {
      getOrComputeActionInputDigest(root.getActionInput());
    } else {
      for (TreeNode child : children(root)) {
        computeMerkleDigests(child);
      }
    }
    getOrComputeFileNode(root);
  }

  /**
   * Should only be used after computeMerkleDigests has been called on one of the node ancestors.
   * Returns the precomputed digest.
   */
  public ContentDigest getMerkleDigest(TreeNode node) {
    return treeNodeDigestCache.get(node);
  }

  /**
   * Returns the precomputed digests for both data and metadata. Should only be used after
   * computeMerkleDigests has been called on one of the node ancestors.
   */
  public ImmutableCollection<ContentDigest> getAllDigests(TreeNode root) {
    ImmutableSet.Builder<ContentDigest> digests = ImmutableSet.builder();
    for (TreeNode node : descendants(root)) {
      digests.add(Preconditions.checkNotNull(treeNodeDigestCache.get(node)));
      if (node.isLeaf()) {
        digests.add(Preconditions.checkNotNull(fileContentsDigestCache.get(node.getActionInput())));
      }
    }
    return digests.build();
  }

  /**
   * Serializes all of the subtree to the file node list. TODO(olaola): add a version that only
   * copies a part of the tree that we are interested in. Should only be used after
   * computeMerkleDigests has been called on one of the node ancestors.
   */
  // Note: this is not, strictly speaking, thread safe. If someone is deleting cached Merkle hashes
  // while this is executing, it will trigger an exception. But I think this is WAI.
  public ImmutableList<FileNode> treeToFileNodes(TreeNode root) throws IOException {
    ImmutableList.Builder<FileNode> fileNodes = ImmutableList.builder();
    for (TreeNode node : descendants(root)) {
      fileNodes.add(Preconditions.checkNotNull(fileNodeCache.get(node)));
    }
    return fileNodes.build();
  }

  /**
   * Should only be used on digests created by a call to computeMerkleDigests. Looks up ActionInputs
   * or FileNodes by cached digests and adds them to the lists.
   */
  public void getDataFromDigests(
      Iterable<ContentDigest> digests, List<ActionInput> actionInputs, List<FileNode> nodes) {
    for (ContentDigest digest : digests) {
      TreeNode treeNode = digestTreeNodeCache.get(digest);
      if (treeNode != null) {
        nodes.add(Preconditions.checkNotNull(fileNodeCache.get(treeNode)));
      } else { // If not there, it must be an ActionInput.
        actionInputs.add(Preconditions.checkNotNull(digestFileContentsCache.get(digest)));
      }
    }
  }
}
