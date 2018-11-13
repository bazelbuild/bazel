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

import static java.nio.charset.StandardCharsets.US_ASCII;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.graph.Traverser;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.DigestOfDirectoryException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedMap;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * A factory and repository for {@link TreeNode} objects. Provides directory structure traversals,
 * computing and caching Merkle hashes on all objects.
 */
@ThreadSafe
public final class TreeNodeRepository {
  private static final BaseEncoding LOWER_CASE_HEX = BaseEncoding.base16().lowerCase();

  private final Traverser<TreeNode> traverser =
      Traverser.forTree((TreeNode node) -> children(node));

  /**
   * A single node in a hierarchical directory structure. Leaves are the Artifacts, although we only
   * use the ActionInput interface. We assume that the objects used for the ActionInputs are unique
   * (same data corresponds to a canonical object in memory).
   *
   * <p>There are three cases:
   *
   * <ol>
   *   <li>The node is a leaf that represents an artifact file.
   *   <li>The node is a directory optionally associated with an artifact (an "artifact directory").
   *   <li>The node is a leaf that is the descendant of an artifact directory. In this case, the
   *       node is associated with a BasicActionInput, not a full Artifact.
   * </ol>
   */
  @Immutable
  @ThreadSafe
  public static final class TreeNode {

    private final int hashCode;
    private final ImmutableList<ChildEntry> childEntries; // no need to make it a map thus far.
    @Nullable private final ActionInput actionInput;
    private final boolean isLeaf;

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
        // Pointer comparison for the TreeNode as it is interned
        return other.segment.equals(segment) && other.child == child;
      }

      @Override
      public int hashCode() {
        return Objects.hash(segment, child);
      }

      public static Comparator<ChildEntry> segmentOrder =
          Comparator.comparing(ChildEntry::getSegment);
    }

    // Should only be called by the TreeNodeRepository.
    private TreeNode(Iterable<ChildEntry> childEntries, @Nullable ActionInput actionInput) {
      isLeaf = false;
      this.actionInput = actionInput;
      this.childEntries = ImmutableList.copyOf(childEntries);
      if (actionInput != null) {
        hashCode = actionInput.hashCode(); // This will ensure efficient interning of TreeNodes as
        // long as all ActionInputs either implement data-based hashCode or are interned themselves.
      } else {
        hashCode = Arrays.hashCode(this.childEntries.toArray());
      }
    }

    // Should only be called by the TreeNodeRepository.
    private TreeNode(ActionInput actionInput) {
      isLeaf = true;
      this.actionInput =
          Preconditions.checkNotNull(actionInput, "a TreeNode leaf should have an ActionInput");
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
      return isLeaf;
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

  private static final TreeNode EMPTY_NODE =
      new TreeNode(ImmutableList.<TreeNode.ChildEntry>of(), null);

  // Keep only one canonical instance of every TreeNode in the repository.
  private final Interner<TreeNode> interner = BlazeInterners.newWeakInterner();
  // Merkle hashes are computed and cached by the repository, therefore execRoot must
  // be part of the state.
  private final Path execRoot;
  private final MetadataProvider inputFileCache;
  private final Map<ByteString, ActionInput> reverseInputMap = new ConcurrentHashMap<>();
  // For directories that are themselves artifacts, map of the ActionInput to the Merkle hash
  private final Map<ActionInput, Digest> inputDirectoryDigestCache = new HashMap<>();
  private final Map<TreeNode, Digest> treeNodeDigestCache = new HashMap<>();
  private final Map<Digest, TreeNode> digestTreeNodeCache = new HashMap<>();
  private final Map<TreeNode, Directory> directoryCache = new HashMap<>();
  private final Map<VirtualActionInput, Digest> virtualInputDigestCache = new HashMap<>();
  private final Map<Digest, VirtualActionInput> digestVirtualInputCache = new HashMap<>();
  private final DigestUtil digestUtil;
  private final boolean uploadSymlinks;

  public TreeNodeRepository(
      Path execRoot,
      MetadataProvider inputFileCache,
      DigestUtil digestUtil,
      boolean uploadSymlinks) {
    this.execRoot = execRoot;
    this.inputFileCache = inputFileCache;
    this.digestUtil = digestUtil;
    this.uploadSymlinks = uploadSymlinks;
  }

  public MetadataProvider getInputFileCache() {
    return inputFileCache;
  }

  public Iterable<TreeNode> children(TreeNode node) {
    return Iterables.transform(node.getChildEntries(), TreeNode.ChildEntry::getChild);
  }

  /** Traverse the directory structure in order (pre-order tree traversal). */
  public Iterable<TreeNode> descendants(TreeNode node) {
    return traverser.depthFirstPreOrder(node);
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
  public TreeNode buildFromActionInputs(SortedMap<PathFragment, ActionInput> sortedMap)
      throws IOException {
    ImmutableList.Builder<ImmutableList<String>> segments = ImmutableList.builder();
    for (PathFragment path : sortedMap.keySet()) {
      segments.add(path.getSegments());
    }
    List<ActionInput> inputs = new ArrayList<>();
    for (Map.Entry<PathFragment, ActionInput> e : sortedMap.entrySet()) {
      inputs.add(e.getValue());
    }
    return buildParentNode(inputs, segments.build(), 0, inputs.size(), 0);
  }

  // Expand the descendant of an artifact (input) directory
  private List<TreeNode.ChildEntry> buildInputDirectoryEntries(Path path) throws IOException {
    List<Dirent> sortedDirent = new ArrayList<>(path.readdir(Symlinks.NOFOLLOW));
    sortedDirent.sort(Comparator.comparing(Dirent::getName));

    List<TreeNode.ChildEntry> entries = new ArrayList<>(sortedDirent.size());
    for (Dirent dirent : sortedDirent) {
      String name = dirent.getName();
      Path child = path.getRelative(name);
      TreeNode childNode;
      if (dirent.getType() == Dirent.Type.DIRECTORY) {
        childNode = interner.intern(new TreeNode(buildInputDirectoryEntries(child), null));
      } else if (dirent.getType() == Dirent.Type.SYMLINK) {
        PathFragment target = child.readSymbolicLink();
        // Need to resolve the symbolic link to know what to expand it to, file or directory.
        FileStatus statFollow = child.statIfFound(Symlinks.FOLLOW);
        Preconditions.checkNotNull(statFollow, "Dangling symbolic link %s to %s", child, target);
        boolean uploadSymlinkAsDirectory = !uploadSymlinks || target.isAbsolute();
        if (statFollow.isDirectory() && uploadSymlinkAsDirectory) {
          childNode = interner.intern(new TreeNode(buildInputDirectoryEntries(child), null));
        } else {
          childNode = interner.intern(new TreeNode(ActionInputHelper.fromPath(child.asFragment())));
        }
      } else {
        childNode = interner.intern(new TreeNode(ActionInputHelper.fromPath(child.asFragment())));
      }
      entries.add(new TreeNode.ChildEntry(name, childNode));
    }

    return entries;
  }

  @SuppressWarnings("ReferenceEquality") // Segments are interned.
  private TreeNode buildParentNode(
      List<ActionInput> inputs,
      ImmutableList<ImmutableList<String>> segments,
      int inputsStart,
      int inputsEnd,
      int segmentIndex)
      throws IOException {
    if (segments.isEmpty()) {
      // We sometimes have actions with no inputs (e.g., echo "xyz" > $@), so we need to handle that
      // case here.
      Preconditions.checkState(inputs.isEmpty());
      return EMPTY_NODE;
    }
    if (segmentIndex == segments.get(inputsStart).size()) {
      // Leaf node reached. Must be unique.
      Preconditions.checkArgument(
          inputsStart == inputsEnd - 1, "Encountered two inputs with the same path.");
      ActionInput input = inputs.get(inputsStart);
      Path leafPath = execRoot.getRelative(input.getExecPathString());
      if (!(input instanceof VirtualActionInput) && uploadSymlinks) {
        FileStatus stat = leafPath.stat(Symlinks.NOFOLLOW);
        if (stat.isSymbolicLink()) {
          PathFragment target = leafPath.readSymbolicLink();
          FileStatus statFollow = leafPath.statIfFound(Symlinks.FOLLOW);
          Preconditions.checkNotNull(
              statFollow, "Action input %s is a dangling symbolic link to %s ", leafPath, target);
          if (!target.isAbsolute()) {
            return interner.intern(new TreeNode(input));
          }
        }
      }
      try {
        if (!(input instanceof VirtualActionInput)
            && getInputMetadata(input).getType().isDirectory()) {
          return interner.intern(new TreeNode(buildInputDirectoryEntries(leafPath), input));
        }
      } catch (DigestOfDirectoryException e) {
        return interner.intern(new TreeNode(buildInputDirectoryEntries(leafPath), input));
      }
      return interner.intern(new TreeNode(input));
    }
    ArrayList<TreeNode.ChildEntry> entries = new ArrayList<>();
    String segment = segments.get(inputsStart).get(segmentIndex);
    for (int inputIndex = inputsStart; inputIndex < inputsEnd; ++inputIndex) {
      if (inputIndex + 1 == inputsEnd
          || !segment.equals(segments.get(inputIndex + 1).get(segmentIndex))) {
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
    Collections.sort(entries, TreeNode.ChildEntry.segmentOrder);
    return interner.intern(new TreeNode(entries, null));
  }

  private synchronized Directory getOrComputeDirectory(TreeNode node) throws IOException {
    // Assumes all child digests have already been computed!
    Preconditions.checkArgument(!node.isLeaf());
    Directory directory = directoryCache.get(node);
    if (directory == null) {
      Directory.Builder b = Directory.newBuilder();
      for (TreeNode.ChildEntry entry : node.getChildEntries()) {
        TreeNode child = entry.getChild();
        if (child.isLeaf()) {
          ActionInput input = child.getActionInput();
          if (input instanceof VirtualActionInput) {
            VirtualActionInput virtualInput = (VirtualActionInput) input;
            Digest digest = digestUtil.compute(virtualInput);
            virtualInputDigestCache.put(virtualInput, digest);
            // There may be multiple inputs with the same digest. In that case, we don't care which
            // one we get back from the digestVirtualInputCache later.
            digestVirtualInputCache.put(digest, virtualInput);
            b.addFilesBuilder().setName(entry.getSegment()).setDigest(digest).setIsExecutable(true);
            continue;
          }
          if (uploadSymlinks) {
            // We need to stat the input to check whether it is a symlink.
            // getInputMetadata only gives target metadata.
            Path inputPath = execRoot.getRelative(input.getExecPath());
            FileStatus stat = inputPath.stat(Symlinks.NOFOLLOW);
            if (stat.isSymbolicLink()) {
              PathFragment target = inputPath.readSymbolicLink();
              if (!target.isAbsolute()) {
                b.addSymlinksBuilder().setName(entry.getSegment()).setTarget(target.toString());
                continue;
              }
            }
          }
          Digest digest = DigestUtil.getFromInputCache(input, inputFileCache);
          b.addFilesBuilder().setName(entry.getSegment()).setDigest(digest).setIsExecutable(true);
        } else {
          Digest childDigest = Preconditions.checkNotNull(treeNodeDigestCache.get(child));
          if (child.getActionInput() != null) {
            inputDirectoryDigestCache.put(child.getActionInput(), childDigest);
          }
          b.addDirectoriesBuilder().setName(entry.getSegment()).setDigest(childDigest);
        }
      }
      directory = b.build();
      directoryCache.put(node, directory);
      Digest digest = digestUtil.compute(directory);
      treeNodeDigestCache.put(node, digest);
      digestTreeNodeCache.put(digest, node);
    }
    return directory;
  }

  // Recursively traverses the tree, expanding and computing Merkle digests for nodes for which
  // they have not yet been computed and cached.
  public void computeMerkleDigests(TreeNode root) throws IOException {
    synchronized (this) {
      if (directoryCache.get(root) != null) {
        // Strong assumption: the cache is valid, i.e. parent present implies children present.
        return;
      }
    }
    if (!root.isLeaf()) {
      for (TreeNode child : children(root)) {
        computeMerkleDigests(child);
      }
      getOrComputeDirectory(root);
    }
  }

  /**
   * Should only be used after computeMerkleDigests has been called on one of the node ancestors.
   * Returns the precomputed digest.
   */
  public Digest getMerkleDigest(TreeNode node) throws IOException {
    return node.isLeaf()
        ? actionInputToDigest(node.getActionInput())
        : treeNodeDigestCache.get(node);
  }

  /**
   * Returns the precomputed digests for both data and metadata. Should only be used after
   * computeMerkleDigests has been called on one of the node ancestors.
   */
  public ImmutableCollection<Digest> getAllDigests(TreeNode root) throws IOException {
    ImmutableSet.Builder<Digest> digests = ImmutableSet.builder();
    for (TreeNode node : descendants(root)) {
      digests.add(
          node.isLeaf()
              ? actionInputToDigest(node.getActionInput())
              : Preconditions.checkNotNull(treeNodeDigestCache.get(node)));
    }
    return digests.build();
  }

  private Digest actionInputToDigest(ActionInput input) throws IOException {
    if (input instanceof VirtualActionInput) {
      return Preconditions.checkNotNull(virtualInputDigestCache.get(input));
    }
    FileArtifactValue metadata = getInputMetadata(input);
    byte[] digest = metadata.getDigest();
    if (digest == null) {
      // If the artifact does not have a digest, it is because it is a directory.
      // We get the digest from the set of Merkle hashes computed in this TreeNodeRepository.
      return Preconditions.checkNotNull(
          inputDirectoryDigestCache.get(input),
          "a directory should have a precomputed Merkle hash (instead of a digest)");
    }
    return DigestUtil.getFromInputCache(input, inputFileCache);
  }

  /**
   * Serializes all of the subtree to a Directory list. TODO(olaola): add a version that only copies
   * a part of the tree that we are interested in. Should only be used after computeMerkleDigests
   * has been called on one of the node ancestors.
   */
  // Note: this is not, strictly speaking, thread safe. If someone is deleting cached Merkle hashes
  // while this is executing, it will trigger an exception. But I think this is WAI.
  public ImmutableList<Directory> treeToDirectories(TreeNode root) {
    ImmutableList.Builder<Directory> directories = ImmutableList.builder();
    for (TreeNode node : descendants(root)) {
      if (!node.isLeaf()) {
        directories.add(Preconditions.checkNotNull(directoryCache.get(node)));
      }
    }
    return directories.build();
  }

  /**
   * Should only be used on digests created by a call to computeMerkleDigests. Looks up ActionInputs
   * or Directory messages by cached digests and adds them to the lists.
   */
  public void getDataFromDigests(
      Iterable<Digest> digests,
      Map<Digest, ActionInput> actionInputs,
      Map<Digest, Directory> nodes) {
    for (Digest digest : digests) {
      TreeNode treeNode = digestTreeNodeCache.get(digest);
      if (treeNode != null) {
        nodes.put(digest, Preconditions.checkNotNull(directoryCache.get(treeNode)));
      } else { // If not there, it must be an ActionInput.
        ByteString hexDigest = ByteString.copyFromUtf8(digest.getHash());
        ActionInput input = reverseInputMap.get(hexDigest);
        if (input == null) {
          // ... or a VirtualActionInput.
          input = digestVirtualInputCache.get(digest);
        }
        actionInputs.put(digest, Preconditions.checkNotNull(input));
      }
    }
  }

  private FileArtifactValue getInputMetadata(ActionInput input) throws IOException {
    FileArtifactValue metadata =
        Preconditions.checkNotNull(
            inputFileCache.getMetadata(input), "Missing metadata for: %s", input);
    if (metadata.getDigest() != null) {
      reverseInputMap.put(
          ByteString.copyFrom(LOWER_CASE_HEX.encode(metadata.getDigest()).getBytes(US_ASCII)),
          input);
    }
    return metadata;
  }
}
