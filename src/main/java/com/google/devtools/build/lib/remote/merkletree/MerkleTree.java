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
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** A merkle tree representation as defined by the remote execution api. */
public class MerkleTree {

  /** A path or contents */
  public static class PathOrBytes {

    private final Path path;
    private final ByteString bytes;

    public PathOrBytes(Path path) {
      this.path = Preconditions.checkNotNull(path, "path");
      this.bytes = null;
    }

    public PathOrBytes(ByteString bytes) {
      this.bytes = Preconditions.checkNotNull(bytes, "bytes");
      this.path = null;
    }

    @Nullable
    public Path getPath() {
      return path;
    }

    @Nullable
    public ByteString getBytes() {
      return bytes;
    }
  }

  private final Map<Digest, Directory> digestDirectoryMap;
  private final Map<Digest, PathOrBytes> digestFileMap;
  private final Digest rootDigest;

  private MerkleTree(
      Map<Digest, Directory> digestDirectoryMap,
      Map<Digest, PathOrBytes> digestFileMap,
      Digest rootDigest) {
    this.digestDirectoryMap = digestDirectoryMap;
    this.digestFileMap = digestFileMap;
    this.rootDigest = rootDigest;
  }

  /** Returns the digest of the merkle tree's root. */
  public Digest getRootDigest() {
    return rootDigest;
  }

  @Nullable
  public Directory getDirectoryByDigest(Digest digest) {
    return digestDirectoryMap.get(digest);
  }

  @Nullable
  public PathOrBytes getFileByDigest(Digest digest) {
    return digestFileMap.get(digest);
  }

  /**
   * Returns the hashes of all nodes and leafs of the merkle tree. That is, the hashes of the {@link
   * Directory} protobufs and {@link ActionInput} files.
   */
  public Iterable<Digest> getAllDigests() {
    return Iterables.concat(digestDirectoryMap.keySet(), digestFileMap.keySet());
  }

  /**
   * Constructs a merkle tree from a lexicographically sorted map of inputs (files).
   *
   * @param inputs a map of path to input. The map is required to be sorted lexicographically by
   *     paths. Inputs of type tree artifacts are not supported and are expected to have been
   *     expanded before.
   * @param metadataProvider provides metadata for all {@link ActionInput}s in {@code inputs}, as
   *     well as any {@link ActionInput}s being discovered via directory expansion.
   * @param execRoot all paths in {@code inputs} need to be relative to this {@code execRoot}.
   * @param digestUtil a hashing utility
   */
  public static MerkleTree build(
      SortedMap<PathFragment, ActionInput> inputs,
      MetadataProvider metadataProvider,
      Path execRoot,
      DigestUtil digestUtil)
      throws IOException {
    try (SilentCloseable c = Profiler.instance().profile("MerkleTree.build(ActionInput)")) {
      DirectoryTree tree =
          DirectoryTreeBuilder.fromActionInputs(inputs, metadataProvider, execRoot, digestUtil);
      return build(tree, digestUtil);
    }
  }

  /**
   * Constructs a merkle tree from a lexicographically sorted map of files.
   *
   * @param inputFiles a map of path to files. The map is required to be sorted lexicographically by
   *     paths.
   * @param digestUtil a hashing utility
   */
  public static MerkleTree build(SortedMap<PathFragment, Path> inputFiles, DigestUtil digestUtil)
      throws IOException {
    try (SilentCloseable c = Profiler.instance().profile("MerkleTree.build(Path)")) {
      DirectoryTree tree = DirectoryTreeBuilder.fromPaths(inputFiles, digestUtil);
      return build(tree, digestUtil);
    }
  }

  private static MerkleTree build(DirectoryTree tree, DigestUtil digestUtil) {
    Preconditions.checkNotNull(tree);
    if (tree.isEmpty()) {
      return new MerkleTree(ImmutableMap.of(), ImmutableMap.of(), digestUtil.compute(new byte[0]));
    }
    Map<Digest, Directory> digestDirectoryMap =
        Maps.newHashMapWithExpectedSize(tree.numDirectories());
    Map<Digest, PathOrBytes> digestPathMap = Maps.newHashMapWithExpectedSize(tree.numFiles());
    Map<PathFragment, Digest> m = new HashMap<>();
    tree.visit(
        (dirname, files, dirs) -> {
          Directory.Builder b = Directory.newBuilder();
          for (DirectoryTree.FileNode file : files) {
            b.addFiles(buildProto(file));
            digestPathMap.put(file.getDigest(), toPathOrBytes(file));
          }
          for (DirectoryTree.DirectoryNode dir : dirs) {
            PathFragment subDirname = dirname.getRelative(dir.getPathSegment());
            Digest protoDirDigest =
                Preconditions.checkNotNull(m.remove(subDirname), "protoDirDigest was null");
            b.addDirectories(buildProto(dir, protoDirDigest));
          }
          Directory protoDir = b.build();
          Digest protoDirDigest = digestUtil.compute(protoDir);
          digestDirectoryMap.put(protoDirDigest, protoDir);
          m.put(dirname, protoDirDigest);
        });
    return new MerkleTree(digestDirectoryMap, digestPathMap, m.get(PathFragment.EMPTY_FRAGMENT));
  }

  private static FileNode buildProto(DirectoryTree.FileNode file) {
    return FileNode.newBuilder()
        .setName(file.getPathSegment())
        .setDigest(file.getDigest())
        .setIsExecutable(true)
        .build();
  }

  private static DirectoryNode buildProto(DirectoryTree.DirectoryNode dir, Digest protoDirDigest) {
    return DirectoryNode.newBuilder()
        .setName(dir.getPathSegment())
        .setDigest(protoDirDigest)
        .build();
  }

  private static PathOrBytes toPathOrBytes(DirectoryTree.FileNode file) {
    return file.getPath() != null
        ? new PathOrBytes(file.getPath())
        : new PathOrBytes(file.getBytes());
  }
}
