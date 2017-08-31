// Copyright 2018 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;
import java.util.SortedMap;

/**
 * A visitor for {@link TreeNodeRepository} objects. Provides a context for
 * computing and caching Merkle hashes via the repository on all objects with
 * a single inputFileCache.
 */
@ThreadSafe
public final class TreeNodeRepositoryVisitor {
  private final Path execRoot;
  private final DigestUtil digestUtil;
  private final TreeNodeRepository repository;
  private final MetadataProvider inputFileCache;

  public TreeNodeRepositoryVisitor(
      Path execRoot,
      DigestUtil digestUtil,
      TreeNodeRepository repository,
      MetadataProvider inputFileCache) {
    this.execRoot = execRoot;
    this.digestUtil = digestUtil;
    this.repository = repository;
    this.inputFileCache = inputFileCache;
  }

  public TreeNodeRepository getRepository() {
    return repository;
  }

  public MetadataProvider getInputFileCache() {
    return inputFileCache;
  }

  public TreeNode buildFromActionInputs(SortedMap<PathFragment, ActionInput> sortedMap)
      throws IOException {
    return repository.buildFromActionInputs(sortedMap, execRoot, inputFileCache);
  }

  public void computeMerkleDigests(TreeNode root) throws IOException {
    repository.computeMerkleDigests(root, digestUtil, inputFileCache);
  }

  public Digest getMerkleDigest(TreeNode node) throws IOException {
    return repository.getMerkleDigest(node, inputFileCache);
  }

  public ImmutableCollection<Digest> getAllDigests(TreeNode root) throws IOException {
    return repository.getAllDigests(root, inputFileCache);
  }

  public void getDataFromDigests(
      Iterable<Digest> digests,
      Map<Digest, ActionInput> actionInputs,
      Map<Digest, Directory> nodes) {
    repository.getDataFromDigests(digests, actionInputs, nodes, inputFileCache);
  }
}
