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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.remote.ContentDigests.ActionKey;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Collection;
import javax.annotation.Nullable;

/** A cache for storing artifacts (input and output) as well as the output of running an action. */
@ThreadCompatible
interface RemoteActionCache {
  // CAS API

  // TODO(olaola): create a unified set of exceptions raised by the cache to encapsulate the
  // underlying CasStatus messages and gRPC errors errors.

  /**
   * Upload enough of the tree metadata and data into remote cache so that the entire tree can be
   * reassembled remotely using the root digest.
   */
  void uploadTree(TreeNodeRepository repository, Path execRoot, TreeNode root)
      throws IOException, InterruptedException;

  /**
   * Download the entire tree data rooted by the given digest and write it into the given location.
   */
  void downloadTree(ContentDigest rootDigest, Path rootLocation)
      throws IOException, CacheNotFoundException;

  /**
   * Download all results of a remotely executed action locally. TODO(olaola): will need to amend to
   * include the {@link com.google.devtools.build.lib.remote.TreeNodeRepository} for updating.
   */
  void downloadAllResults(ActionResult result, Path execRoot)
      throws IOException, CacheNotFoundException;

  /**
   * Upload all results of a locally executed action to the cache. Add the files to the ActionResult
   * builder.
   */
  void uploadAllResults(Path execRoot, Collection<Path> files, ActionResult.Builder result)
      throws IOException, InterruptedException;

  /**
   * Put the file contents in cache if it is not already in it. No-op if the file is already stored
   * in cache. The given path must be a full absolute path.
   *
   * @return The key for fetching the file contents blob from cache.
   */
  ContentDigest uploadFileContents(Path file) throws IOException, InterruptedException;

  /**
   * Put the input file contents in cache if it is not already in it. No-op if the data is already
   * stored in cache.
   *
   * @return The key for fetching the file contents blob from cache.
   */
  ContentDigest uploadFileContents(
      ActionInput input, Path execRoot, ActionInputFileCache inputCache)
      throws IOException, InterruptedException;

  /**
   * Download a blob keyed by the given digest and write it to the specified path. Set the
   * executable parameter to the specified value.
   */
  void downloadFileContents(ContentDigest digest, Path dest, boolean executable)
      throws IOException, CacheNotFoundException;

  /** Upload the given blobs to the cache, and return their digests. */
  ImmutableList<ContentDigest> uploadBlobs(Iterable<byte[]> blobs) throws InterruptedException;

  /** Upload the given blob to the cache, and return its digests. */
  ContentDigest uploadBlob(byte[] blob) throws InterruptedException;

  /** Download and return a blob with a given digest from the cache. */
  byte[] downloadBlob(ContentDigest digest) throws CacheNotFoundException;

  /** Download and return blobs with given digests from the cache. */
  ImmutableList<byte[]> downloadBlobs(Iterable<ContentDigest> digests)
      throws CacheNotFoundException;

  // Execution Cache API

  /** Returns a cached result for a given Action digest, or null if not found in cache. */
  @Nullable
  ActionResult getCachedActionResult(ActionKey actionKey);

  /** Sets the given result as result of the given Action. */
  void setCachedActionResult(ActionKey actionKey, ActionResult result) throws InterruptedException;
}
