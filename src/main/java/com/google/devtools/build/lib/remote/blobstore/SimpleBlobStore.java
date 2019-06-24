// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.blobstore;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.shared.Chunker;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.Message;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An interface for storing BLOBs each one indexed by a string (hash in hexadecimal).
 *
 * <p>Implementations must be thread-safe.
 */
public interface SimpleBlobStore {
  /** Returns {@code true} if the provided {@code key} is stored in the CAS. */
  boolean contains(String key) throws IOException, InterruptedException;

  /** Returns {@code true} if the provided {@code key} is stored in the Action Cache. */
  boolean containsActionResult(String key) throws IOException, InterruptedException;

  /**
   * Fetches the BLOB associated with the {@code key} from the CAS and writes it to {@code out}.
   *
   * <p>The caller is responsible to close {@code out}.
   *
   * @return {@code true} if the {@code key} was found. {@code false} otherwise.
   */
  ListenableFuture<Boolean> get(String key, @Nullable Digest digest, OutputStream out);

  /**
   * Fetches the BLOB associated with the {@code key} from the Action Cache and writes it to {@code
   * out}.
   *
   * <p>The caller is responsible to close {@code out}.
   *
   * @return {@code true} if the {@code key} was found. {@code false} otherwise.
   */
  ListenableFuture<Boolean> getActionResult(Digest digest, OutputStream out);

  /**
   * Uploads a BLOB (as {@code in}) with length {@code length} indexed by {@code key} to the CAS.
   *
   * <p>The caller is responsible to close {@code in}.
   */
  void put(String key, @Nullable Digest digest, long length, @Nullable Chunker chunker, InputStream in) throws IOException, InterruptedException;

  /** Uploads a bytearray BLOB (as {@code in}) indexed by {@code key} to the Action Cache. */
  void putActionResult(Digest digest, ActionResult actionResult) throws IOException, InterruptedException;

  /** Close resources associated with the blob store. */
  void close();

  /**
   * Ensures that the tree structure of the inputs, the input files themselves, and the command are
   * available in the remote cache, such that the tree can be reassembled and executed on another
   * machine given the root digest.
   *
   * <p>The cache may check whether files or parts of the tree structure are already present, and do
   * not need to be uploaded again.
   *
   * <p>Note that this method is only required for remote execution, not for caching itself.
   * However, remote execution uses a cache to store input files, and that may be a separate
   * end-point from the executor itself, so the functionality lives here.
   */
  void ensureInputsPresent(MerkleTree merkleTree, Map<Digest, Message> additionalInputs, Path execRoot) throws IOException, InterruptedException;

  ImmutableSet<Digest> getMissingDigests(Iterable<Digest> digests) throws IOException, InterruptedException;
}
