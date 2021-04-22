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
package com.google.devtools.build.lib.remote;

import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static java.lang.String.format;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree.PathOrBytes;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.protobuf.Message;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A {@link RemoteCache} with additional functionality needed for remote execution. */
public class RemoteExecutionCache extends RemoteCache {
  private final ConcurrentHashMap<Digest, ListenableFuture<Void>> uploadFutures =
      new ConcurrentHashMap<>();

  public RemoteExecutionCache(
      RemoteCacheClient protocolImpl, RemoteOptions options, DigestUtil digestUtil) {
    super(protocolImpl, options, digestUtil);
  }

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
   *
   * <p>Callers should pass {@code true} for {@code checkAll} on retries as a precaution to avoid
   * getting stale results from other threads.
   */
  public void ensureInputsPresent(
      RemoteActionExecutionContext context,
      MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs,
      boolean checkAll)
      throws IOException, InterruptedException {
    Iterable<Digest> allDigests =
        Iterables.concat(merkleTree.getAllDigests(), additionalInputs.keySet());
    List<ListenableFuture<Void>> futures = new ArrayList<>();

    if (checkAll || !options.experimentalRemoteDeduplicateUploads) {
      ImmutableSet<Digest> missingDigests =
          getFromFuture(cacheProtocol.findMissingDigests(context, allDigests));
      for (Digest missingDigest : missingDigests) {
        futures.add(uploadBlob(context, missingDigest, merkleTree, additionalInputs));
      }
    } else {
      // This code deduplicates findMissingDigests calls as well as file uploads. It works by
      // distinguishing owned vs. unowned futures. Owned futures are owned by the current thread,
      // i.e., the current thread was the first thread to add a future to the uploadFutures map.
      // Unowned futures are those which another thread added first.
      // This thread completes all owned futures, and also waits for all unowned futures (some or
      // all of which may already be completed).
      // Note that we add all futures (both owned and unowned) to the futures list, so we only need
      // a single waitForBulkTransfer call below.
      Map<Digest, SettableFuture<Void>> ownedFutures = new HashMap<>();
      for (Digest d : allDigests) {
        // We expect the majority of digests to already have an entry in the map. Therefore, we
        // check the map first *before* we create a SettableFuture instance (to avoid unnecessary
        // gc for those cases). Unfortunately, we cannot use computeIfAbsent here because we need to
        // know if this thread owns the future or not.
        ListenableFuture<Void> future = uploadFutures.get(d);
        if (future == null) {
          SettableFuture<Void> settableFuture = SettableFuture.create();
          ListenableFuture<Void> previous = uploadFutures.putIfAbsent(d, settableFuture);
          if (previous != null) {
            // We raced and we lost.
            future = previous;
          } else {
            ownedFutures.put(d, settableFuture);
            future = settableFuture;
          }
        }
        futures.add(future);
      }

      // Call findMissingDigests for all owned digests.
      ImmutableSet<Digest> missingDigests;
      try {
        missingDigests =
            getFromFuture(cacheProtocol.findMissingDigests(context, ownedFutures.keySet()));
      } catch (IOException | RuntimeException e) {
        // If something goes wrong in the findMissingDigests call, we need to complete the owned
        // futures, otherwise other threads may hang.
        for (SettableFuture<Void> future : ownedFutures.values()) {
          future.setException(e);
        }
        throw e;
      } catch (InterruptedException e) {
        for (SettableFuture<Void> future : ownedFutures.values()) {
          future.cancel(false);
        }
        throw e;
      }

      for (Map.Entry<Digest, SettableFuture<Void>> e : ownedFutures.entrySet()) {
        Digest digest = e.getKey();
        SettableFuture<Void> future = e.getValue();
        if (missingDigests.contains(digest)) {
          // Upload if the digest is missing from the remote cache.
          ListenableFuture<Void> uploadFuture =
              uploadBlob(context, digest, merkleTree, additionalInputs);
          future.setFuture(uploadFuture);
        } else {
          // We need to complete *all* futures we own, including when they are actually present in
          // the remote cache.
          future.set(null);
        }
      }
    }

    waitForBulkTransfer(futures, /* cancelRemainingOnInterrupt=*/ false);
  }

  private ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context,
      Digest digest,
      MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs) {
    Directory node = merkleTree.getDirectoryByDigest(digest);
    if (node != null) {
      return cacheProtocol.uploadBlob(context, digest, node.toByteString());
    }

    PathOrBytes file = merkleTree.getFileByDigest(digest);
    if (file != null) {
      if (file.getBytes() != null) {
        return cacheProtocol.uploadBlob(context, digest, file.getBytes());
      }
      return cacheProtocol.uploadFile(context, digest, file.getPath());
    }

    Message message = additionalInputs.get(digest);
    if (message != null) {
      return cacheProtocol.uploadBlob(context, digest, message.toByteString());
    }

    return Futures.immediateFailedFuture(
        new IOException(
            format(
                "findMissingDigests returned a missing digest that has not been requested: %s",
                digest)));
  }
}
