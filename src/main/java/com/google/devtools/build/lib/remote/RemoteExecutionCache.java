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
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static java.lang.String.format;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree.PathOrBytes;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxFutures;
import com.google.protobuf.Message;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.subjects.AsyncSubject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A {@link RemoteCache} with additional functionality needed for remote execution. */
public class RemoteExecutionCache extends RemoteCache {

  public RemoteExecutionCache(
      RemoteCacheClient protocolImpl,
      RemoteOptions options,
      DigestUtil digestUtil) {
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
   */
  public void ensureInputsPresent(
      RemoteActionExecutionContext context,
      MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs,
      boolean force)
      throws IOException, InterruptedException {
    ImmutableSet<Digest> allDigests =
        ImmutableSet.<Digest>builder()
            .addAll(merkleTree.getAllDigests())
            .addAll(additionalInputs.keySet())
            .build();

    // Collect digests that are not being or already uploaded
    ConcurrentHashMap<Digest, AsyncSubject<Boolean>> missingDigestSubjects =
        new ConcurrentHashMap<>();

    List<ListenableFuture<Void>> uploadFutures = new ArrayList<>();
    for (Digest digest : allDigests) {
      Completable upload =
          casUploadCache.execute(
              digest,
              Completable.defer(
                  () -> {
                    // The digest hasn't been processed, add it to the collection which will be used
                    // later for findMissingDigests call
                    AsyncSubject<Boolean> missingDigestSubject = AsyncSubject.create();
                    missingDigestSubjects.put(digest, missingDigestSubject);

                    return missingDigestSubject.flatMapCompletable(
                        missing -> {
                          if (!missing) {
                            return Completable.complete();
                          }
                          return RxFutures.toCompletable(
                              () -> uploadBlob(context, digest, merkleTree, additionalInputs),
                              MoreExecutors.directExecutor());
                        });
                  }),
              force);
      uploadFutures.add(RxFutures.toListenableFuture(upload));
    }

    ImmutableSet<Digest> missingDigests;
    try {
      missingDigests = getFromFuture(findMissingDigests(context, missingDigestSubjects.keySet()));
    } catch (IOException | InterruptedException e) {
      for (Map.Entry<Digest, AsyncSubject<Boolean>> entry : missingDigestSubjects.entrySet()) {
        entry.getValue().onError(e);
      }

      if (e instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      throw e;
    }

    for (Map.Entry<Digest, AsyncSubject<Boolean>> entry : missingDigestSubjects.entrySet()) {
      AsyncSubject<Boolean> missingSubject = entry.getValue();
      if (missingDigests.contains(entry.getKey())) {
        missingSubject.onNext(true);
      } else {
        // The digest is already existed in the remote cache, skip the upload.
        missingSubject.onNext(false);
      }
      missingSubject.onComplete();
    }

    waitForBulkTransfer(uploadFutures, /* cancelRemainingOnInterrupt=*/ false);
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
