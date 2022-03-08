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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toSingle;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;
import static java.lang.String.format;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree.PathOrBytes;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.protobuf.Message;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.subjects.AsyncSubject;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

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

    if (allDigests.isEmpty()) {
      return;
    }

    MissingDigestFinder missingDigestFinder = new MissingDigestFinder(context, allDigests.size());
    Flowable<TransferResult> uploads =
        Flowable.fromIterable(allDigests)
            .flatMapSingle(
                digest ->
                    toTransferResult(
                        casUploadCache.execute(
                            digest,
                            Completable.defer(
                                () ->
                                    // We only reach here if the digest is missing or is not being
                                    // uploaded.
                                    missingDigestFinder
                                        .registerAndCount(digest)
                                        .flatMapCompletable(
                                            missingDigests -> {
                                              if (missingDigests.contains(digest)) {
                                                return toCompletable(
                                                    () ->
                                                        uploadBlob(
                                                            context,
                                                            digest,
                                                            merkleTree,
                                                            additionalInputs),
                                                    directExecutor());
                                              } else {
                                                return Completable.complete();
                                              }
                                            })),
                            /*onIgnore=*/ missingDigestFinder::count,
                            force)));

    try {
      mergeBulkTransfer(uploads).blockingAwait();
    } catch (RuntimeException e) {
      Throwable cause = e.getCause();
      if (cause != null) {
        Throwables.throwIfInstanceOf(cause, InterruptedException.class);
        Throwables.throwIfInstanceOf(cause, IOException.class);
      }
      throw e;
    }
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

  /**
   * A missing digest finder that initiates the request when the internal counter reaches an
   * expected count.
   */
  class MissingDigestFinder {
    private final int expectedCount;
    private final AsyncSubject<Set<Digest>> digestsSubject;
    private final Single<ImmutableSet<Digest>> resultSingle;
    private final Set<Digest> digests;

    private int currentCount = 0;

    MissingDigestFinder(RemoteActionExecutionContext context, int expectedCount) {
      checkArgument(expectedCount > 0, "expectedCount should be greater than 0");
      this.expectedCount = expectedCount;
      this.digestsSubject = AsyncSubject.create();
      this.resultSingle =
          Single.fromObservable(
              digestsSubject
                  .flatMapSingle(
                      digests ->
                          toSingle(() -> findMissingDigests(context, digests), directExecutor()))
                  .replay()
                  .refCount());
      this.digests = new HashSet<>();
    }

    /**
     * Register the {@code digest} and increase the counter.
     *
     * @return Single that emits the result of the {@code FindMissingDigest} request.
     */
    synchronized Single<ImmutableSet<Digest>> registerAndCount(Digest digest) {
      digests.add(digest);
      count();
      return resultSingle;
    }

    /** Increase the counter. */
    synchronized void count() {
      if (currentCount < expectedCount) {
        currentCount++;
        if (currentCount == expectedCount) {
          digestsSubject.onNext(digests);
          digestsSubject.onComplete();
        }
      }
    }
  }
}
