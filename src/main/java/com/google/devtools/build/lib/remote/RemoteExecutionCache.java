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
import static com.google.common.base.Preconditions.checkState;
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
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.concurrent.GuardedBy;

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
                    uploadBlobIfMissing(
                        context, merkleTree, additionalInputs, force, missingDigestFinder, digest));

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

  private Single<TransferResult> uploadBlobIfMissing(
      RemoteActionExecutionContext context,
      MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs,
      boolean force,
      MissingDigestFinder missingDigestFinder,
      Digest digest) {
    Completable upload =
        casUploadCache.execute(
            digest,
            Completable.defer(
                () ->
                    // Only reach here if the digest is missing and is not being uploaded.
                    missingDigestFinder
                        .registerAndCount(digest)
                        .flatMapCompletable(
                            missingDigests -> {
                              if (missingDigests.contains(digest)) {
                                return toCompletable(
                                    () -> uploadBlob(context, digest, merkleTree, additionalInputs),
                                    directExecutor());
                              } else {
                                return Completable.complete();
                              }
                            })),
            /* onIgnored= */ missingDigestFinder::count,
            force);
    return toTransferResult(upload);
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

    private final AsyncSubject<ImmutableSet<Digest>> digestsSubject;
    private final Single<ImmutableSet<Digest>> resultSingle;

    @GuardedBy("this")
    private final Set<Digest> digests;

    @GuardedBy("this")
    private int currentCount = 0;

    MissingDigestFinder(RemoteActionExecutionContext context, int expectedCount) {
      checkArgument(expectedCount > 0, "expectedCount should be greater than 0");
      this.expectedCount = expectedCount;
      this.digestsSubject = AsyncSubject.create();
      this.digests = new HashSet<>();

      AtomicBoolean findMissingDigestsCalled = new AtomicBoolean(false);
      this.resultSingle =
          Single.fromObservable(
              digestsSubject
                  .flatMapSingle(
                      digests -> {
                        boolean wasCalled = findMissingDigestsCalled.getAndSet(true);
                        // Make sure we don't have re-subscription caused by refCount() below.
                        checkState(!wasCalled, "FindMissingDigests is called more than once");
                        return toSingle(
                            () -> findMissingDigests(context, digests), directExecutor());
                      })
                  // Use replay here because we could have a race condition that downstream hasn't
                  // been added to the subscription list (to receive the upstream result) while
                  // upstream is completed.
                  .replay(1)
                  .refCount());
    }

    /**
     * Register the {@code digest} and increase the counter.
     *
     * <p>Returned Single cannot be subscribed more than once.
     *
     * @return Single that emits the result of the {@code FindMissingDigest} request.
     */
    Single<ImmutableSet<Digest>> registerAndCount(Digest digest) {
      AtomicBoolean subscribed = new AtomicBoolean(false);
      // count() will potentially trigger the findMissingDigests call. Adding and counting before
      // returning the Single could introduce a race that the result of findMissingDigests is
      // available but the consumer doesn't get it because it hasn't subscribed the returned
      // Single. In this case, it subscribes after upstream is completed resulting a re-run of
      // findMissingDigests (due to refCount()).
      //
      // Calling count() inside doOnSubscribe to ensure the consumer already subscribed to the
      // returned Single to avoid a re-execution of findMissingDigests.
      return resultSingle.doOnSubscribe(
          d -> {
            boolean wasSubscribed = subscribed.getAndSet(true);
            checkState(!wasSubscribed, "Single is subscribed more than once");
            synchronized (this) {
              digests.add(digest);
            }
            count();
          });
    }

    /** Increase the counter. */
    void count() {
      ImmutableSet<Digest> digestsResult = null;

      synchronized (this) {
        if (currentCount < expectedCount) {
          currentCount++;
          if (currentCount == expectedCount) {
            digestsResult = ImmutableSet.copyOf(digests);
          }
        }
      }

      if (digestsResult != null) {
        digestsSubject.onNext(digestsResult);
        digestsSubject.onComplete();
      }
    }
  }
}
