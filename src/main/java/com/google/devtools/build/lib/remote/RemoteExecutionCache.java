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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toSingle;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;
import static java.lang.String.format;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.merkletree.v2.MerkleTreeComputer;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableObserver;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Maybe;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleEmitter;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.subjects.AsyncSubject;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** A {@link CombinedCache} with additional functionality needed for remote execution. */
public class RemoteExecutionCache extends CombinedCache {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * An interface used to check whether a given {@link Path} is stored in a remote or a disk cache.
   */
  public interface RemotePathChecker {
    boolean isRemote(RemoteActionExecutionContext context, Path path) throws IOException;
  }

  private RemotePathChecker remotePathChecker =
      new RemotePathChecker() {
        @Override
        public boolean isRemote(RemoteActionExecutionContext context, Path path)
            throws IOException {
          var fs = path.getFileSystem();
          if (fs instanceof RemoteActionFileSystem remoteActionFileSystem) {
            if (remoteActionFileSystem.isRemote(path)) {
              if (context.getReadCachePolicy().allowDiskCache()) {
                try (var inputStream = path.getInputStream()) {
                  // If the file exists in the disk cache, download it and continue the upload.
                  return false;
                } catch (IOException e) {
                  logger.atWarning().withCause(e).log(
                      "Failed to get input stream for %s", path.getPathString());
                }
              }
              return true;
            }
          }
          return false;
        }
      };

  public RemoteExecutionCache(
      RemoteCacheClient remoteCacheClient,
      @Nullable DiskCacheClient diskCacheClient,
      RemoteOptions options,
      DigestUtil digestUtil) {
    super(checkNotNull(remoteCacheClient), diskCacheClient, options, digestUtil);
  }

  @VisibleForTesting
  void setRemotePathChecker(RemotePathChecker remotePathChecker) {
    this.remotePathChecker = remotePathChecker;
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
      MerkleTreeComputer.MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs,
      boolean force,
      @Nullable RemotePathResolver remotePathResolver)
      throws IOException, InterruptedException {
    Iterable<Digest> merkleTreeAllDigests;
    try (SilentCloseable s = Profiler.instance().profile("merkleTree.getAllDigests()")) {
      merkleTreeAllDigests = merkleTree.blobs().keySet();
    }
    Iterable<Digest> allDigests = Iterables.concat(merkleTreeAllDigests, additionalInputs.keySet());

    if (Iterables.isEmpty(allDigests)) {
      return;
    }

    Flowable<TransferResult> uploads =
        createUploadTasks(
                context, merkleTree, additionalInputs, allDigests, force, remotePathResolver)
            .flatMapPublisher(
                result ->
                    Flowable.using(
                        () -> result,
                        uploadTasks ->
                            findMissingBlobs(context, uploadTasks)
                                .flatMapPublisher(this::waitForUploadTasks),
                        uploadTasks -> {
                          for (UploadTask uploadTask : uploadTasks) {
                            Disposable d = uploadTask.disposable.getAndSet(null);
                            if (d != null) {
                              d.dispose();
                            }
                          }
                        }));

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

  private static final class VirtualActionInputBlob implements Blob {
    private VirtualActionInput virtualActionInput;
    // Can be large compared to the retained size of the VirtualActionInput and thus shouldn't be
    // kept in memory for an extended period of time.
    private volatile ByteString data;

    VirtualActionInputBlob(VirtualActionInput virtualActionInput) {
      this.virtualActionInput = Preconditions.checkNotNull(virtualActionInput);
    }

    @Override
    public InputStream get() throws IOException {
      if (data == null) {
        synchronized (this) {
          if (data == null) {
            data = Preconditions.checkNotNull(virtualActionInput, "used after close()").getBytes();
          }
        }
      }
      return data.newInput();
    }

    @Override
    public void close() {
      virtualActionInput = null;
      data = null;
    }
  }

  private ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context,
      Digest digest,
      MerkleTreeComputer.MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs,
      @Nullable RemotePathResolver remotePathResolver) {
    // TODO: Move this into context.
    MerkleTreeComputer.BlobUploader blobUploader =
        new MerkleTreeComputer.BlobUploader() {
          @Override
          public ListenableFuture<Void> upload(Digest digest, byte[] data) {
            return remoteCacheClient.uploadBlob(context, digest, ByteString.copyFrom(data));
          }

          @Override
          public ListenableFuture<Void> upload(Digest digest, Path path) {
            try {
              if (remotePathChecker.isRemote(context, path)) {
                // If we get here, the remote input was determined to exist in the remote or disk
                // cache at some point before action execution, but reported to be missing when
                // querying the remote for missing action inputs; possibly because it was evicted in
                // the interim.
                if (remotePathResolver != null) {
                  throw new CacheNotFoundException(
                      digest, remotePathResolver.localPathToExecPath(path.asFragment()));
                } else {
                  // This path should only be taken for RemoteRepositoryRemoteExecutor, which has no
                  // way to handle lost inputs.
                  throw new CacheNotFoundException(digest, path.getPathString());
                }
              }
            } catch (IOException e) {
              return immediateFailedFuture(e);
            }
            return remoteCacheClient.uploadFile(context, digest, path);
          }

          @Override
          public ListenableFuture<Void> upload(
              Digest digest, VirtualActionInput virtualActionInput) {
            return remoteCacheClient.uploadBlob(
                context, digest, new VirtualActionInputBlob(virtualActionInput));
          }
        };
    var upload = merkleTree.upload(blobUploader, digest);
    if (upload.isPresent()) {
      return upload.get();
    }

    Message message = additionalInputs.get(digest);
    if (message != null) {
      return remoteCacheClient.uploadBlob(context, digest, message.toByteString());
    }

    return immediateFailedFuture(
        new IOException(
            format(
                "findMissingDigests returned a missing digest that has not been requested: %s",
                digest)));
  }

  static class UploadTask {
    Digest digest;
    AtomicReference<Disposable> disposable;
    SingleEmitter<Boolean> continuation;
    Completable completion;
  }

  private Single<List<UploadTask>> createUploadTasks(
      RemoteActionExecutionContext context,
      MerkleTreeComputer.MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs,
      Iterable<Digest> allDigests,
      boolean force,
      @Nullable RemotePathResolver remotePathResolver) {
    return Single.using(
        () -> Profiler.instance().profile("collect digests"),
        ignored ->
            Flowable.fromIterable(allDigests)
                .flatMapMaybe(
                    digest ->
                        maybeCreateUploadTask(
                            context,
                            merkleTree,
                            additionalInputs,
                            digest,
                            force,
                            remotePathResolver))
                .collect(toImmutableList()),
        SilentCloseable::close);
  }

  private Maybe<UploadTask> maybeCreateUploadTask(
      RemoteActionExecutionContext context,
      MerkleTreeComputer.MerkleTree merkleTree,
      Map<Digest, Message> additionalInputs,
      Digest digest,
      boolean force,
      @Nullable RemotePathResolver remotePathResolver) {
    return Maybe.create(
        emitter -> {
          AsyncSubject<Void> completion = AsyncSubject.create();
          UploadTask uploadTask = new UploadTask();
          uploadTask.digest = digest;
          uploadTask.disposable = new AtomicReference<>();
          uploadTask.completion = Completable.fromObservable(completion);
          Completable upload =
              casUploadCache.execute(
                  digest,
                  Single.<Boolean>create(
                          continuation -> {
                            uploadTask.continuation = continuation;
                            emitter.onSuccess(uploadTask);
                          })
                      .flatMapCompletable(
                          shouldUpload -> {
                            if (!shouldUpload) {
                              return Completable.complete();
                            }

                            return toCompletable(
                                () ->
                                    uploadBlob(
                                        context,
                                        uploadTask.digest,
                                        merkleTree,
                                        additionalInputs,
                                        remotePathResolver),
                                directExecutor());
                          }),
                  /* onAlreadyRunning= */ () -> emitter.onSuccess(uploadTask),
                  /* onAlreadyFinished= */ emitter::onComplete,
                  force);
          upload.subscribe(
              new CompletableObserver() {
                @Override
                public void onSubscribe(@NonNull Disposable d) {
                  uploadTask.disposable.set(d);
                }

                @Override
                public void onComplete() {
                  completion.onComplete();
                }

                @Override
                public void onError(@NonNull Throwable e) {
                  completion.onError(e);
                }
              });
        });
  }

  private Single<List<UploadTask>> findMissingBlobs(
      RemoteActionExecutionContext context, List<UploadTask> uploadTasks) {
    return Single.using(
        () -> Profiler.instance().profile("findMissingDigests"),
        ignored ->
            Single.fromObservable(
                Observable.fromSingle(
                        toSingle(
                                () -> {
                                  ImmutableList<Digest> digestsToQuery =
                                      uploadTasks.stream()
                                          .filter(uploadTask -> uploadTask.continuation != null)
                                          .map(uploadTask -> uploadTask.digest)
                                          .collect(toImmutableList());
                                  if (digestsToQuery.isEmpty()) {
                                    return immediateFuture(ImmutableSet.of());
                                  }
                                  return remoteCacheClient.findMissingDigests(
                                      context, digestsToQuery);
                                },
                                directExecutor())
                            .map(
                                missingDigests -> {
                                  for (UploadTask uploadTask : uploadTasks) {
                                    if (uploadTask.continuation != null) {
                                      uploadTask.continuation.onSuccess(
                                          missingDigests.contains(uploadTask.digest));
                                    }
                                  }
                                  return uploadTasks;
                                }))
                    // Use AsyncSubject so that if downstream is disposed, the
                    // findMissingDigests call is not cancelled (because it may be needed by
                    // other threads).
                    .subscribeWith(AsyncSubject.create())),
        SilentCloseable::close);
  }

  private Flowable<TransferResult> waitForUploadTasks(List<UploadTask> uploadTasks) {
    return Flowable.using(
        () -> Profiler.instance().profile("upload"),
        ignored ->
            Flowable.fromIterable(uploadTasks)
                .flatMapSingle(uploadTask -> toTransferResult(uploadTask.completion)),
        SilentCloseable::close);
  }
}
