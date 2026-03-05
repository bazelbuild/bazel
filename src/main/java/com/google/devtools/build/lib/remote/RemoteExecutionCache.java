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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toSingle;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;
import static java.lang.String.format;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.merkletree.MerkleTreeUploader;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.devtools.build.lib.vfs.Path;
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
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** A {@link CombinedCache} with additional functionality needed for remote execution. */
public class RemoteExecutionCache extends CombinedCache implements MerkleTreeUploader {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * An interface used to check whether a given {@link Path} is available without contacting the
   * remote cache, i.e., it is present on the local disk, perhaps after being downloaded from the
   * disk cache.
   */
  public interface RemotePathChecker {
    ListenableFuture<Boolean> isAvailableLocally(RemoteActionExecutionContext context, Path path);
  }

  private RemotePathChecker remotePathChecker =
      new RemotePathChecker() {
        @Override
        public ListenableFuture<Boolean> isAvailableLocally(
            RemoteActionExecutionContext context, Path path) {
          var fs = path.getFileSystem();
          if (!(fs instanceof RemoteActionFileSystem remoteActionFileSystem)) {
            return immediateFuture(true);
          }
          // If the file is available in the disk cache, we can attempt to download it from there.
          ListenableFuture<Void> downloadFromDiskCache = immediateVoidFuture();
          if (context.getReadCachePolicy().allowDiskCache()) {
            downloadFromDiskCache =
                Futures.catchingAsync(
                    remoteActionFileSystem.downloadIfRemote(path.asFragment()),
                    IOException.class,
                    e -> {
                      logger.atWarning().withCause(e).log(
                          "Failed to download %s", path.getPathString());
                      return immediateVoidFuture();
                    },
                    directExecutor());
          }
          return Futures.transform(
              downloadFromDiskCache,
              unused -> remoteActionFileSystem.getHostFileSystem().exists(path.asFragment()),
              directExecutor());
        }
      };

  public RemoteExecutionCache(
      RemoteCacheClient remoteCacheClient,
      @Nullable DiskCacheClient diskCacheClient,
      @Nullable String symlinkTemplate,
      DigestUtil digestUtil) {
    super(checkNotNull(remoteCacheClient), diskCacheClient, symlinkTemplate, digestUtil);
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
      MerkleTree.Uploadable merkleTree,
      Map<Digest, Message> additionalInputs,
      boolean force,
      @Nullable RemotePathResolver remotePathResolver)
      throws IOException, InterruptedException {
    Flowable<TransferResult> uploads =
        createUploadTasks(context, merkleTree, additionalInputs, force, remotePathResolver)
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

  @Override
  public void ensureInputsPresent(
      RemoteActionExecutionContext context,
      MerkleTree.Uploadable merkleTree,
      boolean force,
      RemotePathResolver remotePathResolver)
      throws IOException, InterruptedException {
    ensureInputsPresent(context, merkleTree, ImmutableMap.of(), force, remotePathResolver);
  }

  @Override
  public ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context,
      RemotePathResolver remotePathResolver,
      Digest digest,
      Path path) {
    return Futures.transformAsync(
        remotePathChecker.isAvailableLocally(context, path),
        isAvailableLocally -> {
          if (!isAvailableLocally) {
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
          return remoteCacheClient.uploadFile(context, digest, path);
        },
        directExecutor());
  }

  @Override
  public ListenableFuture<Void> uploadVirtualActionInput(
      RemoteActionExecutionContext context, Digest digest, VirtualActionInput virtualActionInput) {
    return remoteCacheClient.uploadBlob(
        context, digest, new VirtualActionInputBlob(virtualActionInput));
  }

  private record VirtualActionInputBlob(VirtualActionInput virtualActionInput) implements Blob {
    @SuppressWarnings("AllowVirtualThreads")
    private static final ExecutorService VIRTUAL_ACTION_INPUT_PIPE_EXECUTOR =
        Executors.newThreadPerTaskExecutor(
            Thread.ofVirtual().name("virtual-action-input-pipe-", 0).factory());

    @Override
    public InputStream get() {
      // Avoid materializing and retaining VirtualActionInput.getBytes() during the upload. This
      // can result in high memory usage with many parallel actions with large virtual inputs. Limit
      // this memory usage to the fixed buffer size by using a piped stream.
      var pipedIn = new PipedInputStream(Chunker.getDefaultChunkSize());
      PipedOutputStream pipedOut;
      try {
        pipedOut = new PipedOutputStream(pipedIn);
      } catch (IOException e) {
        throw new IllegalStateException(
            "PipedOutputStream constructor is not expected to throw", e);
      }
      // Note that while Piped{Input,Output}Stream are not directly I/O-bound, bytes read from
      // pipedIn are sent out via gRPC before more bytes are read. As a result, pipedOut is expected
      // to block frequently enough to make virtual threads suitable here.
      var unused =
          VIRTUAL_ACTION_INPUT_PIPE_EXECUTOR.submit(
              () -> {
                try (pipedOut) {
                  virtualActionInput.writeTo(pipedOut);
                } catch (IOException e) {
                  // Since VirtualActionInput#writeTo only throws when pipedOut does, this means
                  // that the reader has closed pipedIn early, perhaps due to interruption. Since
                  // the reader is gone, there is no way to propagate this exception back.
                }
              });
      return pipedIn;
    }
  }

  @Override
  public ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, byte[] data) {
    return remoteCacheClient.uploadBlob(context, digest, () -> new ByteArrayInputStream(data));
  }

  private ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context,
      Digest digest,
      MerkleTree.Uploadable merkleTree,
      Map<Digest, Message> additionalInputs,
      @Nullable RemotePathResolver remotePathResolver) {
    var upload = merkleTree.upload(this, context, remotePathResolver, digest);
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
      MerkleTree.Uploadable merkleTree,
      Map<Digest, Message> additionalInputs,
      boolean force,
      @Nullable RemotePathResolver remotePathResolver) {
    var allDigests = Iterables.concat(merkleTree.allDigests(), additionalInputs.keySet());
    if (Iterables.isEmpty(allDigests)) {
      return Single.just(ImmutableList.of());
    }
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
      MerkleTree.Uploadable merkleTree,
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
