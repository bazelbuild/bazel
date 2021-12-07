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
package com.google.devtools.build.lib.remote.disk;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.Utils.shouldAcceptCachedResultFromRemoteCache;
import static com.google.devtools.build.lib.remote.util.Utils.shouldUploadLocalResultsToRemoteCache;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.LazyFileOutputStream;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.util.UUID;

/**
 * A {@link RemoteCacheClient} implementation combining two blob stores. A local disk blob store and
 * a remote blob store. If a blob isn't found in the first store, the second store is used, and the
 * blob added to the first. Put puts the blob on both stores.
 */
public final class DiskAndRemoteCacheClient implements RemoteCacheClient {

  private final RemoteCacheClient remoteCache;
  private final DiskCacheClient diskCache;
  private final RemoteOptions options;

  public DiskAndRemoteCacheClient(
      DiskCacheClient diskCache, RemoteCacheClient remoteCache, RemoteOptions options) {
    this.diskCache = Preconditions.checkNotNull(diskCache);
    this.remoteCache = Preconditions.checkNotNull(remoteCache);
    this.options = options;
  }

  @Override
  public ListenableFuture<Void> uploadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, ActionResult actionResult) {
    ListenableFuture<Void> future = diskCache.uploadActionResult(context, actionKey, actionResult);
    if (shouldUploadLocalResultsToRemoteCache(options, context.getSpawn())) {
      future =
          Futures.transformAsync(
              future,
              v -> remoteCache.uploadActionResult(context, actionKey, actionResult),
              directExecutor());
    }
    return future;
  }

  @Override
  public void close() {
    diskCache.close();
    remoteCache.close();
  }

  @Override
  public ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context, Digest digest, Path file) {
    ListenableFuture<Void> future = diskCache.uploadFile(context, digest, file);

    boolean uploadForSpawn = context.getSpawn() != null;
    // If not upload for spawn e.g. for build event artifacts, we always upload files to remote
    // cache.
    if (!uploadForSpawn
        || options.isRemoteExecutionEnabled()
        || shouldUploadLocalResultsToRemoteCache(options, context.getSpawn())) {
      future =
          Futures.transformAsync(
              future, v -> remoteCache.uploadFile(context, digest, file), directExecutor());
    }
    return future;
  }

  @Override
  public ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, ByteString data) {
    ListenableFuture<Void> future = diskCache.uploadBlob(context, digest, data);
    if (options.isRemoteExecutionEnabled()
        || shouldUploadLocalResultsToRemoteCache(options, context.getSpawn())) {
      future =
          Futures.transformAsync(
              future, v -> remoteCache.uploadBlob(context, digest, data), directExecutor());
    }
    return future;
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
      RemoteActionExecutionContext context, Iterable<Digest> digests) {
    // If remote execution, find missing digests should only look at
    // the remote cache, not the disk cache because the remote executor only
    // has access to the remote cache, not the disk cache.
    // Also, the DiskCache always returns all digests as missing
    // and we don't want to transfer all the files all the time.
    if (options.isRemoteExecutionEnabled()) {
      return remoteCache.findMissingDigests(context, digests);
    }
    ListenableFuture<ImmutableSet<Digest>> diskQuery =
        diskCache.findMissingDigests(context, digests);
    if (shouldUploadLocalResultsToRemoteCache(options, context.getSpawn())) {
      ListenableFuture<ImmutableSet<Digest>> remoteQuery =
          remoteCache.findMissingDigests(context, digests);
      return Futures.whenAllSucceed(remoteQuery, diskQuery)
          .call(
              () ->
                  ImmutableSet.<Digest>builder()
                      .addAll(remoteQuery.get())
                      .addAll(diskQuery.get())
                      .build(),
              directExecutor());
    } else {
      return diskQuery;
    }
  }

  private Path newTempPath() {
    return diskCache.toPathNoSplit(UUID.randomUUID().toString());
  }

  private static ListenableFuture<Void> closeStreamOnError(
      ListenableFuture<Void> f, OutputStream out) {
    return Futures.catchingAsync(
        f,
        Exception.class,
        (rootCause) -> {
          try {
            out.close();
          } catch (IOException e) {
            rootCause.addSuppressed(e);
          }
          return Futures.immediateFailedFuture(rootCause);
        },
        directExecutor());
  }

  @Override
  public ListenableFuture<Void> downloadBlob(
      RemoteActionExecutionContext context, Digest digest, OutputStream out) {
    if (diskCache.contains(digest)) {
      return diskCache.downloadBlob(context, digest, out);
    }

    Path tempPath = newTempPath();
    final OutputStream tempOut;
    tempOut = new LazyFileOutputStream(tempPath);

    if (options.isRemoteExecutionEnabled()
        || shouldAcceptCachedResultFromRemoteCache(options, context.getSpawn())) {
      ListenableFuture<Void> download =
          closeStreamOnError(remoteCache.downloadBlob(context, digest, tempOut), tempOut);
      return Futures.transformAsync(
          download,
          (unused) -> {
            try {
              tempOut.close();
              diskCache.captureFile(tempPath, digest, /* isActionCache= */ false);
            } catch (IOException e) {
              return Futures.immediateFailedFuture(e);
            }
            return diskCache.downloadBlob(context, digest, out);
          },
          directExecutor());
    } else {
      return Futures.immediateFuture(null);
    }
  }

  @Override
  public ListenableFuture<CachedActionResult> downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr) {
    if (diskCache.containsActionResult(actionKey)) {
      return diskCache.downloadActionResult(context, actionKey, inlineOutErr);
    }

    if (shouldAcceptCachedResultFromRemoteCache(options, context.getSpawn())) {
      return Futures.transformAsync(
          remoteCache.downloadActionResult(context, actionKey, inlineOutErr),
          (cachedActionResult) -> {
            if (cachedActionResult == null) {
              return Futures.immediateFuture(null);
            } else {
              return Futures.transform(
                  diskCache.uploadActionResult(
                      context, actionKey, cachedActionResult.actionResult()),
                  v -> cachedActionResult,
                  directExecutor());
            }
          },
          directExecutor());
    } else {
      return Futures.immediateFuture(null);
    }
  }
}
