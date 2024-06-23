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

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.DigestFunction;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.Store;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.LazyFileOutputStream;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;

/**
 * A {@link RemoteCacheClient} implementation combining two blob stores. A local disk blob store and
 * a remote blob store. If a blob isn't found in the first store, the second store is used, and the
 * blob added to the first. Put puts the blob on both stores.
 */
public final class DiskAndRemoteCacheClient implements RemoteCacheClient {

  private final RemoteCacheClient remoteCache;
  private final DiskCacheClient diskCache;

  public DiskAndRemoteCacheClient(DiskCacheClient diskCache, RemoteCacheClient remoteCache) {
    this.diskCache = Preconditions.checkNotNull(diskCache);
    this.remoteCache = Preconditions.checkNotNull(remoteCache);
  }

  @Override
  public ListenableFuture<Void> uploadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, ActionResult actionResult) {
    ListenableFuture<Void> future = Futures.immediateVoidFuture();

    if (context.getWriteCachePolicy().allowDiskCache()) {
      future = diskCache.uploadActionResult(context, actionKey, actionResult);
    }

    if (context.getWriteCachePolicy().allowRemoteCache()) {
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
    ListenableFuture<Void> future = Futures.immediateVoidFuture();

    if (context.getWriteCachePolicy().allowDiskCache()) {
      future = diskCache.uploadFile(context, digest, file);
    }

    if (context.getWriteCachePolicy().allowRemoteCache()) {
      future =
          Futures.transformAsync(
              future, v -> remoteCache.uploadFile(context, digest, file), directExecutor());
    }
    return future;
  }

  @Override
  public ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, ByteString data) {
    ListenableFuture<Void> future = Futures.immediateVoidFuture();

    if (context.getWriteCachePolicy().allowDiskCache()) {
      future = diskCache.uploadBlob(context, digest, data);
    }

    if (context.getWriteCachePolicy().allowRemoteCache()) {
      future =
          Futures.transformAsync(
              future, v -> remoteCache.uploadBlob(context, digest, data), directExecutor());
    }
    return future;
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
      RemoteActionExecutionContext context, Iterable<Digest> digests) {
    ListenableFuture<ImmutableSet<Digest>> diskQuery = immediateFuture(ImmutableSet.of());
    if (context.getWriteCachePolicy().allowDiskCache()) {
      diskQuery = diskCache.findMissingDigests(context, digests);
    }

    ListenableFuture<ImmutableSet<Digest>> remoteQuery = immediateFuture(ImmutableSet.of());
    if (context.getWriteCachePolicy().allowRemoteCache()) {
      remoteQuery = remoteCache.findMissingDigests(context, digests);
    }

    ListenableFuture<ImmutableSet<Digest>> diskQueryFinal = diskQuery;
    ListenableFuture<ImmutableSet<Digest>> remoteQueryFinal = remoteQuery;

    return Futures.whenAllSucceed(remoteQueryFinal, diskQueryFinal)
        .call(
            () ->
                ImmutableSet.<Digest>builder()
                    .addAll(remoteQueryFinal.get())
                    .addAll(diskQueryFinal.get())
                    .build(),
            directExecutor());
  }

  private Path getTempPath() {
    return diskCache.getTempPath();
  }

  private static ListenableFuture<Void> cleanupTempFileOnError(
      ListenableFuture<Void> f, Path tempPath, OutputStream tempOut) {
    return Futures.catchingAsync(
        f,
        Exception.class,
        (rootCause) -> {
          try {
            tempOut.close();
          } catch (IOException e) {
            rootCause.addSuppressed(e);
          }
          try {
            tempPath.delete();
          } catch (IOException e) {
            rootCause.addSuppressed(e);
          }
          return immediateFailedFuture(rootCause);
        },
        directExecutor());
  }

  @Override
  public ListenableFuture<Void> downloadBlob(
      RemoteActionExecutionContext context,
      Digest digest,
      @Nullable DigestFunction.Value digestFunction,
      OutputStream out) {
    if (context.getReadCachePolicy().allowDiskCache()) {
      return Futures.catchingAsync(
          diskCache.downloadBlob(context, digest, digestFunction, out),
          CacheNotFoundException.class,
          (unused) -> downloadBlobFromRemote(context, digest, digestFunction, out),
          directExecutor());
    } else {
      return downloadBlobFromRemote(context, digest, digestFunction, out);
    }
  }

  private ListenableFuture<Void> downloadBlobFromRemote(
      RemoteActionExecutionContext context,
      Digest digest,
      @Nullable DigestFunction.Value digestFunction,
      OutputStream out) {
    if (context.getReadCachePolicy().allowRemoteCache()) {
      Path tempPath = getTempPath();
      LazyFileOutputStream tempOut = new LazyFileOutputStream(tempPath);

      ListenableFuture<Void> download =
          cleanupTempFileOnError(
              remoteCache.downloadBlob(context, digest, digestFunction, tempOut),
              tempPath,
              tempOut);
      return Futures.transformAsync(
          download,
          (unused) -> {
            try {
              // Fsync temp before we rename it to avoid data loss in the case of machine
              // crashes (the OS may reorder the writes and the rename).
              tempOut.syncIfPossible();
              tempOut.close();
              diskCache.captureFile(tempPath, digest, Store.CAS);
            } catch (IOException e) {
              return immediateFailedFuture(e);
            }
            return diskCache.downloadBlob(context, digest, digestFunction, out);
          },
          directExecutor());
    } else {
      return immediateFuture(null);
    }
  }

  private ListenableFuture<CachedActionResult> downloadActionResultFromRemote(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr) {
    return Futures.transformAsync(
        remoteCache.downloadActionResult(context, actionKey, inlineOutErr),
        (cachedActionResult) -> {
          if (cachedActionResult == null) {
            return immediateFuture(null);
          } else {
            return Futures.transform(
                diskCache.uploadActionResult(context, actionKey, cachedActionResult.actionResult()),
                v -> cachedActionResult,
                directExecutor());
          }
        },
        directExecutor());
  }

  @Override
  public CacheCapabilities getCacheCapabilities() throws IOException {
    return remoteCache.getCacheCapabilities();
  }

  @Override
  public ListenableFuture<String> getAuthority() {
    return remoteCache.getAuthority();
  }

  @Override
  public ListenableFuture<CachedActionResult> downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr) {
    ListenableFuture<CachedActionResult> future = immediateFuture(null);

    if (context.getReadCachePolicy().allowDiskCache()) {
      // If Build without the Bytes is enabled, the future will likely return null
      // and fallback to remote cache because AC integrity check is enabled and referenced blobs are
      // probably missing from disk cache due to BwoB.
      //
      // TODO(chiwang): With lease service, instead of doing the integrity check against local
      // filesystem, we can check whether referenced blobs are alive in the lease service to
      // increase the cache-hit rate for disk cache.
      future = diskCache.downloadActionResult(context, actionKey, inlineOutErr);
    }

    if (context.getReadCachePolicy().allowRemoteCache()) {
      future =
          Futures.transformAsync(
              future,
              (result) -> {
                if (result == null) {
                  return downloadActionResultFromRemote(context, actionKey, inlineOutErr);
                } else {
                  return immediateFuture(result);
                }
              },
              directExecutor());
    }

    return future;
  }
}
