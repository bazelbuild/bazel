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

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

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
  public void uploadActionResult(ActionKey actionKey, ActionResult actionResult)
      throws IOException, InterruptedException {
    diskCache.uploadActionResult(actionKey, actionResult);
    if (!options.incompatibleRemoteResultsIgnoreDisk || options.remoteUploadLocalResults) {
      remoteCache.uploadActionResult(actionKey, actionResult);
    }
  }

  @Override
  public void close() {
    diskCache.close();
    remoteCache.close();
  }

  @Override
  public ListenableFuture<Void> uploadFile(Digest digest, Path file) {
    try {
      diskCache.uploadFile(digest, file).get();
      if (!options.incompatibleRemoteResultsIgnoreDisk || options.remoteUploadLocalResults) {
        remoteCache.uploadFile(digest, file).get();
      }
    } catch (ExecutionException e) {
      return Futures.immediateFailedFuture(e.getCause());
    } catch (InterruptedException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(Digest digest, ByteString data) {
    try {
      diskCache.uploadBlob(digest, data).get();
      if (!options.incompatibleRemoteResultsIgnoreDisk || options.remoteUploadLocalResults) {
        remoteCache.uploadBlob(digest, data).get();
      }
    } catch (ExecutionException e) {
      return Futures.immediateFailedFuture(e.getCause());
    } catch (InterruptedException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
    ListenableFuture<ImmutableSet<Digest>> remoteQuery = remoteCache.findMissingDigests(digests);
    ListenableFuture<ImmutableSet<Digest>> diskQuery = diskCache.findMissingDigests(digests);
    return Futures.whenAllSucceed(remoteQuery, diskQuery)
        .call(
            () ->
                ImmutableSet.<Digest>builder()
                    .addAll(remoteQuery.get())
                    .addAll(diskQuery.get())
                    .build(),
            MoreExecutors.directExecutor());
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
        MoreExecutors.directExecutor());
  }

  @Override
  public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    if (diskCache.contains(digest)) {
      return diskCache.downloadBlob(digest, out);
    }

    Path tempPath = newTempPath();
    final OutputStream tempOut;
    try {
      tempOut = tempPath.getOutputStream();
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }

    if (!options.incompatibleRemoteResultsIgnoreDisk || options.remoteAcceptCached) {
      ListenableFuture<Void> download =
          closeStreamOnError(remoteCache.downloadBlob(digest, tempOut), tempOut);
      ListenableFuture<Void> saveToDiskAndTarget =
          Futures.transformAsync(
              download,
              (unused) -> {
                try {
                  tempOut.close();
                  diskCache.captureFile(tempPath, digest, /* isActionCache= */ false);
                } catch (IOException e) {
                  return Futures.immediateFailedFuture(e);
                }
                return diskCache.downloadBlob(digest, out);
              },
              MoreExecutors.directExecutor());
      return saveToDiskAndTarget;
    } else {
      return Futures.immediateFuture(null);
    }
  }

  @Override
  public ListenableFuture<ActionResult> downloadActionResult(
      ActionKey actionKey, boolean inlineOutErr) {
    if (diskCache.containsActionResult(actionKey)) {
      return diskCache.downloadActionResult(actionKey, inlineOutErr);
    }

    if (!options.incompatibleRemoteResultsIgnoreDisk || options.remoteAcceptCached) {
      return Futures.transformAsync(
          remoteCache.downloadActionResult(actionKey, inlineOutErr),
          (actionResult) -> {
            if (actionResult == null) {
              return Futures.immediateFuture(null);
            } else {
              diskCache.uploadActionResult(actionKey, actionResult);
              return Futures.immediateFuture(actionResult);
            }
          },
          MoreExecutors.directExecutor());
    } else {
      return Futures.immediateFuture(null);
    }
  }
}
