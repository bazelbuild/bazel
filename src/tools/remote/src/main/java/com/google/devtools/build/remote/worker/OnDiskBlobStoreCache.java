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
package com.google.devtools.build.remote.worker;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.ActionCacheUpdateCapabilities;
import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkAbsolutePathStrategy;
import build.bazel.remote.execution.v2.SymlinkNode;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.CombinedCache;
import com.google.devtools.build.lib.remote.Store;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/** A {@link CombinedCache} backed by an {@link DiskCacheClient}. */
class OnDiskBlobStoreCache extends CombinedCache {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final ExecutorService executorService =
      MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(1));

  private final RemoteWorkerOptions remoteWorkerOptions;
  private final ConcurrentHashMap<String, AtomicInteger> numberOfDownloadsPerDigestAndInvocation =
      new ConcurrentHashMap<>();

  public OnDiskBlobStoreCache(
      RemoteOptions options, Path cacheDir, DigestUtil digestUtil, RemoteWorkerOptions remoteWorkerOptions)
      throws IOException {
    super(
        /* remoteCacheClient= */ null,
        new DiskCacheClient(cacheDir, digestUtil, executorService, /* verifyDownloads= */ true),
        options,
        digestUtil);
    this.remoteWorkerOptions = remoteWorkerOptions;
  }

  @Override
  public CacheCapabilities getRemoteCacheCapabilities() {
    return CacheCapabilities.newBuilder()
        .setActionCacheUpdateCapabilities(
            ActionCacheUpdateCapabilities.newBuilder().setUpdateEnabled(true).build())
        .setSymlinkAbsolutePathStrategy(SymlinkAbsolutePathStrategy.Value.ALLOWED)
        .build();
  }

  /** If the given blob exists, updates its mtime and returns true. Otherwise, returns false. */
  boolean refresh(Digest digest) throws IOException {
    return diskCacheClient.refresh(diskCacheClient.toPath(digest, Store.CAS));
  }

  @SuppressWarnings("ProtoParseWithRegistry")
  public void downloadTree(
      RemoteActionExecutionContext context, Digest rootDigest, Path rootLocation)
      throws IOException, InterruptedException {
    rootLocation.createDirectoryAndParents();
    Directory directory = Directory.parseFrom(getFromFuture(downloadBlob(context, rootDigest)));
    for (FileNode file : directory.getFilesList()) {
      Path dst = rootLocation.getRelative(file.getName());
      getFromFuture(downloadFile(context, dst, file.getDigest()));
      dst.setExecutable(file.getIsExecutable());
    }
    for (SymlinkNode symlink : directory.getSymlinksList()) {
      Path dst = rootLocation.getRelative(symlink.getName());
      // TODO(fmeum): The following line is not generally correct: The remote execution API allows
      //  for non-normalized symlink targets, but the normalization applied by PathFragment.create
      //  does not take directory symlinks into account. However, Bazel's file system API does not
      //  currently offer a way to specify a raw String as a symlink target.
      // https://github.com/bazelbuild/bazel/issues/14224
      dst.createSymbolicLink(PathFragment.create(symlink.getTarget()));
    }
    for (DirectoryNode child : directory.getDirectoriesList()) {
      downloadTree(context, child.getDigest(), rootLocation.getRelative(child.getName()));
    }
  }

  @Override
  public ListenableFuture<byte[]> downloadBlob(RemoteActionExecutionContext context, Digest digest) {
    logger.atFine().log("downloadBlob digest=%s invocationId=%s",
        digest.getHash(), context.getRequestMetadata().getToolInvocationId());

    if (remoteWorkerOptions.fakeErrorForDuplicatedDownloads) {
      // Only populate numberOfDownloadsPerDigestAndInvocation when fakeErrorForDuplicatedDownloads is enabled
      // to avoid unnecessary unbounded memory growth.
      String uniqueDownloadKey = String.format("%s_%s",
          digest.getHash(), context.getRequestMetadata().getToolInvocationId());
      int numberOfDownloads = numberOfDownloadsPerDigestAndInvocation.computeIfAbsent(
          uniqueDownloadKey,
          d -> new AtomicInteger()).incrementAndGet();
      if (numberOfDownloads > 1) {
        return Futures.immediateFailedFuture(new IOException(
            String.format("Faked error for duplicated download of blob digest %s with invocation id %s",
                digest, context.getRequestMetadata().getToolInvocationId())));
      }
    }

    return super.downloadBlob(context, digest);
  }

  public DigestUtil getDigestUtil() {
    return digestUtil;
  }

  public RemoteOptions getRemoteOptions() {
    return options;
  }

  @Override
  public ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, ByteString data) {
    return uploadBlob(context, digest, data, /* force= */ true);
  }

  @Override
  public ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context, Digest digest, Path file) {
    return uploadFile(context, digest, file, /* force= */ true);
  }

  public DiskCacheClient getDiskCacheClient() {
    return checkNotNull(diskCacheClient);
  }
}
