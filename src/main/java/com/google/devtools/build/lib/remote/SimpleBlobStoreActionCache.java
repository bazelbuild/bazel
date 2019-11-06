// Copyright 2016 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashingOutputStream;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.common.SimpleBlobStore;
import com.google.devtools.build.lib.remote.common.SimpleBlobStore.ActionKey;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;

/**
 * A RemoteActionCache implementation that uses a simple blob store for files and action output.
 *
 * <p>The thread safety is guaranteed by the underlying simple blob store.
 *
 * <p>Note that this class is used from src/tools/remote.
 */
@ThreadSafe
public class SimpleBlobStoreActionCache extends AbstractRemoteActionCache {
  protected final SimpleBlobStore blobStore;

  public SimpleBlobStoreActionCache(
      RemoteOptions options, SimpleBlobStore blobStore, DigestUtil digestUtil) {
    super(options, digestUtil);
    this.blobStore = blobStore;
  }

  public void downloadTree(Digest rootDigest, Path rootLocation)
      throws IOException, InterruptedException {
    rootLocation.createDirectoryAndParents();
    Directory directory = Directory.parseFrom(Utils.getFromFuture(downloadBlob(rootDigest)));
    for (FileNode file : directory.getFilesList()) {
      Path dst = rootLocation.getRelative(file.getName());
      Utils.getFromFuture(downloadFile(dst, file.getDigest()));
      dst.setExecutable(file.getIsExecutable());
    }
    for (DirectoryNode child : directory.getDirectoriesList()) {
      downloadTree(child.getDigest(), rootLocation.getRelative(child.getName()));
    }
  }

  @Override
  public ListenableFuture<Void> uploadFile(Digest digest, Path file) {
    return blobStore.uploadFile(digest, file);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(Digest digest, ByteString data) {
    return blobStore.uploadBlob(digest, data);
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
    return Futures.immediateFuture(ImmutableSet.copyOf(digests));
  }

  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey)
      throws IOException, InterruptedException {
    return Utils.getFromFuture(blobStore.downloadActionResult(actionKey));
  }

  public void setCachedActionResult(ActionKey actionKey, ActionResult result)
      throws IOException, InterruptedException {
    blobStore.uploadActionResult(actionKey, result);
  }

  @Override
  public void close() {
    blobStore.close();
  }

  @Override
  protected ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    SettableFuture<Void> outerF = SettableFuture.create();
    @Nullable
    HashingOutputStream hashOut =
        options.remoteVerifyDownloads ? digestUtil.newHashingOutputStream(out) : null;
    Futures.addCallback(
        blobStore.downloadBlob(digest, hashOut != null ? hashOut : out),
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void unused) {
            try {
              if (hashOut != null) {
                verifyContents(digest.getHash(), DigestUtil.hashCodeToString(hashOut.hash()));
              }
              out.flush();
              outerF.set(null);
            } catch (IOException e) {
              outerF.setException(e);
            }
          }

          @Override
          public void onFailure(Throwable throwable) {
            outerF.setException(throwable);
          }
        },
        MoreExecutors.directExecutor());
    return outerF;
  }

  public DigestUtil getDigestUtil() {
    return digestUtil;
  }
}
