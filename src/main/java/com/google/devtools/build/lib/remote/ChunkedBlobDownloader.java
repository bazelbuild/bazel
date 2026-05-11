// Copyright 2026 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestOutputStream;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;
import javax.annotation.Nullable;

/** Downloads blobs by sequentially fetching chunks via the SplitBlob API. */
public class ChunkedBlobDownloader {
  private final GrpcCacheClient grpcCacheClient;
  private final CombinedCache combinedCache;
  private final DigestUtil digestUtil;

  public ChunkedBlobDownloader(
      GrpcCacheClient grpcCacheClient, CombinedCache combinedCache, DigestUtil digestUtil) {
    this.grpcCacheClient = grpcCacheClient;
    this.combinedCache = combinedCache;
    this.digestUtil = digestUtil;
  }

  /**
   * Downloads a blob using chunked download via the SplitBlob API. This should be called with
   * virtual threads, as it blocks on futures via {@link
   * com.google.devtools.build.lib.remote.util.Utils#getFromFuture}.
   */
  public void downloadChunked(
      RemoteActionExecutionContext context, Digest blobDigest, OutputStream out)
      throws IOException, InterruptedException {
    @Nullable DigestOutputStream digestOut = null;
    if (grpcCacheClient.shouldVerifyDownloads()) {
      digestOut = digestUtil.newDigestOutputStream(out);
      out = digestOut;
    }

    List<Digest> chunkDigests = getChunkDigests(context, blobDigest);
    downloadAndReassembleChunks(context, chunkDigests, out);
    if (digestOut != null) {
      Utils.verifyBlobContents(blobDigest, digestOut.digest());
    }
  }

  private List<Digest> getChunkDigests(RemoteActionExecutionContext context, Digest blobDigest)
      throws IOException, InterruptedException {
    if (blobDigest.getSizeBytes() == 0) {
      return ImmutableList.of();
    }
    ListenableFuture<SplitBlobResponse> splitResponseFuture =
        grpcCacheClient.splitBlob(context, blobDigest);
    if (splitResponseFuture == null) {
      throw new CacheNotFoundException(blobDigest);
    }
    List<Digest> chunkDigests = getFromFuture(splitResponseFuture).getChunkDigestsList();
    if (chunkDigests.isEmpty()) {
      throw new CacheNotFoundException(blobDigest);
    }
    return chunkDigests;
  }

  private void downloadAndReassembleChunks(
      RemoteActionExecutionContext context, List<Digest> chunkDigests, OutputStream out)
      throws IOException, InterruptedException {
    for (Digest chunkDigest : chunkDigests) {
      getFromFuture(combinedCache.downloadBlob(context, chunkDigest, out));
    }
  }
}
