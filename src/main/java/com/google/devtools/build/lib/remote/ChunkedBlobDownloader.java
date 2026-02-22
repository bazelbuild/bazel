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
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

/**
 * Downloads blobs by sequentially fetching chunks via the SplitBlob API.
 */
public class ChunkedBlobDownloader {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final GrpcCacheClient grpcCacheClient;
  private final CombinedCache combinedCache;

  public ChunkedBlobDownloader(GrpcCacheClient grpcCacheClient, CombinedCache combinedCache) {
    this.grpcCacheClient = grpcCacheClient;
    this.combinedCache = combinedCache;
  }

  /**
   * Downloads a blob using chunked download via the SplitBlob API. This should be called with
   * virtual threads, as it blocks on futures via {@link
   * com.google.devtools.build.lib.remote.util.Utils#getFromFuture}.
   */
  public void downloadChunked(
      RemoteActionExecutionContext context, Digest blobDigest, OutputStream out)
      throws CacheNotFoundException, IOException, InterruptedException {
    List<Digest> chunkDigests;
    try {
      chunkDigests = getChunkDigests(context, blobDigest);
    } catch (IOException | StatusRuntimeException e) {
      logger.atWarning().withCause(e).log(
          "SplitBlob failed for %s/%d", blobDigest.getHash(), blobDigest.getSizeBytes());
      throw new CacheNotFoundException(blobDigest);
    }
    downloadAndReassembleChunks(context, chunkDigests, out);
  }

  private List<Digest> getChunkDigests(
      RemoteActionExecutionContext context, Digest blobDigest)
      throws IOException, InterruptedException {
    ListenableFuture<SplitBlobResponse> splitResponseFuture =
        grpcCacheClient.splitBlob(context, blobDigest);
    if (splitResponseFuture == null) {
      throw new CacheNotFoundException(blobDigest);
    }
    List<Digest> chunkDigests = getFromFuture(splitResponseFuture).getChunkDigestsList();
    if (chunkDigests.isEmpty() && blobDigest.getSizeBytes() > 0) {
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
