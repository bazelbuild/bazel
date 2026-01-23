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
import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.ExecutionException;
import java.util.List;

/**
 * Downloads blobs by sequentially fetching chunks via the SplitBlob API.
 */
public class ChunkedBlobDownloader {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final GrpcCacheClient grpcCacheClient;

  public ChunkedBlobDownloader(GrpcCacheClient grpcCacheClient) {
    this.grpcCacheClient = grpcCacheClient;
  }

  /**
   * Downloads a blob using chunked download via the SplitBlob API. This should be
   * called with virtual threads.
   */
  public void downloadChunked(
      RemoteActionExecutionContext context, Digest blobDigest, OutputStream out)
      throws CacheNotFoundException, InterruptedException {
    try {
      doDownloadChunked(context, blobDigest, out);
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Chunked download failed for %s", blobDigest.getHash());
      throw new CacheNotFoundException(blobDigest);
    }
  }

  private void doDownloadChunked(
      RemoteActionExecutionContext context, Digest blobDigest, OutputStream out)
      throws IOException, InterruptedException {
    ListenableFuture<SplitBlobResponse> splitResponseFuture = grpcCacheClient.splitBlob(context, blobDigest);
    if (splitResponseFuture == null) {
      throw new CacheNotFoundException(blobDigest);
    }
    downloadAndReassembleChunks(context, getFromFuture(splitResponseFuture).getChunkDigestsList(), out);
  }

  private void downloadAndReassembleChunks(
      RemoteActionExecutionContext context, List<Digest> chunkDigests, OutputStream out)
      throws IOException, InterruptedException {
    for (Digest chunkDigest : chunkDigests) {
      getFromFuture(grpcCacheClient.downloadBlob(context, chunkDigest, out));
    }
  }
}
