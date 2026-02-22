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
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.remote.chunking.ChunkingConfig;
import com.google.devtools.build.lib.remote.chunking.FastCDCChunker;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Uploads blobs in chunks using Content-Defined Chunking with FastCDC 2020.
 *
 * <p>
 * Upload flow for blobs above threshold:
 *
 * <ol>
 * <li>Chunk file with FastCDC
 * <li>Call findMissingDigests on chunk digests
 * <li>Upload only missing chunks
 * <li>Call SpliceBlob to register the blob as the concatenation of chunks
 * </ol>
 */
public class ChunkedBlobUploader {

  private final GrpcCacheClient grpcCacheClient;
  private final CombinedCache combinedCache;
  private final FastCDCChunker chunker;
  private final long chunkingThreshold;

  public ChunkedBlobUploader(
      GrpcCacheClient grpcCacheClient,
      CombinedCache combinedCache,
      ChunkingConfig config,
      DigestUtil digestUtil) {
    this.grpcCacheClient = grpcCacheClient;
    this.combinedCache = combinedCache;
    this.chunker = new FastCDCChunker(config, digestUtil);
    this.chunkingThreshold = config.chunkingThreshold();
  }

  public long getChunkingThreshold() {
    return chunkingThreshold;
  }

  public void uploadChunked(RemoteActionExecutionContext context, Digest blobDigest, Path file)
      throws IOException, InterruptedException {
    List<Digest> chunkDigests;
    try (InputStream input = file.getInputStream()) {
      chunkDigests = chunker.chunkToDigests(input);
    }
    if (chunkDigests.isEmpty()) {
      return;
    }

    ImmutableSet<Digest> missingDigests = getFromFuture(grpcCacheClient.findMissingDigests(context, chunkDigests));
    uploadMissingChunks(context, missingDigests, chunkDigests, file);
    getFromFuture(grpcCacheClient.spliceBlob(context, blobDigest, chunkDigests));
  }

  private void uploadMissingChunks(
      RemoteActionExecutionContext context,
      ImmutableSet<Digest> missingDigests,
      List<Digest> chunkDigests,
      Path file)
      throws IOException, InterruptedException {
    if (missingDigests.isEmpty()) {
      return;
    }

    Set<Digest> uploaded = new HashSet<>();
    try (InputStream input = file.getInputStream()) {
      for (Digest chunkDigest : chunkDigests) {
        if (missingDigests.contains(chunkDigest) && uploaded.add(chunkDigest)) {
          ByteString.Output out = ByteString.newOutput((int) chunkDigest.getSizeBytes());
          ByteStreams.limit(input, chunkDigest.getSizeBytes()).transferTo(out);
          getFromFuture(combinedCache.uploadBlob(context, chunkDigest, out.toByteString()));
        } else {
          input.skipNBytes(chunkDigest.getSizeBytes());
        }
      }
    }
  }
}
