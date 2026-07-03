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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.chunking.ChunkingConfig;
import com.google.devtools.build.lib.remote.chunking.ContentDefinedChunker;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.FileChannel;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Uploads blobs in chunks using Content-Defined Chunking.
 *
 * <p>Upload flow for blobs above threshold:
 *
 * <ol>
 *   <li>Chunk file with the configured chunking function
 *   <li>Call findMissingDigests on chunk digests
 *   <li>Upload only missing chunks
 *   <li>Call SpliceBlob to register the blob as the concatenation of chunks
 * </ol>
 */
public class ChunkedBlobUploader {
  // Guard against pathological fanout from a single large chunked blob. This is only a per-blob
  // cap; chunk uploads still flow through CombinedCache and the shared remote cache transport
  // stack below it, which is what bounds active remote RPC concurrency across blobs.
  private static final int MAX_IN_FLIGHT_CHUNK_UPLOADS = 16;

  private final GrpcCacheClient grpcCacheClient;
  private final CombinedCache combinedCache;
  private final ContentDefinedChunker chunker;
  private final long chunkingThreshold;

  /**
   * Creates a new uploader with the given chunking configuration.
   *
   * @param grpcCacheClient client used for {@code FindMissingDigests} and {@code SpliceBlob} RPCs
   * @param combinedCache cache used to upload individual chunks
   * @param config chunking parameters negotiated from server capabilities
   * @param digestUtil utility for computing chunk digests
   */
  public ChunkedBlobUploader(
      GrpcCacheClient grpcCacheClient,
      CombinedCache combinedCache,
      ChunkingConfig config,
      DigestUtil digestUtil) {
    this.grpcCacheClient = grpcCacheClient;
    this.combinedCache = combinedCache;
    this.chunker = config.newChunker(digestUtil);
    this.chunkingThreshold = config.chunkingThreshold();
  }

  /** Returns the minimum blob size for chunked upload. */
  public long getChunkingThreshold() {
    return chunkingThreshold;
  }

  /**
   * Uploads a blob in content-defined chunks. The file is chunked with the configured chunking
   * function, missing chunks are uploaded, and {@code SpliceBlob} is called to register the blob
   * as the concatenation of its chunks.
   */
  public void uploadChunked(RemoteActionExecutionContext context, Digest blobDigest, Path file)
      throws IOException, InterruptedException {
    List<Digest> chunkDigests;
    try (InputStream input = file.getInputStream()) {
      chunkDigests = chunker.chunkToDigests(input);
    }
    if (chunkDigests.isEmpty()) {
      return;
    }

    ImmutableSet<Digest> missingDigests =
        getFromFuture(grpcCacheClient.findMissingDigests(context, chunkDigests));
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
    new UploadSession(context, missingDigests, chunkDigests).run(file);
  }

  private final class UploadSession {
    private final LinkedBlockingQueue<ListenableFuture<Void>> completedUploads =
        new LinkedBlockingQueue<>();
    private final Set<ListenableFuture<Void>> inFlightUploads =
        new HashSet<>(MAX_IN_FLIGHT_CHUNK_UPLOADS);
    private final Set<Digest> scheduledDigests = new HashSet<>();
    private final RemoteActionExecutionContext context;
    private final ImmutableSet<Digest> missingDigests;
    private final List<Digest> chunkDigests;

    UploadSession(
        RemoteActionExecutionContext context,
        ImmutableSet<Digest> missingDigests,
        List<Digest> chunkDigests) {
      this.context = context;
      this.missingDigests = missingDigests;
      this.chunkDigests = chunkDigests;
    }

    void run(Path file) throws IOException, InterruptedException {
      try {
        long offset = 0;
        for (Digest chunkDigest : chunkDigests) {
          drainCompletedUploads();
          long chunkOffset = offset;
          offset += chunkDigest.getSizeBytes();
          if (!shouldScheduleUpload(chunkDigest)) {
            continue;
          }
          if (inFlightUploads.size() >= MAX_IN_FLIGHT_CHUNK_UPLOADS) {
            awaitCompletedUpload();
          }
          startUpload(file, chunkOffset, chunkDigest);
        }
        while (!inFlightUploads.isEmpty()) {
          awaitCompletedUpload();
        }
      } finally {
        cancelAllUploads();
      }
    }

    private boolean shouldScheduleUpload(Digest chunkDigest) {
      return missingDigests.contains(chunkDigest) && scheduledDigests.add(chunkDigest);
    }

    private void startUpload(Path file, long chunkOffset, Digest chunkDigest) {
      ListenableFuture<Void> upload =
          combinedCache.uploadBlob(
              context, chunkDigest, new ChunkBlob(file, chunkOffset, chunkDigest));
      inFlightUploads.add(upload);
      upload.addListener(() -> completedUploads.add(upload), directExecutor());
    }

    private void drainCompletedUploads() throws IOException, InterruptedException {
      while (true) {
        ListenableFuture<Void> upload = completedUploads.poll();
        if (upload == null) {
          return;
        }
        finishUpload(upload);
      }
    }

    private void awaitCompletedUpload() throws IOException, InterruptedException {
      finishUpload(completedUploads.take());
      drainCompletedUploads();
    }

    private void finishUpload(ListenableFuture<Void> upload)
        throws IOException, InterruptedException {
      inFlightUploads.remove(upload);
      getFromFuture(upload);
    }

    private void cancelAllUploads() {
      for (ListenableFuture<Void> upload : inFlightUploads) {
        upload.cancel(/* mayInterruptIfRunning= */ true);
      }
    }
  }

  private static final class ChunkBlob implements Blob {
    private final Path file;
    private final long offset;
    private final Digest digest;

    private ChunkBlob(Path file, long offset, Digest digest) {
      this.file = file;
      this.offset = offset;
      this.digest = digest;
    }

    @Override
    public InputStream get() throws IOException {
      InputStream input = file.getInputStream();
      boolean success = false;
      try {
        seekOrSkip(input, offset);
        InputStream limitedInput = ByteStreams.limit(input, digest.getSizeBytes());
        success = true;
        return limitedInput;
      } catch (EOFException e) {
        throw new IOException("file was concurrently modified during upload: " + file, e);
      } finally {
        if (!success) {
          input.close();
        }
      }
    }

    @Override
    public String description() {
      return "chunk %s at offset %d of file %s"
          .formatted(DigestUtil.toString(digest), offset, file);
    }
  }

  private static void seekOrSkip(InputStream input, long offset) throws IOException {
    if (offset == 0) {
      return;
    }
    if (input instanceof FileInputStream fileInputStream) {
      FileChannel channel = fileInputStream.getChannel();
      if (channel.size() < offset) {
        throw new EOFException();
      }
      channel.position(offset);
      return;
    }
    input.skipNBytes(offset);
  }
}
