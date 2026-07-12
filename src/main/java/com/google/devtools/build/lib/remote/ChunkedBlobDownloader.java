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
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.metrics.RemoteCacheCdcEvent;
import com.google.devtools.build.lib.metrics.RemoteCacheCdcEvent.Download;
import com.google.devtools.build.lib.metrics.RemoteCacheCdcEvent.Outcome;
import com.google.devtools.build.lib.remote.CombinedCache.CdcChunk;
import com.google.devtools.build.lib.remote.chunking.ChunkingConfig;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestOutputStream;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** Downloads blobs by fetching chunks through a per-blob sliding window via the SplitBlob API. */
public class ChunkedBlobDownloader {
  // Guard against pathological fanout from a single large chunked blob. This is only a per-blob
  // cap; chunk requests still flow through CombinedCache and the shared remote cache transport
  // stack below it, which is what bounds active remote RPC concurrency across blobs.
  private static final int MAX_IN_FLIGHT_CHUNK_DOWNLOADS = 16;

  private final GrpcCacheClient grpcCacheClient;
  private final CombinedCache combinedCache;
  private final DigestUtil digestUtil;
  private final long maxChunkSize;
  @Nullable private final Consumer<RemoteCacheCdcEvent> metricsSink;

  public ChunkedBlobDownloader(
      GrpcCacheClient grpcCacheClient,
      CombinedCache combinedCache,
      ChunkingConfig chunkingConfig,
      DigestUtil digestUtil) {
    this(grpcCacheClient, combinedCache, chunkingConfig, digestUtil, /* metricsSink= */ null);
  }

  public ChunkedBlobDownloader(
      GrpcCacheClient grpcCacheClient,
      CombinedCache combinedCache,
      ChunkingConfig chunkingConfig,
      DigestUtil digestUtil,
      @Nullable Consumer<RemoteCacheCdcEvent> metricsSink) {
    this.grpcCacheClient = grpcCacheClient;
    this.combinedCache = combinedCache;
    this.digestUtil = digestUtil;
    this.maxChunkSize = chunkingConfig.maxChunkSize();
    this.metricsSink = metricsSink;
  }

  /**
   * Downloads a blob using chunked download via the SplitBlob API. This should be called with
   * virtual threads, as it may block while waiting for chunk metadata and completed chunk
   * downloads.
   */
  public void downloadChunked(
      RemoteActionExecutionContext context, Digest blobDigest, OutputStream out)
      throws IOException, InterruptedException {
    List<Digest> chunkDigests = List.of();
    DownloadStats downloadStats = new DownloadStats();
    Outcome outcome = Outcome.FAILURE;
    try {
      @Nullable DigestOutputStream digestOut = null;
      if (grpcCacheClient.shouldVerifyDownloads()) {
        digestOut = digestUtil.newDigestOutputStream(out);
        out = digestOut;
      }

      chunkDigests = getChunkDigests(context, blobDigest);
      downloadAndReassembleChunks(context, chunkDigests, out, downloadStats);
      if (digestOut != null) {
        Utils.verifyBlobContents(blobDigest, digestOut.digest());
      }
      outcome = Outcome.SUCCESS;
    } catch (CacheNotFoundException e) {
      outcome = Outcome.FALLBACK;
      throw e;
    } finally {
      if (metricsSink != null) {
        metricsSink.accept(
            new Download(
                outcome,
                blobDigest.getSizeBytes(),
                chunkDigests.size(),
                downloadStats.diskCacheHitChunks,
                downloadStats.diskCacheHitBytes,
                downloadStats.diskCacheMissChunks,
                downloadStats.diskCacheMissBytes,
                downloadStats.remoteChunks,
                downloadStats.remoteBytes));
      }
    }
  }

  private static final class DownloadStats {
    private long diskCacheHitChunks;
    private long diskCacheHitBytes;
    private long diskCacheMissChunks;
    private long diskCacheMissBytes;
    private long remoteChunks;
    private long remoteBytes;

    void record(CdcChunk chunk) {
      if (chunk.diskCacheHit()) {
        diskCacheHitChunks++;
        diskCacheHitBytes += chunk.data().length;
      } else {
        if (chunk.diskCacheLookupAttempted()) {
          diskCacheMissChunks++;
          diskCacheMissBytes += chunk.data().length;
        }
        remoteChunks++;
        remoteBytes += chunk.data().length;
      }
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
    validateChunkDigests(blobDigest, chunkDigests);
    return chunkDigests;
  }

  private void validateChunkDigests(Digest blobDigest, List<Digest> chunkDigests)
      throws IOException {
    long remainingSize = blobDigest.getSizeBytes();
    if (remainingSize < 0) {
      throw new IOException(
          "Invalid SplitBlob response for %s: blob size is negative"
              .formatted(DigestUtil.toString(blobDigest)));
    }
    for (Digest chunkDigest : chunkDigests) {
      long chunkSize = chunkDigest.getSizeBytes();
      if (chunkSize <= 0) {
        throw new IOException(
            "Invalid SplitBlob response for %s: chunk %s has non-positive size"
                .formatted(DigestUtil.toString(blobDigest), DigestUtil.toString(chunkDigest)));
      }
      if (chunkSize > maxChunkSize) {
        throw new IOException(
            "Invalid SplitBlob response for %s: chunk %s exceeds max chunk size %d"
                .formatted(
                    DigestUtil.toString(blobDigest),
                    DigestUtil.toString(chunkDigest),
                    maxChunkSize));
      }
      if (chunkSize > remainingSize) {
        throw new IOException(
            "Invalid SplitBlob response for %s: chunk sizes exceed blob size"
                .formatted(DigestUtil.toString(blobDigest)));
      }
      remainingSize -= chunkSize;
    }
    if (remainingSize != 0) {
      throw new IOException(
          "Invalid SplitBlob response for %s: chunk sizes do not match blob size"
              .formatted(DigestUtil.toString(blobDigest)));
    }
  }

  private static final class PendingDownload {
    private final Digest digest;
    private final ListenableFuture<CdcChunk> future;
    private final List<Integer> chunkIndices = new ArrayList<>(1);

    PendingDownload(Digest digest, ListenableFuture<CdcChunk> future, int firstChunkIndex) {
      this.digest = digest;
      this.future = future;
      chunkIndices.add(firstChunkIndex);
    }

    void addChunkIndex(int chunkIndex) {
      chunkIndices.add(chunkIndex);
    }

    Digest digest() {
      return digest;
    }

    ListenableFuture<CdcChunk> future() {
      return future;
    }

    List<Integer> chunkIndices() {
      return chunkIndices;
    }
  }

  private void downloadAndReassembleChunks(
      RemoteActionExecutionContext context,
      List<Digest> chunkDigests,
      OutputStream out,
      DownloadStats downloadStats)
      throws IOException, InterruptedException {
    new DownloadSession(context, chunkDigests, out, downloadStats).run();
  }

  private final class DownloadSession {
    private final LinkedBlockingQueue<PendingDownload> completedDownloads =
        new LinkedBlockingQueue<>();
    private final Map<Digest, PendingDownload> activeDownloads =
        new HashMap<>(MAX_IN_FLIGHT_CHUNK_DOWNLOADS);
    private final Map<Integer, byte[]> readyChunks = new HashMap<>(MAX_IN_FLIGHT_CHUNK_DOWNLOADS);
    private final RemoteActionExecutionContext context;
    private final List<Digest> chunkDigests;
    private final OutputStream out;
    private final DownloadStats downloadStats;
    private int nextToStart = 0;
    private int nextToWrite = 0;

    DownloadSession(
        RemoteActionExecutionContext context,
        List<Digest> chunkDigests,
        OutputStream out,
        DownloadStats downloadStats) {
      this.context = context;
      this.chunkDigests = chunkDigests;
      this.out = out;
      this.downloadStats = downloadStats;
    }

    void run() throws IOException, InterruptedException {
      try {
        fillWindow();
        while (nextToWrite < chunkDigests.size()) {
          drainCompletedDownloads();
          drainReadyChunks();
          fillWindow();
        }
      } finally {
        cancelAllDownloads();
      }
    }

    private void fillWindow() {
      while (nextToStart < chunkDigests.size()) {
        if (nextToStart - nextToWrite >= MAX_IN_FLIGHT_CHUNK_DOWNLOADS) {
          return;
        }
        Digest chunkDigest = chunkDigests.get(nextToStart);
        PendingDownload existing = activeDownloads.get(chunkDigest);
        if (existing != null) {
          existing.addChunkIndex(nextToStart);
          nextToStart++;
          continue;
        }
        startDownload(chunkDigest, nextToStart);
        nextToStart++;
      }
    }

    private void startDownload(Digest chunkDigest, int chunkIndex) {
      ListenableFuture<CdcChunk> future;
      if (metricsSink == null) {
        future =
            Futures.transform(
                combinedCache.downloadBlob(context, chunkDigest),
                data ->
                    new CdcChunk(
                        data,
                        /* diskCacheHit= */ false,
                        /* diskCacheLookupAttempted= */ false),
                directExecutor());
      } else {
        future = combinedCache.downloadCdcChunk(context, chunkDigest);
      }
      PendingDownload download =
          new PendingDownload(chunkDigest, future, chunkIndex);
      activeDownloads.put(chunkDigest, download);
      download.future().addListener(() -> completedDownloads.add(download), directExecutor());
    }

    private void drainCompletedDownloads() throws IOException, InterruptedException {
      PendingDownload download = completedDownloads.take();
      do {
        processCompletedDownload(download);
        download = completedDownloads.poll();
      } while (download != null);
    }

    private void processCompletedDownload(PendingDownload download)
        throws IOException, InterruptedException {
      activeDownloads.remove(download.digest());
      CdcChunk chunk = getFromFuture(download.future());
      downloadStats.record(chunk);
      byte[] chunkData = chunk.data();
      for (int chunkIndex : download.chunkIndices()) {
        if (chunkIndex == nextToWrite) {
          out.write(chunkData);
          nextToWrite++;
        } else {
          readyChunks.put(chunkIndex, chunkData);
        }
      }
    }

    private void drainReadyChunks() throws IOException {
      while (true) {
        byte[] chunk = readyChunks.remove(nextToWrite);
        if (chunk == null) {
          return;
        }
        out.write(chunk);
        nextToWrite++;
      }
    }

    private void cancelAllDownloads() {
      for (PendingDownload download : activeDownloads.values()) {
        download.future().cancel(/* mayInterruptIfRunning= */ true);
      }
    }
  }
}
