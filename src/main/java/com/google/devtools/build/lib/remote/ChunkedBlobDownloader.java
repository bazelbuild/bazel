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

import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;

import build.bazel.remote.execution.v2.ChunkingFunction;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.SplitBlobRequest;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.chunking.ChunkingConfig;
import com.google.devtools.build.lib.remote.common.BlobNotSplittableException;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.MaybePathBacked;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestOutputStream;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
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
  private final ChunkLocationMap chunkLocationMap;
  private final ChunkingFunction.Value chunkingFunction;
  private final String remoteInstanceName;
  private final ByteString chunkingParameters;
  private final ImmutableMap<ChunkingFunction.Value, ChunkingConfig> chunkingConfigs;

  ChunkedBlobDownloader(
      GrpcCacheClient grpcCacheClient,
      CombinedCache combinedCache,
      ChunkingConfig chunkingConfig,
      DigestUtil digestUtil,
      ChunkLocationMap chunkLocationMap) {
    this(
        grpcCacheClient,
        combinedCache,
        chunkingConfig,
        digestUtil,
        chunkLocationMap,
        "",
        ByteString.empty(),
        ImmutableMap.of(chunkingConfig.chunkingFunction(), chunkingConfig));
  }

  ChunkedBlobDownloader(
      GrpcCacheClient grpcCacheClient,
      CombinedCache combinedCache,
      ChunkingConfig chunkingConfig,
      DigestUtil digestUtil,
      ChunkLocationMap chunkLocationMap,
      String remoteInstanceName,
      ByteString chunkingParameters,
      ImmutableMap<ChunkingFunction.Value, ChunkingConfig> chunkingConfigs) {
    this.grpcCacheClient = grpcCacheClient;
    this.combinedCache = combinedCache;
    this.digestUtil = digestUtil;
    this.chunkLocationMap = chunkLocationMap;
    this.chunkingFunction = chunkingConfig.chunkingFunction();
    this.remoteInstanceName = remoteInstanceName;
    this.chunkingParameters = chunkingParameters;
    this.chunkingConfigs = chunkingConfigs;
  }

  /**
   * Downloads a blob using chunked download via the SplitBlob API. This should be called with
   * virtual threads, as it may block while waiting for chunk metadata and completed chunk
   * downloads.
   */
  public void downloadChunked(
      RemoteActionExecutionContext context, Digest blobDigest, OutputStream out)
      throws IOException, InterruptedException {
    downloadChunked(context, blobDigest, out, out);
  }

  /**
   * Downloads a blob using chunked download, writing cached-manifest reconstructions to {@code
   * cachedManifestOut}.
   *
   * <p>Returns whether the cached-manifest output was used.
   */
  boolean downloadChunked(
      RemoteActionExecutionContext context,
      Digest blobDigest,
      OutputStream out,
      OutputStream cachedManifestOut)
      throws IOException, InterruptedException {
    ChunkManifest manifest = getChunkManifest(context, blobDigest);
    OutputStream selectedOut = manifest.cached() ? cachedManifestOut : out;
    // Never read chunks from the file currently being written. Record the caller's final path,
    // including when a cached manifest is reconstructed into a temporary file first.
    @Nullable
    Path destination =
        selectedOut instanceof MaybePathBacked pathBacked ? pathBacked.maybeGetPath() : null;
    @Nullable
    Path finalDestination =
        out instanceof MaybePathBacked pathBacked ? pathBacked.maybeGetFinalPath() : null;
    @Nullable DigestOutputStream digestOut = null;
    if (grpcCacheClient.shouldVerifyDownloads()) {
      digestOut = digestUtil.newDigestOutputStream(selectedOut);
      selectedOut = digestOut;
    }

    try {
      new DownloadSession(context, manifest, destination, selectedOut).run();
    } catch (CacheNotFoundException e) {
      if (manifest.cached() && cachedManifestOut != out) {
        // Only the staging stream has been written, so the caller can safely fetch the whole blob.
        throw new BlobNotSplittableException(blobDigest);
      }
      throw e;
    }
    if (digestOut != null) {
      Utils.verifyBlobContents(blobDigest, digestOut.digest());
    }
    if (finalDestination != null) {
      // Locations are hints: until the output is flushed or moved into place, a read is a miss.
      chunkLocationMap.addFile(finalDestination, manifest.chunkDigests());
    }
    return manifest.cached();
  }

  private record ChunkManifest(List<Digest> chunkDigests, boolean cached) {}

  private ChunkManifest getChunkManifest(RemoteActionExecutionContext context, Digest blobDigest)
      throws IOException, InterruptedException {
    if (blobDigest.getSizeBytes() == 0) {
      return new ChunkManifest(ImmutableList.of(), /* cached= */ false);
    }

    Digest manifestKey = getManifestKey(blobDigest);
    SplitBlobResponse splitResponse = combinedCache.downloadSplitBlobManifest(context, manifestKey);
    if (splitResponse != null
        && (splitResponse.getChunkingFunction() == ChunkingFunction.Value.UNKNOWN
            || splitResponse.getChunkingFunction() == chunkingFunction)) {
      try {
        validateChunkDigests(blobDigest, splitResponse);
        if (combinedCache.areBlobsPresentInDiskCache(
            context, splitResponse.getChunkDigestsList())) {
          return new ChunkManifest(splitResponse.getChunkDigestsList(), /* cached= */ true);
        }
      } catch (IOException ignored) {
        // Treat invalid derived metadata as a miss. A successful remote response below overwrites
        // the bad entry.
      }
    }

    ListenableFuture<SplitBlobResponse> splitResponseFuture =
        grpcCacheClient.splitBlob(context, blobDigest, chunkingFunction);
    if (splitResponseFuture == null) {
      throw new BlobNotSplittableException(blobDigest);
    }
    splitResponse = getFromFuture(splitResponseFuture);
    List<Digest> chunkDigests = splitResponse.getChunkDigestsList();
    if (chunkDigests.isEmpty()) {
      throw new BlobNotSplittableException(blobDigest);
    }
    validateChunkDigests(blobDigest, splitResponse);
    ChunkingFunction.Value responseFunction = splitResponse.getChunkingFunction();
    if (responseFunction == ChunkingFunction.Value.UNKNOWN
        || responseFunction == chunkingFunction) {
      combinedCache.uploadSplitBlobManifest(context, manifestKey, splitResponse);
    }
    return new ChunkManifest(chunkDigests, /* cached= */ false);
  }

  private Digest getManifestKey(Digest blobDigest) throws IOException {
    SplitBlobRequest request =
        SplitBlobRequest.newBuilder()
            .setInstanceName(remoteInstanceName)
            .setBlobDigest(blobDigest)
            .setDigestFunction(digestUtil.getDigestFunction())
            .setChunkingFunction(chunkingFunction)
            .build();
    return digestUtil.compute(
        out -> {
          CodedOutputStream coded = CodedOutputStream.newInstance(out);
          coded.writeMessageNoTag(request);
          coded.writeBytesNoTag(chunkingParameters);
          coded.flush();
        });
  }

  private void validateChunkDigests(Digest blobDigest, SplitBlobResponse splitResponse)
      throws IOException {
    List<Digest> chunkDigests = splitResponse.getChunkDigestsList();
    ChunkingFunction.Value responseFunction = splitResponse.getChunkingFunction();
    if (responseFunction == ChunkingFunction.Value.UNKNOWN) {
      responseFunction = chunkingFunction;
    }
    ChunkingConfig responseConfig = chunkingConfigs.get(responseFunction);
    if (responseConfig == null) {
      throw new BlobNotSplittableException(blobDigest);
    }
    long remainingSize = blobDigest.getSizeBytes();
    if (remainingSize < 0) {
      throw new IOException(
          "Invalid SplitBlob response for %s: blob size is negative"
              .formatted(DigestUtil.toString(blobDigest)));
    }
    for (Digest chunkDigest : chunkDigests) {
      long chunkSize = chunkDigest.getSizeBytes();
      try {
        if (chunkDigest.getHash().length() != blobDigest.getHash().length()) {
          throw new IllegalArgumentException();
        }
        var unused = HashCode.fromString(chunkDigest.getHash());
      } catch (IllegalArgumentException e) {
        throw new IOException(
            "Invalid SplitBlob response for %s: chunk digest has an invalid hash"
                .formatted(DigestUtil.toString(blobDigest)),
            e);
      }
      if (chunkSize <= 0) {
        throw new IOException(
            "Invalid SplitBlob response for %s: chunk %s has non-positive size"
                .formatted(DigestUtil.toString(blobDigest), DigestUtil.toString(chunkDigest)));
      }
      if (chunkSize > responseConfig.maxChunkSize()) {
        throw new IOException(
            "Invalid SplitBlob response for %s: chunk %s exceeds max chunk size %d"
                .formatted(
                    DigestUtil.toString(blobDigest),
                    DigestUtil.toString(chunkDigest),
                    responseConfig.maxChunkSize()));
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
    private final ListenableFuture<byte[]> future;
    private final List<Integer> chunkIndices = new ArrayList<>(1);

    PendingDownload(Digest digest, ListenableFuture<byte[]> future, int firstChunkIndex) {
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

    ListenableFuture<byte[]> future() {
      return future;
    }

    List<Integer> chunkIndices() {
      return chunkIndices;
    }
  }

  private final class DownloadSession {
    private final LinkedBlockingQueue<PendingDownload> completedDownloads =
        new LinkedBlockingQueue<>();
    private final Map<Digest, PendingDownload> activeDownloads =
        new HashMap<>(MAX_IN_FLIGHT_CHUNK_DOWNLOADS);
    private final Map<Integer, byte[]> readyChunks = new HashMap<>(MAX_IN_FLIGHT_CHUNK_DOWNLOADS);
    private final RemoteActionExecutionContext context;
    private final List<Digest> chunkDigests;
    private final boolean diskOnly;
    @Nullable private final Path destination;
    private final OutputStream out;
    private int nextToStart = 0;
    private int nextToWrite = 0;

    DownloadSession(
        RemoteActionExecutionContext context,
        ChunkManifest manifest,
        @Nullable Path destination,
        OutputStream out) {
      this.context = context;
      this.chunkDigests = manifest.chunkDigests();
      this.diskOnly = manifest.cached();
      this.destination = destination;
      this.out = out;
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
      PendingDownload download =
          new PendingDownload(
              chunkDigest,
              diskOnly
                  ? combinedCache.downloadBlobFromDisk(context, chunkDigest)
                  : fetchChunk(chunkDigest),
              chunkIndex);
      activeDownloads.put(chunkDigest, download);
      download.future().addListener(() -> completedDownloads.add(download), directExecutor());
    }

    /** Returns the contents of a chunk, preferring a copy already on local disk over a download. */
    private ListenableFuture<byte[]> fetchChunk(Digest chunkDigest) {
      byte[] local = chunkLocationMap.read(chunkDigest, destination, digestUtil);
      return local != null
          ? immediateFuture(local)
          : combinedCache.downloadBlob(context, chunkDigest);
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
      byte[] chunkData = getFromFuture(download.future());
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
