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

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.chunking.ChunkingConfig;
import com.google.devtools.build.lib.remote.chunking.FastCdcChunker;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/** Benchmark for chunk download/upload with per-chunk latency jitter. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 1, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 3, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class ChunkedTransferBenchmark {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final RemoteActionExecutionContext CONTEXT =
      RemoteActionExecutionContext.create(RequestMetadata.getDefaultInstance());

  @Benchmark
  public void downloadChunked(DownloadState state) throws Exception {
    state.downloader.downloadChunked(CONTEXT, state.blobDigest, OutputStream.nullOutputStream());
  }

  @Benchmark
  public void uploadChunked(UploadState state) throws Exception {
    state.uploader.uploadChunked(CONTEXT, state.blobDigest, state.file);
  }

  @State(Scope.Thread)
  public static class DownloadState {
    @Param({"1", "2", "4", "8"})
    public int schedulerThreads;

    @Param({"32"})
    public int chunkCount;

    @Param({"1024"})
    public int chunkSizeBytes;

    @Param({"25"})
    public int delayMillis;

    @Param({"10"})
    public int jitterMillis;

    private ScheduledExecutorService scheduler;
    private ChunkedBlobDownloader downloader;
    private Digest blobDigest;
    private Random latencyJitter;

    @Setup(Level.Trial)
    public void setup() throws Exception {
      scheduler = Executors.newScheduledThreadPool(schedulerThreads);
      latencyJitter = new Random(12345L);

      GrpcCacheClient grpcCacheClient = mock(GrpcCacheClient.class);
      CombinedCache combinedCache = mock(CombinedCache.class);

      List<Digest> chunkDigests = new ArrayList<>(chunkCount);
      Map<Digest, byte[]> chunkDataByDigest = new HashMap<>(chunkCount);
      long totalBytes = 0;
      for (int i = 0; i < chunkCount; i++) {
        byte[] chunkData = new byte[chunkSizeBytes];
        new Random(i).nextBytes(chunkData);
        Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
        chunkDigests.add(chunkDigest);
        chunkDataByDigest.put(chunkDigest, chunkData);
        totalBytes += chunkData.length;
      }

      when(combinedCache.downloadBlob(any(), any(Digest.class)))
          .thenAnswer(
              invocation ->
                  delayedFuture(
                      chunkDataByDigest.get(invocation.getArgument(1)),
                      delayMillis,
                      jitterMillis,
                      latencyJitter,
                      scheduler));

      blobDigest =
          Digest.newBuilder()
              .setHash("chunked-transfer-benchmark-download-" + chunkCount + "-" + chunkSizeBytes)
              .setSizeBytes(totalBytes)
              .build();

      SplitBlobResponse splitBlobResponse =
          SplitBlobResponse.newBuilder().addAllChunkDigests(chunkDigests).build();
      when(grpcCacheClient.splitBlob(any(), any(Digest.class)))
          .thenReturn(Futures.immediateFuture(splitBlobResponse));

      downloader = new ChunkedBlobDownloader(grpcCacheClient, combinedCache, DIGEST_UTIL);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
      scheduler.shutdownNow();
    }
  }

  @State(Scope.Thread)
  public static class UploadState {
    @Param({"1", "2", "4", "8"})
    public int schedulerThreads;

    @Param({"32768"})
    public int fileSizeBytes;

    @Param({"1024"})
    public int avgChunkSizeBytes;

    @Param({"25"})
    public int delayMillis;

    @Param({"10"})
    public int jitterMillis;

    private ScheduledExecutorService scheduler;
    private ChunkedBlobUploader uploader;
    private Path file;
    private Digest blobDigest;
    private Random latencyJitter;

    @Setup(Level.Trial)
    public void setup() throws Exception {
      scheduler = Executors.newScheduledThreadPool(schedulerThreads);
      latencyJitter = new Random(54321L);

      GrpcCacheClient grpcCacheClient = mock(GrpcCacheClient.class);
      CombinedCache combinedCache = mock(CombinedCache.class);

      byte[] data = new byte[fileSizeBytes];
      new Random(42).nextBytes(data);
      blobDigest = DIGEST_UTIL.compute(data);

      FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
      file = fs.getPath("/bench/blob.bin");
      file.getParentDirectory().createDirectoryAndParents();
      try (var out = file.getOutputStream()) {
        out.write(data);
      }

      ChunkingConfig chunkingConfig = new ChunkingConfig(avgChunkSizeBytes, 2, 0);
      uploader =
          new ChunkedBlobUploader(grpcCacheClient, combinedCache, chunkingConfig, DIGEST_UTIL);

      List<Digest> chunkDigests;
      try (var input = file.getInputStream()) {
        chunkDigests = new FastCdcChunker(chunkingConfig, DIGEST_UTIL).chunkToDigests(input);
      }

      when(grpcCacheClient.findMissingDigests(any(), any()))
          .thenReturn(Futures.immediateFuture(ImmutableSet.copyOf(chunkDigests)));
      when(grpcCacheClient.spliceBlob(any(), any(Digest.class), any()))
          .thenReturn(Futures.immediateVoidFuture());
      when(combinedCache.uploadBlob(any(), any(Digest.class), any(Blob.class)))
          .thenAnswer(
              invocation ->
                  delayedFuture(null, delayMillis, jitterMillis, latencyJitter, scheduler));
    }

    @TearDown(Level.Trial)
    public void tearDown() {
      scheduler.shutdownNow();
    }
  }

  private static <T> ListenableFuture<T> delayedFuture(
      T value,
      int delayMillis,
      int jitterMillis,
      Random latencyJitter,
      ScheduledExecutorService scheduler) {
    SettableFuture<T> future = SettableFuture.create();
    var unused =
        scheduler.schedule(
            () -> future.set(value),
            jitteredDelayMillis(delayMillis, jitterMillis, latencyJitter),
            TimeUnit.MILLISECONDS);
    return future;
  }

  private static int jitteredDelayMillis(int delayMillis, int jitterMillis, Random latencyJitter) {
    if (jitterMillis == 0) {
      return delayMillis;
    }
    return Math.max(0, delayMillis + latencyJitter.nextInt((jitterMillis * 2) + 1) - jitterMillis);
  }
}
