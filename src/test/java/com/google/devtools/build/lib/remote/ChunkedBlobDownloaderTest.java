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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ChunkingFunction;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.chunking.ChunkingConfig;
import com.google.devtools.build.lib.remote.chunking.FastCdcChunkingConfig;
import com.google.devtools.build.lib.remote.chunking.RepMaxCdcChunkingConfig;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests for {@link ChunkedBlobDownloader}. */
@RunWith(JUnit4.class)
public class ChunkedBlobDownloaderTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final ChunkingConfig CHUNKING_CONFIG =
      new FastCdcChunkingConfig(
          /* avgChunkSize= */ 1024, /* normalizationLevel= */ 2, /* seed= */ 0);
  private static final int MAX_IN_FLIGHT_CHUNK_DOWNLOADS = 16;

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private GrpcCacheClient grpcCacheClient;
  @Mock private CombinedCache combinedCache;
  @Mock private RemoteActionExecutionContext context;

  private ChunkedBlobDownloader downloader;

  @Before
  public void setUp() {
    when(grpcCacheClient.shouldVerifyDownloads()).thenReturn(true);
    downloader =
        new ChunkedBlobDownloader(grpcCacheClient, combinedCache, CHUNKING_CONFIG, DIGEST_UTIL);
  }

  @Test
  public void downloadChunked_splitBlobReturnsNull_throwsCacheNotFound() {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {1, 2, 3});
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any())).thenReturn(null);

    assertThrows(
        CacheNotFoundException.class,
        () -> downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream()));
  }

  @Test
  public void downloadChunked_singleChunk_downloadsAndReassembles() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream cachedManifestOut = new ByteArrayOutputStream();
    boolean usedCachedManifest =
        downloader.downloadChunked(context, blobDigest, out, cachedManifestOut);

    assertThat(usedCachedManifest).isFalse();
    assertThat(out.toByteArray()).isEqualTo(chunkData);
    assertThat(cachedManifestOut.toByteArray()).isEmpty();
    verify(combinedCache).uploadSplitBlobManifest(eq(context), any(), eq(splitResponse));
  }

  @Test
  public void downloadChunked_responseUsesDifferentFunction_doesNotCacheManifest()
      throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    ChunkingConfig repMaxCdcConfig =
        new RepMaxCdcChunkingConfig(/* minChunkSize= */ 1024, /* horizonSize= */ 8192);
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(chunkDigest)
            .setChunkingFunction(ChunkingFunction.Value.REP_MAX_CDC)
            .build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), eq(ChunkingFunction.Value.FAST_CDC_2020)))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));
    ChunkedBlobDownloader downloader =
        new ChunkedBlobDownloader(
            grpcCacheClient,
            combinedCache,
            CHUNKING_CONFIG,
            DIGEST_UTIL,
            "instance",
            ByteString.copyFromUtf8("fast-cdc-parameters"),
            ImmutableMap.of(
                CHUNKING_CONFIG.chunkingFunction(),
                CHUNKING_CONFIG,
                repMaxCdcConfig.chunkingFunction(),
                repMaxCdcConfig));

    downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream());

    verify(combinedCache, never()).uploadSplitBlobManifest(any(), any(), any());
  }

  @Test
  public void downloadChunked_cachedManifestUsesDifferentFunction_refetches() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    SplitBlobResponse cachedResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(chunkDigest)
            .setChunkingFunction(ChunkingFunction.Value.REP_MAX_CDC)
            .build();
    SplitBlobResponse remoteResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(chunkDigest)
            .setChunkingFunction(ChunkingFunction.Value.FAST_CDC_2020)
            .build();
    when(combinedCache.downloadSplitBlobManifest(eq(context), any())).thenReturn(cachedResponse);
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), eq(ChunkingFunction.Value.FAST_CDC_2020)))
        .thenReturn(Futures.immediateFuture(remoteResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));
    ChunkingConfig repMaxCdcConfig =
        new RepMaxCdcChunkingConfig(/* minChunkSize= */ 1024, /* horizonSize= */ 8192);
    ChunkedBlobDownloader downloader =
        new ChunkedBlobDownloader(
            grpcCacheClient,
            combinedCache,
            CHUNKING_CONFIG,
            DIGEST_UTIL,
            "instance",
            ByteString.copyFromUtf8("fast-cdc-parameters"),
            ImmutableMap.of(
                CHUNKING_CONFIG.chunkingFunction(),
                CHUNKING_CONFIG,
                repMaxCdcConfig.chunkingFunction(),
                repMaxCdcConfig));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEqualTo(chunkData);
    verify(combinedCache, never()).areBlobsPresentInDiskCache(any(), any());
    verify(grpcCacheClient).splitBlob(context, blobDigest, ChunkingFunction.Value.FAST_CDC_2020);
    verify(combinedCache).uploadSplitBlobManifest(eq(context), any(), eq(remoteResponse));
  }

  @Test
  public void downloadChunked_manifestCached_doesNotCallSplitBlob() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(combinedCache.downloadSplitBlobManifest(eq(context), any())).thenReturn(splitResponse);
    when(combinedCache.areBlobsPresentInDiskCache(eq(context), any())).thenReturn(true);
    when(combinedCache.downloadBlobFromDisk(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream cachedManifestOut = new ByteArrayOutputStream();
    boolean usedCachedManifest =
        downloader.downloadChunked(context, blobDigest, out, cachedManifestOut);

    assertThat(usedCachedManifest).isTrue();
    assertThat(out.toByteArray()).isEmpty();
    assertThat(cachedManifestOut.toByteArray()).isEqualTo(chunkData);
    verify(grpcCacheClient, never()).splitBlob(any(), any(), any());
    verify(combinedCache, never()).uploadSplitBlobManifest(any(), any(), any());
  }

  @Test
  public void downloadChunked_cachedManifestInvalid_refetchesAndOverwrites() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    SplitBlobResponse invalidCachedResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(
                DigestUtil.buildDigest(chunkDigest.getHash(), CHUNKING_CONFIG.maxChunkSize() + 1L))
            .build();
    SplitBlobResponse remoteResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(combinedCache.downloadSplitBlobManifest(eq(context), any()))
        .thenReturn(invalidCachedResponse);
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(remoteResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEqualTo(chunkData);
    verify(combinedCache).uploadSplitBlobManifest(eq(context), any(), eq(remoteResponse));
  }

  @Test
  public void downloadChunked_cachedManifestDigestInvalid_refetchesBeforeDiskLookup()
      throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    SplitBlobResponse invalidCachedResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(Digest.newBuilder().setHash("x").setSizeBytes(chunkData.length))
            .build();
    SplitBlobResponse remoteResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(combinedCache.downloadSplitBlobManifest(eq(context), any()))
        .thenReturn(invalidCachedResponse);
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(remoteResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream());

    verify(combinedCache, never()).areBlobsPresentInDiskCache(any(), any());
    verify(combinedCache).uploadSplitBlobManifest(eq(context), any(), eq(remoteResponse));
  }

  @Test
  public void downloadChunked_cachedManifestChunkEvicted_propagatesCacheNotFound()
      throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(combinedCache.downloadSplitBlobManifest(eq(context), any())).thenReturn(splitResponse);
    when(combinedCache.areBlobsPresentInDiskCache(eq(context), any())).thenReturn(true);
    when(combinedCache.downloadBlobFromDisk(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFailedFuture(new CacheNotFoundException(chunkDigest)));

    assertThrows(
        CacheNotFoundException.class,
        () -> downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream()));
  }

  @Test
  public void downloadChunked_chunkingParametersOrInstanceChanged_usesDifferentManifestKey()
      throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));
    ChunkedBlobDownloader firstDownloader =
        new ChunkedBlobDownloader(
            grpcCacheClient,
            combinedCache,
            CHUNKING_CONFIG,
            DIGEST_UTIL,
            "instance",
            ByteString.copyFromUtf8("parameters-v1"),
            ImmutableMap.of(CHUNKING_CONFIG.chunkingFunction(), CHUNKING_CONFIG));
    ChunkedBlobDownloader secondDownloader =
        new ChunkedBlobDownloader(
            grpcCacheClient,
            combinedCache,
            CHUNKING_CONFIG,
            DIGEST_UTIL,
            "instance",
            ByteString.copyFromUtf8("parameters-v2"),
            ImmutableMap.of(CHUNKING_CONFIG.chunkingFunction(), CHUNKING_CONFIG));
    ChunkedBlobDownloader downloaderWithChangedInstance =
        new ChunkedBlobDownloader(
            grpcCacheClient,
            combinedCache,
            CHUNKING_CONFIG,
            DIGEST_UTIL,
            "other-instance",
            ByteString.copyFromUtf8("parameters-v1"),
            ImmutableMap.of(CHUNKING_CONFIG.chunkingFunction(), CHUNKING_CONFIG));

    firstDownloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream());
    secondDownloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream());
    downloaderWithChangedInstance.downloadChunked(context, blobDigest, new ByteArrayOutputStream());

    ArgumentCaptor<Digest> manifestKeys = ArgumentCaptor.forClass(Digest.class);
    verify(combinedCache, times(3)).downloadSplitBlobManifest(eq(context), manifestKeys.capture());
    assertThat(manifestKeys.getAllValues().get(0)).isNotEqualTo(manifestKeys.getAllValues().get(1));
    assertThat(manifestKeys.getAllValues().get(0)).isNotEqualTo(manifestKeys.getAllValues().get(2));
  }

  @Test
  public void downloadChunked_chunkingFunctionChanged_doesNotHitManifestCache() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3, 4, 5};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    Map<Digest, SplitBlobResponse> manifestCache = new HashMap<>();
    when(combinedCache.downloadSplitBlobManifest(eq(context), any()))
        .thenAnswer(invocation -> manifestCache.get(invocation.getArgument(1)));
    when(combinedCache.areBlobsPresentInDiskCache(eq(context), any())).thenReturn(true);
    doAnswer(
            invocation -> {
              manifestCache.put(invocation.getArgument(1), invocation.getArgument(2));
              return null;
            })
        .when(combinedCache)
        .uploadSplitBlobManifest(eq(context), any(), any());

    ChunkingConfig fastCdcConfig = CHUNKING_CONFIG;
    ChunkingConfig repMaxCdcConfig =
        new RepMaxCdcChunkingConfig(/* minChunkSize= */ 1024, /* horizonSize= */ 8192);
    ImmutableMap<ChunkingFunction.Value, ChunkingConfig> chunkingConfigs =
        ImmutableMap.of(
            fastCdcConfig.chunkingFunction(),
            fastCdcConfig,
            repMaxCdcConfig.chunkingFunction(),
            repMaxCdcConfig);
    ChunkedBlobDownloader fastCdcDownloader =
        new ChunkedBlobDownloader(
            grpcCacheClient,
            combinedCache,
            fastCdcConfig,
            DIGEST_UTIL,
            "instance",
            ByteString.copyFromUtf8("same-parameters"),
            chunkingConfigs);
    ChunkedBlobDownloader repMaxCdcDownloader =
        new ChunkedBlobDownloader(
            grpcCacheClient,
            combinedCache,
            repMaxCdcConfig,
            DIGEST_UTIL,
            "instance",
            ByteString.copyFromUtf8("same-parameters"),
            chunkingConfigs);

    fastCdcDownloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream());
    repMaxCdcDownloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream());

    verify(grpcCacheClient)
        .splitBlob(eq(context), eq(blobDigest), eq(ChunkingFunction.Value.FAST_CDC_2020));
    verify(grpcCacheClient)
        .splitBlob(eq(context), eq(blobDigest), eq(ChunkingFunction.Value.REP_MAX_CDC));
    ArgumentCaptor<Digest> manifestKeys = ArgumentCaptor.forClass(Digest.class);
    verify(combinedCache, times(2)).downloadSplitBlobManifest(eq(context), manifestKeys.capture());
    assertThat(manifestKeys.getAllValues().get(0)).isNotEqualTo(manifestKeys.getAllValues().get(1));
  }

  @Test
  public void downloadChunked_multipleChunks_downloadsAndReassemblesInOrder() throws Exception {
    byte[] chunk1Data = new byte[] {1, 2, 3};
    byte[] chunk2Data = new byte[] {4, 5, 6};
    byte[] chunk3Data = new byte[] {7, 8, 9};
    Digest chunk1Digest = DIGEST_UTIL.compute(chunk1Data);
    Digest chunk2Digest = DIGEST_UTIL.compute(chunk2Data);
    Digest chunk3Digest = DIGEST_UTIL.compute(chunk3Data);
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {1, 2, 3, 4, 5, 6, 7, 8, 9});

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(chunk1Digest)
            .addChunkDigests(chunk2Digest)
            .addChunkDigests(chunk3Digest)
            .build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunk1Digest)))
        .thenReturn(Futures.immediateFuture(chunk1Data));
    when(combinedCache.downloadBlob(any(), eq(chunk2Digest)))
        .thenReturn(Futures.immediateFuture(chunk2Data));
    when(combinedCache.downloadBlob(any(), eq(chunk3Digest)))
        .thenReturn(Futures.immediateFuture(chunk3Data));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEqualTo(new byte[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
    verify(combinedCache).downloadBlob(any(), eq(chunk1Digest));
    verify(combinedCache).downloadBlob(any(), eq(chunk2Digest));
    verify(combinedCache).downloadBlob(any(), eq(chunk3Digest));
  }

  @Test
  public void downloadChunked_windowRefillsAfterOneChunkCompletes() throws Exception {
    List<Digest> chunkDigests = new ArrayList<>(MAX_IN_FLIGHT_CHUNK_DOWNLOADS + 1);
    List<SettableFuture<byte[]>> chunkFutures = new ArrayList<>(MAX_IN_FLIGHT_CHUNK_DOWNLOADS + 1);
    byte[] expectedData = new byte[MAX_IN_FLIGHT_CHUNK_DOWNLOADS + 1];
    SplitBlobResponse.Builder splitResponse = SplitBlobResponse.newBuilder();
    for (int i = 0; i < MAX_IN_FLIGHT_CHUNK_DOWNLOADS + 1; i++) {
      byte[] chunkData = new byte[] {(byte) (i + 1)};
      expectedData[i] = chunkData[0];
      chunkDigests.add(DIGEST_UTIL.compute(chunkData));
      chunkFutures.add(SettableFuture.create());
      splitResponse.addChunkDigests(chunkDigests.get(i));
    }
    Digest blobDigest = DIGEST_UTIL.compute(expectedData);

    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse.build()));

    CountDownLatch firstWindowRequested = new CountDownLatch(MAX_IN_FLIGHT_CHUNK_DOWNLOADS);
    CountDownLatch overflowChunkRequested = new CountDownLatch(1);

    when(combinedCache.downloadBlob(any(), any(Digest.class)))
        .thenAnswer(
            invocation -> {
              Digest digest = invocation.getArgument(1);
              int chunkIndex = chunkDigests.indexOf(digest);
              if (chunkIndex < MAX_IN_FLIGHT_CHUNK_DOWNLOADS) {
                firstWindowRequested.countDown();
              } else if (chunkIndex == MAX_IN_FLIGHT_CHUNK_DOWNLOADS) {
                overflowChunkRequested.countDown();
              }
              return chunkFutures.get(chunkIndex);
            });

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Thread downloadThread =
        Thread.ofVirtual()
            .unstarted(
                () -> {
                  try {
                    downloader.downloadChunked(context, blobDigest, out);
                  } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                  }
                });
    downloadThread.start();

    assertThat(firstWindowRequested.await(1, TimeUnit.SECONDS)).isTrue();
    assertThat(overflowChunkRequested.await(100, TimeUnit.MILLISECONDS)).isFalse();

    chunkFutures.get(0).set(new byte[] {expectedData[0]});
    assertThat(overflowChunkRequested.await(1, TimeUnit.SECONDS)).isTrue();

    for (int i = 0; i < chunkFutures.size(); i++) {
      SettableFuture<byte[]> future = chunkFutures.get(i);
      if (!future.isDone()) {
        future.set(new byte[] {expectedData[i]});
      }
    }
    downloadThread.join(TimeUnit.SECONDS.toMillis(1));

    assertThat(downloadThread.isAlive()).isFalse();
    assertThat(out.toByteArray()).isEqualTo(expectedData);
  }

  @Test
  public void downloadChunked_duplicateInFlightChunks_reusesDownload() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {1, 2, 3, 1, 2, 3});

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(chunkDigest)
            .addChunkDigests(chunkDigest)
            .build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    SettableFuture<byte[]> chunkFuture = SettableFuture.create();
    when(combinedCache.downloadBlob(any(), eq(chunkDigest))).thenReturn(chunkFuture);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Thread downloadThread =
        Thread.ofVirtual()
            .unstarted(
                () -> {
                  try {
                    downloader.downloadChunked(context, blobDigest, out);
                  } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                  }
                });
    downloadThread.start();

    chunkFuture.set(chunkData);
    downloadThread.join(TimeUnit.SECONDS.toMillis(1));

    assertThat(downloadThread.isAlive()).isFalse();
    assertThat(out.toByteArray()).isEqualTo(new byte[] {1, 2, 3, 1, 2, 3});
    verify(combinedCache, times(1)).downloadBlob(any(), eq(chunkDigest));
  }

  @Test
  public void downloadChunked_longDuplicateRun_resumesAfterDrain() throws Exception {
    byte[] firstChunkData = new byte[] {1};
    byte[] duplicateChunkData = new byte[] {2};
    byte[] finalChunkData = new byte[] {3};
    Digest firstChunkDigest = DIGEST_UTIL.compute(firstChunkData);
    Digest duplicateChunkDigest = DIGEST_UTIL.compute(duplicateChunkData);
    Digest finalChunkDigest = DIGEST_UTIL.compute(finalChunkData);

    byte[] blobData = new byte[MAX_IN_FLIGHT_CHUNK_DOWNLOADS + 1];
    blobData[0] = firstChunkData[0];
    for (int i = 1; i < MAX_IN_FLIGHT_CHUNK_DOWNLOADS; i++) {
      blobData[i] = duplicateChunkData[0];
    }
    blobData[MAX_IN_FLIGHT_CHUNK_DOWNLOADS] = finalChunkData[0];
    Digest blobDigest = DIGEST_UTIL.compute(blobData);

    SplitBlobResponse.Builder splitResponse = SplitBlobResponse.newBuilder();
    splitResponse.addChunkDigests(firstChunkDigest);
    for (int i = 1; i < MAX_IN_FLIGHT_CHUNK_DOWNLOADS; i++) {
      splitResponse.addChunkDigests(duplicateChunkDigest);
    }
    splitResponse.addChunkDigests(finalChunkDigest);
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse.build()));

    SettableFuture<byte[]> firstChunkFuture = SettableFuture.create();
    SettableFuture<byte[]> duplicateChunkFuture = SettableFuture.create();
    SettableFuture<byte[]> finalChunkFuture = SettableFuture.create();
    CountDownLatch initialDownloadsRequested = new CountDownLatch(2);
    CountDownLatch finalChunkRequested = new CountDownLatch(1);

    when(combinedCache.downloadBlob(any(), eq(firstChunkDigest)))
        .thenAnswer(
            invocation -> {
              initialDownloadsRequested.countDown();
              return firstChunkFuture;
            });
    when(combinedCache.downloadBlob(any(), eq(duplicateChunkDigest)))
        .thenAnswer(
            invocation -> {
              initialDownloadsRequested.countDown();
              return duplicateChunkFuture;
            });
    when(combinedCache.downloadBlob(any(), eq(finalChunkDigest)))
        .thenAnswer(
            invocation -> {
              finalChunkRequested.countDown();
              return finalChunkFuture;
            });

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Thread downloadThread =
        Thread.ofVirtual()
            .unstarted(
                () -> {
                  try {
                    downloader.downloadChunked(context, blobDigest, out);
                  } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                  }
                });
    downloadThread.start();

    assertThat(initialDownloadsRequested.await(1, TimeUnit.SECONDS)).isTrue();
    assertThat(finalChunkRequested.await(100, TimeUnit.MILLISECONDS)).isFalse();

    duplicateChunkFuture.set(duplicateChunkData);
    assertThat(finalChunkRequested.await(100, TimeUnit.MILLISECONDS)).isFalse();

    firstChunkFuture.set(firstChunkData);
    assertThat(finalChunkRequested.await(1, TimeUnit.SECONDS)).isTrue();

    finalChunkFuture.set(finalChunkData);
    downloadThread.join(TimeUnit.SECONDS.toMillis(1));

    assertThat(downloadThread.isAlive()).isFalse();
    assertThat(out.toByteArray()).isEqualTo(blobData);
  }

  @Test
  public void downloadChunked_emptyChunkList_producesEmptyOutput() throws Exception {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[0]);

    SplitBlobResponse splitResponse = SplitBlobResponse.getDefaultInstance();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEmpty();
  }

  @Test
  public void downloadChunked_oversizedChunk_throwsIOExceptionBeforeDownload() {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[1024]);
    Digest chunkDigest =
        DigestUtil.buildDigest(blobDigest.getHash(), CHUNKING_CONFIG.maxChunkSize() + 1L);
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    IOException e =
        assertThrows(
            IOException.class,
            () -> downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream()));

    assertThat(e).hasMessageThat().contains("exceeds max chunk size");
    verify(combinedCache, never()).downloadBlob(any(), any(Digest.class));
  }

  @Test
  public void downloadChunked_negativeChunkSize_throwsIOExceptionBeforeDownload() {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[1024]);
    Digest chunkDigest = DigestUtil.buildDigest(blobDigest.getHash(), -1);
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    IOException e =
        assertThrows(
            IOException.class,
            () -> downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream()));

    assertThat(e).hasMessageThat().contains("non-positive size");
    verify(combinedCache, never()).downloadBlob(any(), any(Digest.class));
  }

  @Test
  public void downloadChunked_chunkSizesExceedBlobSize_throwsIOExceptionBeforeDownload() {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[1024]);
    Digest chunkDigest = DigestUtil.buildDigest(blobDigest.getHash(), 1025);
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    IOException e =
        assertThrows(
            IOException.class,
            () -> downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream()));

    assertThat(e).hasMessageThat().contains("chunk sizes exceed blob size");
    verify(combinedCache, never()).downloadBlob(any(), any(Digest.class));
  }

  @Test
  public void downloadChunked_chunkSizesLessThanBlobSize_throwsIOExceptionBeforeDownload() {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[1024]);
    Digest chunkDigest = DigestUtil.buildDigest(blobDigest.getHash(), 1023);
    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    IOException e =
        assertThrows(
            IOException.class,
            () -> downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream()));

    assertThat(e).hasMessageThat().contains("chunk sizes do not match blob size");
    verify(combinedCache, never()).downloadBlob(any(), any(Digest.class));
  }

  @Test
  public void downloadChunked_chunkFails_throwsIOException() throws Exception {
    byte[] chunk1Data = new byte[] {1, 2, 3};
    byte[] chunk2Data = new byte[] {4, 5, 6};
    Digest chunk1Digest = DIGEST_UTIL.compute(chunk1Data);
    Digest chunk2Digest = DIGEST_UTIL.compute(chunk2Data);
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {1, 2, 3, 4, 5, 6});

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(chunk1Digest)
            .addChunkDigests(chunk2Digest)
            .build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunk1Digest)))
        .thenReturn(Futures.immediateFuture(chunk1Data));
    when(combinedCache.downloadBlob(any(), eq(chunk2Digest)))
        .thenReturn(Futures.immediateFailedFuture(new IOException("connection reset")));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    assertThrows(IOException.class, () -> downloader.downloadChunked(context, blobDigest, out));
  }

  @Test
  public void downloadChunked_blobDigestMismatch_throwsOutputDigestMismatch() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {4, 5, 6});

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    OutputDigestMismatchException e =
        assertThrows(
            OutputDigestMismatchException.class,
            () -> downloader.downloadChunked(context, blobDigest, new ByteArrayOutputStream()));

    assertThat(e).hasMessageThat().contains(blobDigest.getHash());
    assertThat(e).hasMessageThat().contains(chunkDigest.getHash());
  }

  @Test
  public void downloadChunked_blobDigestMismatchVerificationDisabled_succeeds() throws Exception {
    when(grpcCacheClient.shouldVerifyDownloads()).thenReturn(false);
    byte[] chunkData = new byte[] {1, 2, 3};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {4, 5, 6});

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEqualTo(chunkData);
  }

  @Test
  public void downloadChunked_cancelledChunk_throwsInterruptedException() throws Exception {
    byte[] chunkData = new byte[] {1, 2, 3};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = chunkDigest;

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    SettableFuture<byte[]> cancelledDownload = SettableFuture.create();
    cancelledDownload.cancel(/* mayInterruptIfRunning= */ true);
    when(combinedCache.downloadBlob(any(), eq(chunkDigest))).thenReturn(cancelledDownload);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    assertThrows(
        InterruptedException.class, () -> downloader.downloadChunked(context, blobDigest, out));
  }

  @Test
  public void downloadChunked_chunkFails_cancelsOtherInFlightDownloads() throws Exception {
    byte[] chunk1Data = new byte[] {1, 2, 3};
    byte[] chunk2Data = new byte[] {4, 5, 6};
    Digest chunk1Digest = DIGEST_UTIL.compute(chunk1Data);
    Digest chunk2Digest = DIGEST_UTIL.compute(chunk2Data);
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {1, 2, 3, 4, 5, 6});

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder()
            .addChunkDigests(chunk1Digest)
            .addChunkDigests(chunk2Digest)
            .build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest), any()))
        .thenReturn(Futures.immediateFuture(splitResponse));

    SettableFuture<byte[]> failedDownload = SettableFuture.create();
    SettableFuture<byte[]> cancelledDownload = SettableFuture.create();
    CountDownLatch downloadsStarted = new CountDownLatch(2);
    when(combinedCache.downloadBlob(any(), eq(chunk1Digest)))
        .thenAnswer(
            invocation -> {
              downloadsStarted.countDown();
              return failedDownload;
            });
    when(combinedCache.downloadBlob(any(), eq(chunk2Digest)))
        .thenAnswer(
            invocation -> {
              downloadsStarted.countDown();
              return cancelledDownload;
            });

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    Thread downloadThread =
        Thread.ofVirtual()
            .unstarted(
                () -> {
                  try {
                    downloader.downloadChunked(context, blobDigest, out);
                  } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                  }
                });
    downloadThread.start();

    assertThat(downloadsStarted.await(1, TimeUnit.SECONDS)).isTrue();
    failedDownload.setException(new IOException("connection reset"));

    downloadThread.join(TimeUnit.SECONDS.toMillis(1));

    assertThat(downloadThread.isAlive()).isFalse();
    assertThat(cancelledDownload.isCancelled()).isTrue();
  }
}
