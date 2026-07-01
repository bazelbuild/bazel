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
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests for {@link ChunkedBlobDownloader}. */
@RunWith(JUnit4.class)
public class ChunkedBlobDownloaderTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final int MAX_IN_FLIGHT_CHUNK_DOWNLOADS = 16;

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private GrpcCacheClient grpcCacheClient;
  @Mock private CombinedCache combinedCache;
  @Mock private RemoteActionExecutionContext context;

  private ChunkedBlobDownloader downloader;

  @Before
  public void setUp() {
    downloader =
        new ChunkedBlobDownloader(
            grpcCacheClient, combinedCache, DIGEST_UTIL, /* verifyDownloads= */ true);
  }

  @Test
  public void downloadChunked_splitBlobReturnsNull_throwsCacheNotFound() {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {1, 2, 3});
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest))).thenReturn(null);

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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
        .thenReturn(Futures.immediateFuture(splitResponse));
    when(combinedCache.downloadBlob(any(), eq(chunkDigest)))
        .thenReturn(Futures.immediateFuture(chunkData));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEqualTo(chunkData);
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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

    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
        .thenReturn(Futures.immediateFuture(splitResponse));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEmpty();
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
    downloader =
        new ChunkedBlobDownloader(
            grpcCacheClient, combinedCache, DIGEST_UTIL, /* verifyDownloads= */ false);
    byte[] chunkData = new byte[] {1, 2, 3};
    Digest chunkDigest = DIGEST_UTIL.compute(chunkData);
    Digest blobDigest = DIGEST_UTIL.compute(new byte[] {4, 5, 6});

    SplitBlobResponse splitResponse =
        SplitBlobResponse.newBuilder().addChunkDigests(chunkDigest).build();
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
    when(grpcCacheClient.splitBlob(any(), eq(blobDigest)))
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
