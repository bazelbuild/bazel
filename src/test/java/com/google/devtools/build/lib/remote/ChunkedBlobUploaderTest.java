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
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.chunking.FastCdcChunker;
import com.google.devtools.build.lib.remote.chunking.FastCdcChunkingConfig;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.protobuf.ByteString;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
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

/** Tests for {@link ChunkedBlobUploader}. */
@RunWith(JUnit4.class)
public class ChunkedBlobUploaderTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final int MAX_IN_FLIGHT_CHUNK_UPLOADS = 16;

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private GrpcCacheClient grpcCacheClient;
  @Mock private CombinedCache combinedCache;
  @Mock private RemoteActionExecutionContext context;

  private FileSystem fs;
  private Path execRoot;
  private ChunkedBlobUploader uploader;

  @Before
  public void setUp() throws Exception {
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot");
    execRoot.createDirectoryAndParents();

    FastCdcChunkingConfig config = new FastCdcChunkingConfig(1024, 2, 0);
    uploader = new ChunkedBlobUploader(grpcCacheClient, combinedCache, config, DIGEST_UTIL);
  }

  @Test
  public void getChunkingThreshold_returnsConfiguredValue() {
    FastCdcChunkingConfig config = new FastCdcChunkingConfig(512, 2, 0);
    ChunkedBlobUploader uploader =
        new ChunkedBlobUploader(grpcCacheClient, combinedCache, config, DIGEST_UTIL);

    assertThat(uploader.getChunkingThreshold()).isEqualTo(512 * 4);
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_allChunksMissing_uploadsAllChunks() throws Exception {
    Path file = execRoot.getRelative("test.txt");
    byte[] data = new byte[8192];
    new Random(42).nextBytes(data);
    writeFile(file, data);
    Digest blobDigest = DIGEST_UTIL.compute(data);

    ArgumentCaptor<List<Digest>> digestsCaptor = ArgumentCaptor.forClass(List.class);
    when(grpcCacheClient.findMissingDigests(any(), digestsCaptor.capture()))
        .thenAnswer(
            invocation -> {
              List<Digest> digests = invocation.getArgument(1);
              return immediateFuture(ImmutableSet.copyOf(digests));
            });
    when(combinedCache.uploadBlob(any(), any(Digest.class), any(Blob.class)))
        .thenReturn(immediateVoidFuture());
    when(grpcCacheClient.spliceBlob(any(), any(), any())).thenReturn(immediateVoidFuture());

    uploader.uploadChunked(context, blobDigest, file);

    List<Digest> chunkDigests = digestsCaptor.getValue();
    assertThat(chunkDigests.size()).isGreaterThan(1);
    long totalSize = chunkDigests.stream().mapToLong(Digest::getSizeBytes).sum();
    assertThat(totalSize).isEqualTo(data.length);
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_noChunksMissing_skipsChunkUpload() throws Exception {
    Path file = execRoot.getRelative("test.txt");
    byte[] data = new byte[8192];
    new Random(42).nextBytes(data);
    writeFile(file, data);
    Digest blobDigest = DIGEST_UTIL.compute(data);

    when(grpcCacheClient.findMissingDigests(any(), any()))
        .thenReturn(immediateFuture(ImmutableSet.of()));
    when(grpcCacheClient.spliceBlob(any(), any(), any())).thenReturn(immediateVoidFuture());

    uploader.uploadChunked(context, blobDigest, file);

    verify(combinedCache, never()).uploadBlob(any(), any(Digest.class), any(Blob.class));
    verify(grpcCacheClient).spliceBlob(any(), eq(blobDigest), any());
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_someChunksMissing_uploadsOnlyMissingWithCorrectData() throws Exception {
    Path file = execRoot.getRelative("test_partial.txt");
    byte[] fileData = new byte[16384];
    new Random(42).nextBytes(fileData);
    writeFile(file, fileData);
    Digest blobDigest = DIGEST_UTIL.compute(fileData);

    FastCdcChunkingConfig config = new FastCdcChunkingConfig(1024, 2, 0);
    FastCdcChunker testChunker = new FastCdcChunker(config, DIGEST_UTIL);
    List<Digest> allChunkDigests;
    try (InputStream input = file.getInputStream()) {
      allChunkDigests = testChunker.chunkToDigests(input);
    }
    assertThat(allChunkDigests.size()).isAtLeast(5);

    Set<Digest> digestsToReportMissing = new LinkedHashSet<>();
    for (int i = 0; i < allChunkDigests.size(); i++) {
      boolean isFirst = i == 0;
      boolean isLast = i == allChunkDigests.size() - 1;
      boolean isOdd = i % 2 == 1;
      if (isFirst || isLast || isOdd) {
        digestsToReportMissing.add(allChunkDigests.get(i));
      }
    }

    Map<Digest, ByteString> expectedChunkData = new LinkedHashMap<>();
    try (InputStream input = file.getInputStream()) {
      for (Digest digest : allChunkDigests) {
        byte[] chunkBytes = input.readNBytes((int) digest.getSizeBytes());
        if (digestsToReportMissing.contains(digest)) {
          expectedChunkData.put(digest, ByteString.copyFrom(chunkBytes));
        }
      }
    }

    when(grpcCacheClient.findMissingDigests(any(), any()))
        .thenReturn(immediateFuture(ImmutableSet.copyOf(digestsToReportMissing)));
    Map<Digest, ByteString> actualUploads = new HashMap<>();
    when(combinedCache.uploadBlob(any(), any(Digest.class), any(Blob.class)))
        .thenAnswer(
            invocation -> {
              Digest d = invocation.getArgument(1);
              Blob blob = invocation.getArgument(2);
              try (InputStream in = blob.get()) {
                actualUploads.put(d, ByteString.readFrom(in));
              }
              return immediateVoidFuture();
            });
    when(grpcCacheClient.spliceBlob(any(), any(), any())).thenReturn(immediateVoidFuture());

    uploader.uploadChunked(context, blobDigest, file);

    assertThat(actualUploads.keySet()).isEqualTo(expectedChunkData.keySet());
    for (Map.Entry<Digest, ByteString> entry : expectedChunkData.entrySet()) {
      assertThat(actualUploads.get(entry.getKey())).isEqualTo(entry.getValue());
    }
    verify(grpcCacheClient).spliceBlob(any(), eq(blobDigest), eq(allChunkDigests));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_windowRefillsAfterOneChunkCompletes() throws Exception {
    Path file = execRoot.getRelative("test_window.txt");
    byte[] data = new byte[262144];
    new Random(42).nextBytes(data);
    writeFile(file, data);
    Digest blobDigest = DIGEST_UTIL.compute(data);

    FastCdcChunker testChunker =
        new FastCdcChunker(new FastCdcChunkingConfig(1024, 2, 0), DIGEST_UTIL);
    List<Digest> chunkDigests;
    try (InputStream input = file.getInputStream()) {
      chunkDigests = testChunker.chunkToDigests(input);
    }

    List<Digest> uniqueChunkDigests = new ArrayList<>();
    Set<Digest> seen = new HashSet<>();
    for (Digest chunkDigest : chunkDigests) {
      if (seen.add(chunkDigest)) {
        uniqueChunkDigests.add(chunkDigest);
      }
      if (uniqueChunkDigests.size() == MAX_IN_FLIGHT_CHUNK_UPLOADS + 1) {
        break;
      }
    }
    assertThat(uniqueChunkDigests).hasSize(MAX_IN_FLIGHT_CHUNK_UPLOADS + 1);

    when(grpcCacheClient.findMissingDigests(any(), any()))
        .thenReturn(immediateFuture(ImmutableSet.copyOf(uniqueChunkDigests)));
    when(grpcCacheClient.spliceBlob(any(), any(), any())).thenReturn(immediateVoidFuture());

    List<SettableFuture<Void>> uploads = new ArrayList<>(uniqueChunkDigests.size());
    for (int i = 0; i < uniqueChunkDigests.size(); i++) {
      uploads.add(SettableFuture.create());
    }
    CountDownLatch firstWindowRequested = new CountDownLatch(MAX_IN_FLIGHT_CHUNK_UPLOADS);
    CountDownLatch overflowUploadRequested = new CountDownLatch(1);

    when(combinedCache.uploadBlob(any(), any(Digest.class), any(Blob.class)))
        .thenAnswer(
            invocation -> {
              Digest digest = invocation.getArgument(1);
              int chunkIndex = uniqueChunkDigests.indexOf(digest);
              if (chunkIndex < MAX_IN_FLIGHT_CHUNK_UPLOADS) {
                firstWindowRequested.countDown();
              } else if (chunkIndex == MAX_IN_FLIGHT_CHUNK_UPLOADS) {
                overflowUploadRequested.countDown();
              }
              return uploads.get(chunkIndex);
            });

    Thread uploadThread =
        Thread.ofVirtual()
            .unstarted(
                () -> {
                  try {
                    uploader.uploadChunked(context, blobDigest, file);
                  } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                  }
                });
    uploadThread.start();

    assertThat(firstWindowRequested.await(1, TimeUnit.SECONDS)).isTrue();
    assertThat(overflowUploadRequested.await(100, TimeUnit.MILLISECONDS)).isFalse();

    uploads.get(1).set(null);
    assertThat(overflowUploadRequested.await(1, TimeUnit.SECONDS)).isTrue();

    for (SettableFuture<Void> upload : uploads) {
      if (!upload.isDone()) {
        upload.set(null);
      }
    }
    uploadThread.join(TimeUnit.SECONDS.toMillis(1));

    assertThat(uploadThread.isAlive()).isFalse();
    verify(grpcCacheClient).spliceBlob(any(), eq(blobDigest), eq(chunkDigests));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_chunkFails_cancelsOtherInFlightUploads() throws Exception {
    Path file = execRoot.getRelative("test_failure.txt");
    byte[] data = new byte[16384];
    new Random(42).nextBytes(data);
    writeFile(file, data);
    Digest blobDigest = DIGEST_UTIL.compute(data);

    FastCdcChunker testChunker =
        new FastCdcChunker(new FastCdcChunkingConfig(1024, 2, 0), DIGEST_UTIL);
    List<Digest> chunkDigests;
    try (InputStream input = file.getInputStream()) {
      chunkDigests = testChunker.chunkToDigests(input);
    }

    List<Digest> uniqueChunkDigests = new ArrayList<>();
    Set<Digest> seen = new HashSet<>();
    for (Digest chunkDigest : chunkDigests) {
      if (seen.add(chunkDigest)) {
        uniqueChunkDigests.add(chunkDigest);
      }
      if (uniqueChunkDigests.size() == 2) {
        break;
      }
    }
    assertThat(uniqueChunkDigests).hasSize(2);

    when(grpcCacheClient.findMissingDigests(any(), any()))
        .thenReturn(immediateFuture(ImmutableSet.copyOf(uniqueChunkDigests)));

    SettableFuture<Void> failedUpload = SettableFuture.create();
    SettableFuture<Void> cancelledUpload = SettableFuture.create();
    CountDownLatch uploadsStarted = new CountDownLatch(2);
    when(combinedCache.uploadBlob(any(), any(Digest.class), any(Blob.class)))
        .thenAnswer(
            invocation -> {
              Digest digest = invocation.getArgument(1);
              uploadsStarted.countDown();
              if (digest.equals(uniqueChunkDigests.get(0))) {
                return failedUpload;
              }
              if (digest.equals(uniqueChunkDigests.get(1))) {
                return cancelledUpload;
              }
              return immediateVoidFuture();
            });

    Thread uploadThread =
        Thread.ofVirtual()
            .unstarted(
                () -> {
                  try {
                    uploader.uploadChunked(context, blobDigest, file);
                  } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                  }
                });
    uploadThread.start();

    assertThat(uploadsStarted.await(1, TimeUnit.SECONDS)).isTrue();
    failedUpload.setException(new IOException("upload failed"));

    uploadThread.join(TimeUnit.SECONDS.toMillis(1));

    assertThat(uploadThread.isAlive()).isFalse();
    assertThat(cancelledUpload.isCancelled()).isTrue();
    verify(grpcCacheClient, never()).spliceBlob(any(), any(), any());
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_cancelledUpload_throwsInterruptedException() throws Exception {
    Path file = execRoot.getRelative("test_cancelled.txt");
    byte[] data = new byte[8192];
    new Random(42).nextBytes(data);
    writeFile(file, data);
    Digest blobDigest = DIGEST_UTIL.compute(data);

    FastCdcChunker testChunker =
        new FastCdcChunker(new FastCdcChunkingConfig(1024, 2, 0), DIGEST_UTIL);
    List<Digest> chunkDigests;
    try (InputStream input = file.getInputStream()) {
      chunkDigests = testChunker.chunkToDigests(input);
    }
    Digest firstChunkDigest = chunkDigests.get(0);

    when(grpcCacheClient.findMissingDigests(any(), any()))
        .thenReturn(immediateFuture(ImmutableSet.of(firstChunkDigest)));

    SettableFuture<Void> cancelledUpload = SettableFuture.create();
    cancelledUpload.cancel(/* mayInterruptIfRunning= */ true);
    when(combinedCache.uploadBlob(any(), eq(firstChunkDigest), any(Blob.class)))
        .thenReturn(cancelledUpload);

    assertThrows(
        InterruptedException.class, () -> uploader.uploadChunked(context, blobDigest, file));
    verify(grpcCacheClient, never()).spliceBlob(any(), any(), any());
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_failedUploadDuringPendingChunks_surfacesBeforeOpeningChunkStream()
      throws Exception {
    byte[] data = new byte[16384];
    new Random(42).nextBytes(data);
    Digest blobDigest = DIGEST_UTIL.compute(data);

    FastCdcChunker testChunker =
        new FastCdcChunker(new FastCdcChunkingConfig(1024, 2, 0), DIGEST_UTIL);
    List<Digest> chunkDigests;
    try (InputStream input = new ByteArrayInputStream(data)) {
      chunkDigests = testChunker.chunkToDigests(input);
    }
    assertThat(chunkDigests.size()).isAtLeast(2);

    Path file = mock(Path.class);
    when(file.getInputStream()).thenReturn(new ByteArrayInputStream(data));

    when(grpcCacheClient.findMissingDigests(any(), any()))
        .thenReturn(immediateFuture(ImmutableSet.of(chunkDigests.get(0))));

    SettableFuture<Void> failedUpload = SettableFuture.create();
    failedUpload.setException(new IOException("upload failed"));
    when(combinedCache.uploadBlob(any(), eq(chunkDigests.get(0)), any(Blob.class)))
        .thenReturn(failedUpload);

    Thread uploadThread =
        Thread.ofVirtual()
            .unstarted(
                () -> {
                  try {
                    uploader.uploadChunked(context, blobDigest, file);
                  } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                  }
                });
    uploadThread.start();

    uploadThread.join(TimeUnit.SECONDS.toMillis(1));
    assertThat(uploadThread.isAlive()).isFalse();
    verify(file, times(1)).getInputStream();
    verify(grpcCacheClient, never()).spliceBlob(any(), any(), any());
  }

  @Test
  @SuppressWarnings("unchecked")
  public void uploadChunked_fileTruncatedBeforeChunkUpload_reportsConcurrentModification()
      throws Exception {
    byte[] data = new byte[8192];
    new Random(42).nextBytes(data);
    Digest blobDigest = DIGEST_UTIL.compute(data);

    FastCdcChunker testChunker =
        new FastCdcChunker(new FastCdcChunkingConfig(1024, 2, 0), DIGEST_UTIL);
    List<Digest> chunkDigests;
    try (InputStream input = new ByteArrayInputStream(data)) {
      chunkDigests = testChunker.chunkToDigests(input);
    }
    assertThat(chunkDigests.size()).isAtLeast(2);

    Digest secondChunkDigest = chunkDigests.get(1);
    Path file = mock(Path.class);
    when(file.getInputStream())
        .thenReturn(new ByteArrayInputStream(data), new ByteArrayInputStream(new byte[0]));
    when(grpcCacheClient.findMissingDigests(any(), any()))
        .thenReturn(immediateFuture(ImmutableSet.of(secondChunkDigest)));
    when(combinedCache.uploadBlob(any(), eq(secondChunkDigest), any(Blob.class)))
        .thenAnswer(
            invocation -> {
              Blob blob = invocation.getArgument(2);
              try (InputStream in = blob.get()) {
                ByteString unused = ByteString.readFrom(in);
              }
              return immediateVoidFuture();
            });

    IOException e =
        assertThrows(IOException.class, () -> uploader.uploadChunked(context, blobDigest, file));

    assertThat(e).hasMessageThat().contains("file was concurrently modified during upload");
    assertThat(e).hasCauseThat().isInstanceOf(EOFException.class);
    verify(grpcCacheClient, never()).spliceBlob(any(), any(), any());
  }

  private void writeFile(Path path, byte[] data) throws IOException {
    try (var out = path.getOutputStream()) {
      out.write(data);
    }
  }
}
