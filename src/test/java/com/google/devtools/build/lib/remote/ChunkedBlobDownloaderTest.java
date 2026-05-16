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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.SplitBlobResponse;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.MaybePathBacked;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import javax.annotation.Nullable;
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

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private GrpcCacheClient grpcCacheClient;
  @Mock private CombinedCache combinedCache;
  @Mock private RemoteActionExecutionContext context;

  private ChunkedBlobDownloader downloader;

  @Before
  public void setUp() {
    downloader = new ChunkedBlobDownloader(grpcCacheClient, combinedCache);
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
    whenDownloadBlobWrites(chunkDigest, chunkData);

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
    Map<Digest, byte[]> chunks =
        Map.of(
            chunk1Digest, chunk1Data,
            chunk2Digest, chunk2Data,
            chunk3Digest, chunk3Data);

    // Per-chunk downloads must not see the final output's backing path.
    when(combinedCache.downloadBlob(any(), any(Digest.class), any(OutputStream.class)))
        .thenAnswer(
            invocation -> {
              byte[] data = chunks.get(invocation.getArgument(1));
              assertThat(data).isNotNull();
              OutputStream out = invocation.getArgument(2);
              assertThat(out).isNotInstanceOf(MaybePathBacked.class);
              out.write(data);
              return Futures.immediateFuture(null);
            });

    InMemoryFileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    PathBackedOutputStream out = new PathBackedOutputStream(fs.getPath("/output"));
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEqualTo(new byte[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
    verify(combinedCache).downloadBlob(any(), eq(chunk1Digest), any(OutputStream.class));
    verify(combinedCache).downloadBlob(any(), eq(chunk2Digest), any(OutputStream.class));
    verify(combinedCache).downloadBlob(any(), eq(chunk3Digest), any(OutputStream.class));
  }

  @Test
  public void downloadChunked_emptyChunkList_producesEmptyOutput() throws Exception {
    Digest blobDigest = DIGEST_UTIL.compute(new byte[0]);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    downloader.downloadChunked(context, blobDigest, out);

    assertThat(out.toByteArray()).isEmpty();
  }

  @Test
  public void downloadChunked_chunkFailsAfterPartialWrite_throwsIOException() throws Exception {
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
    whenDownloadBlobWrites(chunk1Digest, chunk1Data);
    when(combinedCache.downloadBlob(any(), eq(chunk2Digest), any(OutputStream.class)))
        .thenReturn(Futures.immediateFailedFuture(new IOException("connection reset")));

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    assertThrows(IOException.class, () -> downloader.downloadChunked(context, blobDigest, out));
  }

  private void whenDownloadBlobWrites(Digest digest, byte[] data) {
    when(combinedCache.downloadBlob(any(), eq(digest), any(OutputStream.class)))
        .thenAnswer(
            invocation -> {
              OutputStream out = invocation.getArgument(2);
              out.write(data);
              return Futures.immediateVoidFuture();
            });
  }

  private static class PathBackedOutputStream extends ByteArrayOutputStream
      implements MaybePathBacked {
    private final Path path;

    PathBackedOutputStream(Path path) {
      this.path = path;
    }

    @Nullable
    @Override
    public Path maybeGetPath() {
      return path;
    }
  }
}
