// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.zstd;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.github.luben.zstd.Zstd;
import com.google.common.io.ByteStreams;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ZstdCompressingInputStream}. */
@RunWith(JUnit4.class)
public class ZstdCompressingInputStreamTest {
  @Test
  public void compressionWorks() throws IOException {
    Random rand = new Random(1);
    byte[] data = new byte[50];
    rand.nextBytes(data);

    ByteArrayInputStream bais = new ByteArrayInputStream(data);
    try (ZstdCompressingInputStream zdis = new ZstdCompressingInputStream(bais)) {
      assertThat(Zstd.decompress(ByteStreams.toByteArray(zdis), data.length)).isEqualTo(data);
    }
  }

  @Test
  public void emptyInputProducesCompleteFrame() throws IOException {
    try (var stream = new ZstdCompressingInputStream(new ByteArrayInputStream(new byte[0]))) {
      byte[] compressed = ByteStreams.toByteArray(stream);
      assertThat(compressed).isNotEmpty();
      assertThat(Zstd.decompress(compressed, 0)).isEmpty();
      assertThat(stream.read()).isEqualTo(-1);
      assertThat(stream.read(new byte[1])).isEqualTo(-1);
      assertThat(stream.read(new byte[1], 0, 0)).isEqualTo(0);
    }
  }

  @Test
  public void incompressibleInputSupportsPartialAndSingleByteReads() throws IOException {
    byte[] data = new byte[1024 * 1024 + 17];
    new Random(1).nextBytes(data);
    try (var stream = new ZstdCompressingInputStream(new ByteArrayInputStream(data))) {
      ByteArrayOutputStream compressed = new ByteArrayOutputStream();
      byte[] buffer = new byte[31];
      int first = stream.read();
      assertThat(first).isAtLeast(0);
      assertThat(first).isAtMost(255);
      compressed.write(first);
      int n;
      while ((n = stream.read(buffer, 3, 23)) != -1) {
        assertThat(n).isGreaterThan(0);
        assertThat(n).isAtMost(23);
        assertThat(buffer[0]).isEqualTo((byte) 0);
        assertThat(buffer[30]).isEqualTo((byte) 0);
        compressed.write(buffer, 3, n);
        int single = stream.read();
        if (single != -1) {
          assertThat(single).isAtLeast(0);
          assertThat(single).isAtMost(255);
          compressed.write(single);
        }
      }
      assertThat(Zstd.decompress(compressed.toByteArray(), data.length)).isEqualTo(data);
    }
  }

  @Test
  public void shortSourceReadsDoNotForceShortCompressedBlocks() throws IOException {
    byte[] data = new byte[1024 * 1024];
    new Random(1).nextBytes(data);
    // Repeat a random block: unlike all-zero input, tiny flushed blocks have little redundancy.
    for (int offset = 4096; offset < data.length; offset += 4096) {
      System.arraycopy(data, 0, data, offset, 4096);
    }
    InputStream source =
        new ByteArrayInputStream(data) {
          @Override
          public synchronized int read(byte[] b, int off, int len) {
            return super.read(b, off, Math.min(len, 37));
          }
        };
    try (var stream = new ZstdCompressingInputStream(source)) {
      byte[] compressed = ByteStreams.toByteArray(stream);
      assertThat(Zstd.decompress(compressed, data.length)).isEqualTo(data);
      assertThat(compressed.length).isLessThan(6000);
    }
  }

  @Test
  public void closeDoesNotConsumeRemainingSource() throws IOException {
    for (boolean readFirst : new boolean[] {false, true}) {
      var source =
          new ByteArrayInputStream(new byte[4 * 1024 * 1024]) {
            boolean closed;

            @Override
            public void close() {
              closed = true;
            }
          };
      var stream = new ZstdCompressingInputStream(source);
      if (readFirst) {
        assertThat(stream.read()).isAtLeast(0);
      }
      int remaining = source.available();
      assertThat(remaining).isGreaterThan(0);
      stream.close();
      stream.close();
      assertThat(source.closed).isTrue();
      assertThat(source.available()).isEqualTo(remaining);
    }
  }

  @Test
  public void sourceIsClosedAfterReadFailure() throws IOException {
    var source =
        new InputStream() {
          boolean closed;

          @Override
          public int read() throws IOException {
            throw new IOException("source failure");
          }

          @Override
          public void close() {
            closed = true;
          }
        };
    try (var stream = new ZstdCompressingInputStream(source)) {
      assertThrows(IOException.class, stream::read);
    }
    assertThat(source.closed).isTrue();
  }

  @Test
  public void largeGeneratedInputIsConsumedIncrementally() throws IOException {
    var source =
        new InputStream() {
          long remaining = 64L * 1024 * 1024;

          @Override
          public int read() {
            if (remaining == 0) {
              return -1;
            }
            remaining--;
            return 0;
          }

          @Override
          public int read(byte[] b, int off, int len) {
            if (remaining == 0) {
              return -1;
            }
            int n = (int) Math.min(len, remaining);
            java.util.Arrays.fill(b, off, off + n, (byte) 0);
            remaining -= n;
            return n;
          }
        };
    try (var stream = new ZstdCompressingInputStream(source)) {
      assertThat(stream.read()).isAtLeast(0);
      assertThat(source.remaining).isGreaterThan(0);
      // Drain without retaining a blob-sized input or compressed output array.
      assertThat(stream.transferTo(java.io.OutputStream.nullOutputStream())).isGreaterThan(0L);
      assertThat(source.remaining).isEqualTo(0L);
      assertThat(stream.read()).isEqualTo(-1);
    }
  }

  @Test
  public void skipConsumesCompressedBytes() throws IOException {
    byte[] data = new byte[4096];
    new Random(1).nextBytes(data);
    try (var expected = new ZstdCompressingInputStream(new ByteArrayInputStream(data));
        var actual = new ZstdCompressingInputStream(new ByteArrayInputStream(data))) {
      assertThat(actual.markSupported()).isFalse();
      assertThat(expected.readNBytes(123)).hasLength(123);
      assertThat(actual.skip(123)).isEqualTo(123L);
      assertThat(ByteStreams.toByteArray(actual)).isEqualTo(ByteStreams.toByteArray(expected));
    }
  }
}
