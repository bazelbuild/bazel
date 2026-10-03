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

import com.github.luben.zstd.ZstdOutputStreamNoFinalizer;
import com.google.common.base.Preconditions;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/** An {@link InputStream} that uses zstd to compress its input. */
public class ZstdCompressingInputStream extends InputStream {
  // Retained for callers of the previous pipe-based implementation.
  public static final int MIN_BUFFER_SIZE = 4 + 14 + 3 + 1;

  // Reuse a bounded input buffer. Short reads must not force short compressed blocks.
  private static final int BUFFER_SIZE = 16 * 1024;

  private final InputStream in;
  private final byte[] inputBuffer;
  private final CompressedBuffer compressed = new CompressedBuffer();
  private ByteArrayInputStream compressedInput = new ByteArrayInputStream(new byte[0]);
  private ZstdOutputStreamNoFinalizer zos;
  private boolean closed;

  public ZstdCompressingInputStream(InputStream in) throws IOException {
    this(in, BUFFER_SIZE);
  }

  ZstdCompressingInputStream(InputStream in, int size) throws IOException {
    this.in = Objects.requireNonNull(in);
    Preconditions.checkArgument(
        size >= MIN_BUFFER_SIZE, "The buffer size must be at least %s bytes", MIN_BUFFER_SIZE);
    inputBuffer = new byte[size];
    zos = new ZstdOutputStreamNoFinalizer(compressed);
  }

  private void reFill() throws IOException {
    if (closed) {
      throw new IOException("Stream closed");
    }
    if (compressedInput.available() > 0 || zos == null) {
      return;
    }
    compressed.reset();
    // A write may consume input without producing output. Keep feeding the compressor until a
    // natural block is emitted, or finalize the frame at EOF. Stop as soon as there is output:
    // only one input buffer plus zstd's pending block can accumulate, regardless of blob size.
    while (compressed.size() == 0 && zos != null) {
      int len = in.read(inputBuffer);
      if (len == -1) {
        zos.close();
        zos = null;
      } else {
        zos.write(inputBuffer, 0, len);
      }
    }
    compressedInput = compressed.inputStream();
  }

  @Override
  public int read() throws IOException {
    reFill();
    return compressedInput.read();
  }

  @Override
  public int read(byte[] b, int off, int len) throws IOException {
    Objects.checkFromIndexSize(off, len, b.length);
    if (len == 0) {
      return 0;
    }
    reFill();
    return compressedInput.read(b, off, len);
  }

  @Override
  public int available() throws IOException {
    if (closed) {
      throw new IOException("Stream closed");
    }
    return compressedInput.available();
  }

  @Override
  public void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    try {
      if (zos != null) {
        // Discard unread output before finalizing an abandoned frame.
        compressed.reset();
        zos.close();
        zos = null;
      }
    } finally {
      in.close();
    }
  }

  private static final class CompressedBuffer extends ByteArrayOutputStream {
    CompressedBuffer() {
      super(BUFFER_SIZE);
    }

    ByteArrayInputStream inputStream() {
      // The buffer is only reset/reused after this view has been fully consumed.
      return new ByteArrayInputStream(buf, 0, count);
    }
  }
}
