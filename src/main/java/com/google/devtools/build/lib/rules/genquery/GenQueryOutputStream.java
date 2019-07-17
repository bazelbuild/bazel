// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genquery;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * {@link OutputStream} implementation optimized for {@link GenQuery} by (optionally) compressing
 * query results on the fly. Produces {@link GenQueryResult}s which are preferred for storing the
 * output of {@link GenQuery}'s underlying queries.
 */
class GenQueryOutputStream extends OutputStream {

  /**
   * When compression is enabled, the threshold at which the stream will switch to compressing
   * output. The value of this constant is arbitrary but effective.
   */
  private static final int COMPRESSION_THRESHOLD = 1 << 20;

  /**
   * Encapsulates the output of a {@link GenQuery}'s query. CPU and memory overhead of individual
   * methods depends on the underlying content and settings.
   */
  interface GenQueryResult {
    /** Returns the query output as a {@link ByteString}. */
    ByteString getBytes() throws IOException;

    /**
     * Adds the query output to the supplied {@link Fingerprint}. Equivalent to {@code
     * fingerprint.addBytes(genQueryResult.getBytes())}, but potentially more efficient.
     */
    void fingerprint(Fingerprint fingerprint);

    /**
     * Returns the size of the output. This must be a constant-time operation for all
     * implementations.
     */
    int size();

    /**
     * Writes the query output to the provided {@link OutputStream}. Equivalent to {@code
     * genQueryResult.getBytes().writeTo(out)}, but potentially more efficient.
     */
    void writeTo(OutputStream out) throws IOException;
  }

  private final boolean compressionEnabled;
  private int bytesWritten = 0;
  private boolean compressed = false;
  private boolean closed = false;
  private ByteString.Output bytesOut = ByteString.newOutput();
  private OutputStream out = bytesOut;

  GenQueryOutputStream(boolean compressionEnabled) {
    this.compressionEnabled = compressionEnabled;
  }

  @Override
  public void write(int b) throws IOException {
    maybeStartCompression(1);
    out.write(b);
    bytesWritten += 1;
  }

  @Override
  public void write(byte[] bytes) throws IOException {
    write(bytes, 0, bytes.length);
  }

  @Override
  public void write(byte[] bytes, int off, int len) throws IOException {
    maybeStartCompression(len);
    out.write(bytes, off, len);
    bytesWritten += len;
  }

  @Override
  public void flush() throws IOException {
    out.flush();
  }

  @Override
  public void close() throws IOException {
    out.close();
    closed = true;
  }

  GenQueryResult getResult() {
    Preconditions.checkState(closed, "Must be closed");
    return compressed
        ? new CompressedResult(bytesOut.toByteString(), bytesWritten)
        : new RegularResult(bytesOut.toByteString());
  }

  private void maybeStartCompression(int additionalBytes) throws IOException {
    if (!compressionEnabled) {
      return;
    }

    if (compressed) {
      return;
    }

    if (bytesWritten + additionalBytes < COMPRESSION_THRESHOLD) {
      return;
    }

    ByteString.Output compressedBytesOut = ByteString.newOutput();
    GZIPOutputStream gzipOut = new GZIPOutputStream(compressedBytesOut);
    bytesOut.writeTo(gzipOut);
    bytesOut = compressedBytesOut;
    out = gzipOut;
    compressed = true;
  }

  @VisibleForTesting
  static class RegularResult implements GenQueryResult {
    private final ByteString data;

    RegularResult(ByteString data) {
      this.data = data;
    }

    @Override
    public ByteString getBytes() {
      return data;
    }

    @Override
    public int size() {
      return data.size();
    }

    @Override
    public void fingerprint(Fingerprint fingerprint) {
      fingerprint.addBytes(data);
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      data.writeTo(out);
    }
  }

  @VisibleForTesting
  static class CompressedResult implements GenQueryResult {
    private final ByteString compressedData;
    private final int size;

    CompressedResult(ByteString compressedData, int size) {
      this.compressedData = compressedData;
      this.size = size;
    }

    @Override
    public ByteString getBytes() throws IOException {
      ByteString.Output out = ByteString.newOutput(size);
      try (GZIPInputStream gzipIn = new GZIPInputStream(compressedData.newInput())) {
        ByteStreams.copy(gzipIn, out);
      }
      return out.toByteString();
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      try (GZIPInputStream gzipIn = new GZIPInputStream(compressedData.newInput())) {
        ByteStreams.copy(gzipIn, out);
      }
    }

    @Override
    public void fingerprint(Fingerprint fingerprint) {
      try (GZIPInputStream gzipIn = new GZIPInputStream(compressedData.newInput())) {
        byte[] chunk = new byte[4092];
        int bytesRead;
        while ((bytesRead = gzipIn.read(chunk)) > 0) {
          fingerprint.addBytes(chunk, 0, bytesRead);
        }
      } catch (IOException e) {
        // Unexpected, everything should be in memory!
        throw new IllegalStateException("Unexpected IOException", e);
      }
    }
  }
}
