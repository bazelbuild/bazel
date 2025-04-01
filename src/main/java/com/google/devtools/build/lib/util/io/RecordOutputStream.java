// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.io;

import static com.google.common.math.IntMath.ceilingPowerOfTwo;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * A buffered output stream that only flushes its buffer at record boundaries.
 *
 * <p>The {@link #finishRecord} method marks the current position as the end of a complete record.
 * Whenever a flush occurs (either explicitly via {@link #flush} or implicitly via {@link #write} or
 * {@link #close}), the internal buffer is only flushed up to the last recorded position, with any
 * following bytes remaining in the internal buffer. The internal buffer starts at 4KB but grows to
 * accommodate the largest record seen so far.
 *
 * <p>This is intended as a best-effort attempt to prevent incomplete records from being written to
 * disk in the event of an abrupt exit. It isn't completely safe since partial underlying writes are
 * still possible, but experiments suggest that they're very unlikely for small buffer sizes.
 */
public final class RecordOutputStream extends OutputStream {
  private final OutputStream out;
  private byte[] buf = new byte[4096];
  private int writeOff = 0;
  private int flushOff = 0;

  public RecordOutputStream(OutputStream out) {
    this.out = out;
  }

  /** Marks the current position as the end of a complete record. */
  public void finishRecord() {
    flushOff = writeOff;
  }

  @Override
  public void write(int b) throws IOException {
    write(new byte[] {(byte) b}, 0, 1);
  }

  @Override
  public void write(byte[] b) throws IOException {
    write(b, 0, b.length);
  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    if (len > buf.length - writeOff) {
      // First try to make space by flushing.
      flush();
      if (len > buf.length - writeOff) {
        // If the buffer is too small to fit a single record, grow it to the next power of two.
        buf = Arrays.copyOf(buf, ceilingPowerOfTwo(writeOff + len));
      }
    }
    System.arraycopy(b, off, buf, writeOff, len);
    writeOff += len;
  }

  @Override
  public void flush() throws IOException {
    if (flushOff > 0) {
      out.write(buf, 0, flushOff);
      // TODO(tjgq): Consider using a ring buffer to avoid this copy.
      System.arraycopy(buf, flushOff, buf, 0, writeOff - flushOff);
      writeOff -= flushOff;
      flushOff = 0;
    }
  }

  @Override
  public void close() throws IOException {
    flush();
    out.close();
  }
}
