// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import java.io.IOException;
import java.io.OutputStream;

/**
 * A decorator output stream that does line buffering.
 */
public class LineBufferedOutputStream extends OutputStream {
  private static final int DEFAULT_BUFFER_SIZE = 1024;

  private final OutputStream wrapped;
  private final byte[] buffer;
  private int pos;

  public LineBufferedOutputStream(OutputStream wrapped) {
    this(wrapped, DEFAULT_BUFFER_SIZE);
  }

  public LineBufferedOutputStream(OutputStream wrapped, int bufferSize) {
    this.wrapped = wrapped;
    this.buffer = new byte[bufferSize];
    this.pos = 0;
  }

  private void flushBuffer() throws IOException {
    int oldPos = pos;
    // Set pos to zero first so that if the write below throws, we are still in a consistent state.
    pos = 0;
    wrapped.write(buffer, 0, oldPos);
  }

  @Override
  public synchronized void write(byte[] b, int off, int inlen) throws IOException {
    if (inlen > buffer.length * 2) {
      // Do not buffer large writes
      if (pos > 0) {
        flushBuffer();
      }
      wrapped.write(b, off, inlen);
      return;
    }

    int next = off;
    while (next < off + inlen) {
      buffer[pos++] = b[next];
      if (b[next] == '\n' || pos == buffer.length) {
        flushBuffer();
      }

      next++;
    }
  }

  @Override
  public void write(int byteAsInt) throws IOException {
    byte b = (byte) byteAsInt; // make sure we work with bytes in comparisons
    write(new byte[] {b}, 0, 1);
  }

  @Override
  public synchronized void flush() throws IOException {
    if (pos != 0) {
      wrapped.write(buffer, 0, pos);
      pos = 0;
    }
    wrapped.flush();
  }

  @Override
  public synchronized void close() throws IOException {
    flush();
    wrapped.close();
  }
}
