// Copyright 2014 The Bazel Authors. All rights reserved.
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

import java.io.IOException;
import java.io.OutputStream;

/**
 * This stream maintains a buffer, which it flushes upon encountering bytes
 * that might be new line characters. This stream implements {@link #close()}
 * as {@link #flush()}.
 */
abstract class LineFlushingOutputStream extends OutputStream {

  static final int BUFFER_LENGTH = 8192;
  protected static byte NEWLINE = '\n';

  /**
   * The buffer containing the characters that have not been flushed yet.
   */
  protected final byte[] buffer = new byte[BUFFER_LENGTH];

  /**
   * The length of the buffer that's actually used.
   */
  protected int len = 0;

  @Override
  public synchronized void write(byte[] b, int off, int inlen)
      throws IOException {
    if (len == BUFFER_LENGTH) {
      flush();
    }
    int charsInLine = 0;
    while(inlen > charsInLine) {
      boolean sawNewline = (b[off + charsInLine] == NEWLINE);
      charsInLine++;
      if (sawNewline || len + charsInLine == BUFFER_LENGTH) {
        System.arraycopy(b, off, buffer, len, charsInLine);
        len += charsInLine;
        off += charsInLine;
        inlen -= charsInLine;
        flush();
        charsInLine = 0;
      }
    }
    System.arraycopy(b, off, buffer, len, charsInLine);
    len += charsInLine;
  }

  @Override
  public void write(int byteAsInt) throws IOException {
    byte b = (byte) byteAsInt; // make sure we work with bytes in comparisons
    write(new byte[] {b}, 0, 1);
  }

  /**
   * Close is implemented as {@link #flush()}. Client code must close the
   * underlying output stream itself in case that's desired.
   */
  @Override
  public synchronized void close() throws IOException {
    flush();
  }

  @Override
  public final synchronized void flush() throws IOException {
    flushingHook(); // The point of using a hook is to make it synchronized.
  }

  /**
   * The implementing class must define this method, which must at least flush
   * the bytes in {@code buffer[0] - buffer[len - 1]}, and reset {@code len=0}.
   *
   * Don't forget to synchronized the implementation of this method on whatever
   * underlying object it writes to!
   */
  protected abstract void flushingHook() throws IOException;

}
