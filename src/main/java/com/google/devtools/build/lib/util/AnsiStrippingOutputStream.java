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

package com.google.devtools.build.lib.util;

import java.io.IOException;
import java.io.OutputStream;

/**
 * A pass-thru {@link OutputStream} that strips ANSI control codes.
 */
public class AnsiStrippingOutputStream extends OutputStream {
  // The idea is straightforward: the regexp for ANSI control codes is
  // \x1b\[[;0-9]*[a-zA-Z] . Implementing it as a stream is a little ugly,
  // though.

  private enum State {
    NORMAL,
    AFTER_ESCAPE,
    PARAMETER,
  }

  private byte[] outputBuffer;
  private int outputBufferPos;

  private static final int ESCAPE_BUFFER_LENGTH = 128;
  private byte[] escapeCodeBuffer;
  private int escapeCodeBufferPos;
  private OutputStream output;
  private State state;

  public AnsiStrippingOutputStream(OutputStream output) {
    this.output = output;
    escapeCodeBuffer = new byte[ESCAPE_BUFFER_LENGTH];
    escapeCodeBufferPos = 0;
    state = State.NORMAL;
  }

  @Override
  public synchronized void write(int b) throws IOException {
    // As per the contract of OutputStream.write(int)
    byte[] array = { (byte) (b & 0xff) };
    write(array, 0, 1);
  }

  @Override
  public synchronized void write(byte[] b, int off, int len) throws IOException {
    int i = 0;
    if (state == State.NORMAL) {

      // Avoid outputBuffer allocation entirely if that's possible
      while ((i < len) && (b[off + i] != 0x1b)) {
        i++;
      }
      if (i == len) {
        output.write(b, off, len);
        return;
      }
    }

    // In the worst case, the contents of the escape buffer and the contents
    // of the input buffer are both copied to the output, so the length of the
    // output buffer should be the sum of the length of both these buffers.
    outputBuffer = new byte[len + ESCAPE_BUFFER_LENGTH];
    System.arraycopy(b, off, outputBuffer, 0, i);
    outputBufferPos = i;

    for (; i < len; i++) {
      processByte(b[off + i]);
    }

    try {
      output.write(outputBuffer, 0, outputBufferPos);
    } finally {
      outputBuffer = null;  // Make it possible to garbage collect the array
    }
  }

  private void processByte(byte b) {
    switch (state) {
      case NORMAL:
        if (escapeCodeBufferPos != 0) {
          throw new IllegalStateException();
        }
        if (b == 0x1b) {
          state = State.AFTER_ESCAPE;
          addByteToEscapeBuffer(b);
        } else {
          dumpByte(b);
        }
        break;

      case AFTER_ESCAPE:
        if (b == '[') {
          state = State.PARAMETER;
          addByteToEscapeBuffer(b);
        } else if (b == 0x1b) {
          dumpEscapeBuffer();
          state = State.AFTER_ESCAPE;
          addByteToEscapeBuffer(b);
        } else {
          dumpEscapeBuffer();
          dumpByte(b);
          state = State.NORMAL;
        }
        break;

      case PARAMETER:
        if ((b >= '0' && b <= '9') || b == ';') {
          // Parameter continues
          addByteToEscapeBuffer(b);
        } else if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) {
          // Found a control sequence, discard it and revert to normal state
          discardEscapeBuffer();
          state = State.NORMAL;
        } else if (b == 0x1b) {
          // Another escape sequence begins immediately after, and this is
          // an illegal escape sequence
          dumpEscapeBuffer();
          state = State.AFTER_ESCAPE;
          addByteToEscapeBuffer(b);
        } else {
          // Illegal control sequence, output it
          dumpEscapeBuffer();
          state = State.NORMAL;
        }
        break;
    }
  }

  private void addByteToEscapeBuffer(byte b) {
    escapeCodeBuffer[escapeCodeBufferPos++] = b;
    if (escapeCodeBufferPos == ESCAPE_BUFFER_LENGTH) {
      // Buffer full. Assume that no sane code emits an ANSI control code this
      // long and revert to normal state.
      dumpEscapeBuffer();
      state = State.NORMAL;
    }
  }

  private void discardEscapeBuffer() {
    escapeCodeBufferPos = 0;
  }

  private void dumpByte(byte b) {
    outputBuffer[outputBufferPos++] = b;
  }

  private void dumpEscapeBuffer() {
    System.arraycopy(escapeCodeBuffer, 0,
                     outputBuffer, outputBufferPos, escapeCodeBufferPos);
    outputBufferPos += escapeCodeBufferPos;
    escapeCodeBufferPos = 0;
  }

  @Override
  public void flush() throws IOException {
    output.flush();
  }

  @Override
  public void close() throws IOException {
    output.close();
  }
}
