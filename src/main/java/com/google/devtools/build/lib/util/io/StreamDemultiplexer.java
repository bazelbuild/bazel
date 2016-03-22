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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;

import java.io.IOException;
import java.io.OutputStream;

/**
 * The dual of {@link StreamMultiplexer}: This is an output stream into which
 * you can dump the multiplexed stream, and it delegates the de-multiplexed
 * content back into separate channels (instances of {@link OutputStream}).
 *
 * The format of the tagged output stream is as follows:
 *
 * <pre>
 * combined :: = [ control_line payload ... ]+
 * control_line :: = '@' marker '@'? '\n'
 * payload :: = r'^[^\n]*\n'
 * </pre>
 *
 * For more details, please see {@link StreamMultiplexer}.
 */
@ThreadCompatible
public final class StreamDemultiplexer extends OutputStream {

  @Override
  public void close() throws IOException {
    flush();
  }

  @Override
  public void flush() throws IOException {
    if (selectedStream != null) {
      selectedStream.flush();
    }
  }

  /**
   * The output streams, conveniently in an array indexed by the marker byte.
   * Some of these will be null, most likely.
   */
  private final OutputStream[] outputStreams =
    new OutputStream[Byte.MAX_VALUE + 1];

  /**
   * Each state in this FSM corresponds to a position in the grammar, which is
   * simple enough that we can just move through it from beginning to end as we
   * parse things.
   */
  private enum State {
    EXPECT_MARKER_BYTE,
    EXPECT_SIZE,
    EXPECT_PAYLOAD,
  }

  private final int[] sizeBuffer = new int[4];
  private State state = State.EXPECT_MARKER_BYTE;
  private OutputStream selectedStream;
  private int currentSizeByte = 0;
  private int payloadBytesLeft = 0;

  /**
   * Construct a new demultiplexer. The {@code smallestMarkerByte} indicates
   * the marker byte we would expect for {@code outputStreams[0]} to be used.
   * So, if this first stream is your stdout and you're using the
   * {@link StreamMultiplexer}, then you will need to set this to
   * {@code 1}. Because {@link StreamDemultiplexer} extends
   * {@link OutputStream}, this constructor effectively creates an
   * {@link OutputStream} instance which demultiplexes the tagged data client
   * code writes to it into {@code outputStreams}.
   */
  public StreamDemultiplexer(byte smallestMarkerByte,
                             OutputStream... outputStreams) {
    for (int i = 0; i < outputStreams.length; i++) {
      this.outputStreams[smallestMarkerByte + i] = outputStreams[i];
    }
  }

  @Override
  public void write(int b) throws IOException {
    // This dispatch traverses the finite state machine / grammar.
    switch (state) {
      case EXPECT_MARKER_BYTE:
        parseMarkerByte(b);
        break;
      case EXPECT_SIZE:
        parseSize(b);
        break;
      case EXPECT_PAYLOAD:
        parsePayload(b);
        break;
    }
  }

  private void parseSize(int b) {
    sizeBuffer[currentSizeByte] = b;
    currentSizeByte += 1;
    if (currentSizeByte == 4) {
      state = State.EXPECT_PAYLOAD;
      payloadBytesLeft = (sizeBuffer[0] << 24)
          + (sizeBuffer[1] << 16)
          + (sizeBuffer[2] << 8)
          + sizeBuffer[3];
    }
  }

  /**
   * Handles {@link State#EXPECT_MARKER_BYTE}. The byte determines which stream
   * we will be using, and will set {@link #selectedStream}.
   */
  private void parseMarkerByte(int markerByte) throws IOException {
    if (markerByte < 0 || markerByte > Byte.MAX_VALUE) {
      String msg = "Illegal marker byte (" + markerByte + ")";
      throw new IllegalArgumentException(msg);
    }
    if (markerByte > outputStreams.length
        || outputStreams[markerByte] == null) {
      throw new IOException("stream " + markerByte + " not registered.");
    }
    selectedStream = outputStreams[markerByte];
    state = State.EXPECT_SIZE;
    currentSizeByte = 0;
  }

  private void parsePayload(int b) throws IOException {
    selectedStream.write(b);
    payloadBytesLeft -= 1;
    if (payloadBytesLeft == 0) {
      state = State.EXPECT_MARKER_BYTE;
    }
  }
}
