// Copyright 2014 Google Inc. All rights reserved.
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

  private static final byte AT = '@';
  private static final byte NEWLINE = '\n';

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
    EXPECT_CONTROL_STARTING_AT,
    EXPECT_MARKER_BYTE,
    EXPECT_AT_OR_NEWLINE,
    EXPECT_PAYLOAD_OR_NEWLINE
  }

  private State state = State.EXPECT_CONTROL_STARTING_AT;
  private boolean addNewlineToPayload;
  private OutputStream selectedStream;

  /**
   * Construct a new demultiplexer. The {@code smallestMarkerByte} indicates
   * the marker byte we would expect for {@code outputStreams[0]} to be used.
   * So, if this first stream is your stdout and you're using the
   * {@link StreamMultiplexer}, then you will need to set this to
   * {@code '1'}. Because {@link StreamDemultiplexer} extends
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
      case EXPECT_CONTROL_STARTING_AT:
        parseControlStartingAt((byte) b);
        resetFields();
        break;
      case EXPECT_MARKER_BYTE:
        parseMarkerByte((byte) b);
        break;
      case EXPECT_AT_OR_NEWLINE:
        parseAtOrNewline((byte) b);
        break;
      case EXPECT_PAYLOAD_OR_NEWLINE:
        parsePayloadOrNewline((byte) b);
        break;
    }
  }

  /**
   * Handles {@link State#EXPECT_PAYLOAD_OR_NEWLINE}, which is the payload
   * we are actually transporting over the wire. At this point we can rely
   * on a stream having been preselected into {@link #selectedStream}, and
   * also we will add a newline if {@link #addNewlineToPayload} is set.
   * Flushes at the end of every payload segment.
   */
  private void parsePayloadOrNewline(byte b) throws IOException {
    if (b == NEWLINE) {
      if (addNewlineToPayload) {
        selectedStream.write(NEWLINE);
      }
      selectedStream.flush();
      state = State.EXPECT_CONTROL_STARTING_AT;
    } else {
      selectedStream.write(b);
      selectedStream.flush(); // slow?
    }
  }

  /**
   * Handles {@link State#EXPECT_AT_OR_NEWLINE}, which is either the
   * suppress newline indicator (at) at the end of a control line, or the end
   * of a control line.
   */
  private void parseAtOrNewline(byte b) throws IOException {
    if (b == NEWLINE) {
      state = State.EXPECT_PAYLOAD_OR_NEWLINE;
    } else if (b == AT) {
      addNewlineToPayload = false;
    } else {
      throw new IOException("Expected @ or \\n. (" + b + ")");
    }
  }

  /**
   * Reset the fields that are affected by our state.
   */
  private void resetFields() {
    selectedStream = null;
    addNewlineToPayload = true;
  }

  /**
   * Handles {@link State#EXPECT_MARKER_BYTE}. The byte determines which stream
   * we will be using, and will set {@link #selectedStream}.
   */
  private void parseMarkerByte(byte markerByte) throws IOException {
    if (markerByte < 0 || markerByte > Byte.MAX_VALUE) {
      String msg = "Illegal marker byte (" + markerByte + ")";
      throw new IllegalArgumentException(msg);
    }
    if (markerByte > outputStreams.length
        || outputStreams[markerByte] == null) {
      throw new IOException("stream " + markerByte + " not registered.");
    }
    selectedStream = outputStreams[markerByte];
    state = State.EXPECT_AT_OR_NEWLINE;
  }

  /**
   * Handles {@link State#EXPECT_CONTROL_STARTING_AT}, the very first '@' with
   * which each message starts.
   */
  private void parseControlStartingAt(byte b) throws IOException {
    if (b != AT) {
      throw new IOException("Expected control starting @. (" + b + ", "
          + (char) b + ")");
    }
    state = State.EXPECT_MARKER_BYTE;
  }

}
