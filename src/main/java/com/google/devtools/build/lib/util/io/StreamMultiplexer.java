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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Instances of this class are multiplexers, which redirect multiple
 * output streams into a single output stream with tagging so it can be
 * de-multiplexed into multiple streams as needed. This allows us to
 * use one connection for multiple streams, but more importantly it avoids
 * multiple threads or select etc. on the receiving side: A client on the other
 * end of a networking connection can simply read the tagged lines and then act
 * on them within a sigle thread.
 *
 * The format of the tagged output stream is reasonably simple:
 * <ol>
 *   <li>
 *     Marker byte indicating whether that chunk is for stdout (1), stderr (2) or the control
 *     stream (3).
 *   </li>
 *   <li>
 *     4 bytes indicating the length of the chunk in high-endian format.
 *   </li>
 *   <li>
 *     The payload (as many bytes as the length field before)
 *   </li>
 * </ol>>
 *
 *
 */
@ThreadSafe
public final class StreamMultiplexer {

  public static final byte STDOUT_MARKER = 1;
  public static final byte STDERR_MARKER = 2;
  public static final byte CONTROL_MARKER = 3;

  private final Object mutex = new Object();
  private final OutputStream multiplexed;

  public StreamMultiplexer(OutputStream multiplexed) {
    this.multiplexed = multiplexed;
  }

  private class MarkingStream extends LineFlushingOutputStream {

    private final byte markerByte;

    MarkingStream(byte markerByte) {
      this.markerByte = markerByte;
    }

    @Override
    protected void flushingHook() throws IOException {
      synchronized (mutex) {
        if (len == 0) {
          multiplexed.flush();
          return;
        }

        multiplexed.write(markerByte);
        multiplexed.write((len >> 24) & 0xff);
        multiplexed.write((len >> 16) & 0xff);
        multiplexed.write((len >> 8) & 0xff);
        multiplexed.write(len & 0xff);
        multiplexed.write(buffer, 0, len);
        multiplexed.flush();
      }
      len = 0;
    }

  }

  /**
   * Create a stream that will tag its contributions into the multiplexed stream
   * with the marker '1', which means 'stdout'. Each newline byte leads
   * to a forced automatic flush. Also, this stream never closes the underlying
   * stream it delegates to - calling its {@code close()} method is equivalent
   * to calling {@code flush}.
   */
  public OutputStream createStdout() {
    return new MarkingStream(STDOUT_MARKER);
  }

  /**
   * Like {@link #createStdout()}, except it tags with the marker '2' to
   * indicate 'stderr'.
   */
  public OutputStream createStderr() {
    return new MarkingStream(STDERR_MARKER);
  }

  /**
   * Like {@link #createStdout()}, except it tags with the marker '3' to
   * indicate control flow..
   */
  public OutputStream createControl() {
    return new MarkingStream(CONTROL_MARKER);
  }

}
