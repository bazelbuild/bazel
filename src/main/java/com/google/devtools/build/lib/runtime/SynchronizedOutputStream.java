// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.runtime;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * {@link OutputStream} suitably synchronized for producer-consumer use cases. The method {@link
 * #readAndReset()} allows to read the bytes accumulated so far and simultaneously truncate
 * precisely the bytes read. Moreover, upon such a reset the amount of memory retained is reset to a
 * small constant. This is a difference with resecpt to the behaviour of the standard classes {@link
 * ByteArrayOutputStream} which only resets the index but keeps the array. This difference matters,
 * as we need to support output peeks without retaining this amount of memory for the rest of the
 * build.
 *
 * <p>This class is expected to be used with the {@link BuildEventStreamer}.
 */
public class SynchronizedOutputStream extends OutputStream {

  // The maximal amount of bytes we intend to store in the buffer. However,
  // the requirement that a single write be written in one go is more important,
  // so the actual size we store in this buffer can be the maximum (not the sum)
  // of this value and the amount of bytes written in a single call to the
  // {@link write(byte[] buffer, int offset, int count)} method.
  private final int maxBufferedLength;

  private final int maxChunkSize;

  private byte[] buf;
  private long count;

  // The event streamer that is supposed to flush stdout/stderr.
  private BuildEventStreamer streamer;

  public SynchronizedOutputStream(int maxBufferedLength, int maxChunkSize) {
    Preconditions.checkArgument(maxChunkSize > 0);
    buf = new byte[64];
    count = 0;
    this.maxBufferedLength = maxBufferedLength;
    this.maxChunkSize = Math.max(maxChunkSize, maxBufferedLength);
  }

  public void registerStreamer(BuildEventStreamer streamer) {
    this.streamer = streamer;
  }

  /**
   * Read the contents of the stream and simultaneously clear them. Also, reset the amount of memory
   * retained to a constant amount.
   */
  public synchronized Iterable<String> readAndReset() {
    String content = new String(buf, 0, (int) count, UTF_8);
    buf = new byte[64];
    count = 0;
    return content.isEmpty()
        ? ImmutableList.of()
        : Splitter.fixedLength(maxChunkSize).split(content);
  }

  @Override
  public void write(int oneByte) throws IOException {
    // We change the dependency with respect to that of the super class: write(int)
    // now calls write(int[], int, int) which is implemented without any dependencies.
    write(new byte[] {(byte) oneByte}, 0, 1);
  }

  @Override
  public void write(byte[] buffer, int offset, int count) throws IOException {
    // As we base the less common write(int) on this method, we may not depend not call write(int)
    // directly or indirectly (e.g., by calling super.write(int[], int, int)).
    boolean shouldFlush = false;
    // As we have to do the flushing outside the synchronized block, we have to expect
    // other writes to come immediately after flushing, so we have to do the check inside
    // a while loop.
    boolean didWrite = false;
    while (!didWrite) {
      synchronized (this) {
        if (this.count + (long) count < maxBufferedLength || this.count == 0) {
          if (this.count + (long) count >= (long) buf.length) {
            // We need to increase the buffer; if within the permissible range range for array
            // sizes, we at least double it, otherwise we only increase as far as needed.
            long newsize;
            if (2 * (long) buf.length + count < (long) Integer.MAX_VALUE) {
              newsize = 2 * (long) buf.length + count;
            } else {
              newsize = this.count + count;
            }
            byte[] newbuf = new byte[(int) newsize];
            System.arraycopy(buf, 0, newbuf, 0, (int) this.count);
            this.buf = newbuf;
          }
          System.arraycopy(buffer, offset, buf, (int) this.count, count);
          this.count += (long) count;
          didWrite = true;
        } else {
          shouldFlush = true;
        }
        if (this.count >= maxBufferedLength) {
          shouldFlush = true;
        }
      }
      if (shouldFlush && streamer != null) {
        streamer.flush();
        shouldFlush = false;
      }
    }
  }
}
