// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 * An API for reading big files through a direct byte buffer spanning a region of the file.
 * This object maintains an internal buffer, which may store all or some of the file content.
 * When a request for data is made ({@link #getBuffer(long, int) }, the implementation will
 * first determine if the requested data range is within the region specified at time of
 * construction. If it is, it checks to see if the request is within the capacity range of
 * the current internal buffer. If not, the buffer is reallocated, based at the requested offset.
 * Then the implementation checks to see if the requested data falls within the current fill limit
 * of the internal buffer. If not additional data is read from the file. Finally, a slice of
 * the internal buffer is returned, with the requested data.
 *
 * <p>This is optimized for forward scanning of files. Random access is supported, but will likely
 * be inefficient, especially if the entire file doesn't fit in the internal buffer.
 *
 * <p>Clients of this API should take care not to keep references to returned buffers indefinitely,
 * as this would prevent collection of buffers discarded by the {@code BufferedFile} object.
 */
public class BufferedFile {

   private int maxAlloc;
   private long offset;
   private long limit;
   private FileChannel channel;
   private ByteBuffer current;
   private long currOff;

  /**
   * Same as {@code BufferedFile(channel, 0, channel.size(), blockSize)}.
   *
   * @param channel file channel opened for reading.
   * @param blockSize maximum buffer allocation.
   * @throws NullPointerException if {@code channel} is {@code null}.
   * @throws IllegalArgumentException if {@code maxAlloc}, {@code off}, or {@code len} are negative
   * or if {@code off + len > channel.size()}.
   * @throws IOException
   */
  public BufferedFile(FileChannel channel, int blockSize) throws IOException {
    this(channel, 0, channel.size(), blockSize);
  }

  /**
   * Allocates a buffered file.
   *
   * @param channel file channel opened for reading.
   * @param off the first byte that can be read through this object.
   * @param len the max number of bytes that can be read through this object.
   * @param blockSize default max buffer allocation size is {@code Math.min(blockSize, len)}.
   * @throws NullPointerException if {@code channel} is {@code null}.
   * @throws IllegalArgumentException if {@code blockSize}, {@code off}, or {@code len} are negative
   * or if {@code off + len > channel.size()}.
   * @throws IOException if thrown by the underlying file channel.
   */
  public BufferedFile(FileChannel channel, long off, long len, int blockSize) throws IOException {
    Preconditions.checkNotNull(channel);
    Preconditions.checkArgument(blockSize >= 0);
    Preconditions.checkArgument(off >= 0);
    Preconditions.checkArgument(len >= 0);
    Preconditions.checkArgument(off + len <= channel.size());
    this.maxAlloc = (int) Math.min(blockSize, len);
    this.offset = off;
    this.limit = off + len;
    this.channel = channel;
    this.current = null;
    currOff = -1;
  }

  /**
   * Returns the offset of the first byte beyond the readable region.
   * @return the file offset just beyond the readable region.
   */
  public long limit() {
    return limit;
  }

  /**
   * Returns a byte buffer for reading {@code len} bytes from the {@code off} position in the file.
   * If the requested bytes are already loaded in the internal buffer, a slice is returned, with
   * position 0 and limit set to {@code len}. The slice may have a capacity greater than its limit,
   * if more bytes are already available in the internal buffer. If the requested bytes are not
   * available, but can fit in the current internal buffer, then more data is read, before a slice
   * is created as described above. If the requested data falls outside the range that can be fitted
   * into the current internal buffer, then a new internal buffer is allocated. The prior internal
   * buffer (if any), is no longer referenced by this object (but it may still be referenced by the
   * client, holding references to byte buffers returned from prior call to this method). The new
   * internal buffer will be based at {@code off} file position, and have a capacity equal to the
   * maximum of the {@code blockSize} of this buffer and {@code len}, except that it will never
   * exceed the number of bytes from {@code off} to the end of the readable region of the file
   * (min-max rule).
   *
   * @param off
   * @param len
   * @return a slice of the internal byte buffer containing the requested data. Except, if the
   *     client request data beyond the readable region of the file, the {@code len} value is
   *     reduced to the maximum number of bytes available from the given {@code off}.
   * @throws IllegalArgumentException if {@code len} is less than 0, or {@code off} is outside the
   *     readable region specified when constructing this object.
   * @throws IOException if thrown by the underlying file channel.
   */
  public synchronized ByteBuffer getBuffer(long off, int len) throws IOException {
    Preconditions.checkArgument(off >= offset);
    Preconditions.checkArgument(len >= 0);
    Preconditions.checkArgument(off < limit || (off == limit && len == 0));
    if (limit - off < len) { // never return data beyond limit
      len = (int) (limit - off);
    }
    Preconditions.checkState(off + len <= limit);
    if (current == null || off < currOff || off + len > currOff + current.capacity()) {
      allocate(off, len);
      Preconditions.checkState(current != null && off == currOff
          && off + len <= currOff + current.capacity());
    }
    Preconditions.checkState(current != null && off >= currOff
        && off + len <= currOff + current.capacity());
    if (off - currOff + len > current.limit()) {
      readMore((int) (off - currOff) + len);
    }
    Preconditions.checkState(current != null && off >= currOff
        && off + len <= currOff + current.limit());
    current.position((int) (off - currOff));
    return (ByteBuffer) current.slice().limit(len);
  }

  private void readMore(int newMin) throws IOException {
    channel.position(currOff + current.limit());
    current.position(current.limit());
    current.limit(current.capacity());
    do {
      channel.read(current);
    } while(current.position() < newMin);
    current.limit(current.position()).position(0);
  }

  private void allocate(long off, int len) {
    current = ByteBuffer.allocateDirect(bufferSize(off, len));
    current.limit(0);
    currOff = off;
  }

  private int bufferSize(long off, int len) {
    return (int) Math.min(Math.max(len, maxAlloc), limit - off);
  }
}
