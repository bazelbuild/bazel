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
//
package com.google.devtools.build.lib.vfs.inmemoryfs;

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.common.math.IntMath;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ClosedChannelException;
import java.nio.channels.NonReadableChannelException;
import java.nio.channels.NonWritableChannelException;
import java.nio.channels.SeekableByteChannel;
import java.util.Arrays;
import javax.annotation.concurrent.GuardedBy;

/**
 * InMemoryFileInfo manages file contents by storing them entirely in memory.
 */
@ThreadSafe
public class InMemoryFileInfo extends FileInfo {

  // The minimum storage size, to avoid small reallocations.
  private static final int MIN_SIZE = 32;

  // The maximum file size. For simplicity, use the largest power of two representable as an int.
  private static final int MAX_SIZE = 1 << 30;

  // A byte array storing the file contents, possibly with extra unused bytes at the end.
  @GuardedBy("this")
  private byte[] content;

  // The file size.
  @GuardedBy("this")
  private int size;

  InMemoryFileInfo(Clock clock) {
    super(clock);
    // New files start out empty.
    content = new byte[MIN_SIZE];
    size = 0;
  }

  @Override
  public synchronized long getSize() {
    return size;
  }

  @Override
  public byte[] getxattr(String name) {
    return null;
  }

  @Override
  public byte[] getFastDigest() {
    return null;
  }

  @Override
  public InputStream getInputStream() {
    return Channels.newInputStream(
        new InMemoryByteChannel(
            /* readable= */ true,
            /* writable= */ false,
            /* append= */ false,
            /* truncate= */ false));
  }

  @Override
  public OutputStream getOutputStream(boolean append) {
    return Channels.newOutputStream(
        new InMemoryByteChannel(
            /* readable= */ false,
            /* writable= */ true,
            /* append= */ append,
            /* truncate= */ !append));
  }

  @Override
  public SeekableByteChannel createReadWriteByteChannel() {
    return new InMemoryByteChannel(
        /* readable= */ true, /* writable= */ true, /* append= */ false, /* truncate= */ true);
  }

  /**
   * A {@link SeekableByteChannel} manipulating the contents of the parent {@link InMemoryFileInfo}
   * instance.
   *
   * <p>Supports concurrent operations, possibly through multiple channels.
   */
  private final class InMemoryByteChannel implements SeekableByteChannel {
    private final boolean readable;
    private final boolean writable;
    private final boolean append;
    private boolean closed = false;
    private int position = 0;

    InMemoryByteChannel(boolean readable, boolean writable, boolean append, boolean truncate) {
      this.readable = readable;
      this.writable = writable;
      this.append = append;

      if (truncate) {
        synchronized (InMemoryFileInfo.this) {
          size = 0;
        }
      }
    }

    private void ensureOpen() throws IOException {
      if (closed) {
        throw new ClosedChannelException();
      }
    }

    private void ensureReadable() {
      if (!readable) {
        throw new NonReadableChannelException();
      }
    }

    private void ensureWritable() {
      if (!writable) {
        throw new NonWritableChannelException();
      }
    }

    private int checkSize(long size) throws IOException {
      if (size > MAX_SIZE) {
        throw new IOException("InMemoryFileSystem does not support files larger than 1GB");
      }
      return (int) size;
    }

    private void maybeGrow(int newSize) {
      synchronized (InMemoryFileInfo.this) {
        if (newSize <= content.length) {
          return;
        }
        content = Arrays.copyOf(content, IntMath.ceilingPowerOfTwo(newSize));
      }
    }

    @Override
    public synchronized boolean isOpen() {
      return !closed;
    }

    @Override
    public synchronized void close() {
      closed = true;
    }

    @Override
    public synchronized int read(ByteBuffer dst) throws IOException {
      ensureOpen();
      ensureReadable();
      synchronized (InMemoryFileInfo.this) {
        if (position >= size) {
          // End of file.
          return -1;
        }
        int len = min(dst.remaining(), size - position);
        if (len == 0) {
          return 0;
        }
        dst.put(content, position, len);
        position += len;
        return len;
      }
    }

    @Override
    public synchronized int write(ByteBuffer src) throws IOException {
      ensureOpen();
      ensureWritable();
      synchronized (InMemoryFileInfo.this) {
        if (append) {
          position = size;
        }
        int len = src.remaining();
        if (len == 0) {
          // Zero write should not cause hole to be filled below.
          return 0;
        }
        int newSize = checkSize(max(size, (long) position + len));
        maybeGrow(newSize);
        if (position > size) {
          // Fill hole left by previous seek, as it's not guaranteed to have been freshly allocated.
          Arrays.fill(content, size, position, (byte) 0);
        }
        src.get(content, position, len);
        position += len;
        size = newSize;
        markModificationTime();
        return len;
      }
    }

    @Override
    public synchronized long position() throws IOException {
      ensureOpen();
      return position;
    }

    @Override
    public synchronized SeekableByteChannel position(long newPosition) throws IOException {
      checkArgument(newPosition >= 0, "new position must be non-negative: %s", newPosition);
      ensureOpen();
      position = checkSize(newPosition);
      return this;
    }

    @Override
    public synchronized long size() throws IOException {
      ensureOpen();
      synchronized (InMemoryFileInfo.this) {
        return size;
      }
    }

    @Override
    public synchronized SeekableByteChannel truncate(long newSize) throws IOException {
      checkArgument(newSize >= 0, "new size must be non-negative: %s", newSize);
      ensureOpen();
      ensureWritable();
      int truncatedSize = checkSize(newSize);
      synchronized (InMemoryFileInfo.this) {
        if (truncatedSize < size) {
          size = truncatedSize;
          markModificationTime();
        }
        if (position > truncatedSize) {
          position = truncatedSize;
        }
        return this;
      }
    }
  }
}
