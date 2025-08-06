package com.google.devtools.build.lib.util;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Objects;
import jdk.internal.vm.Continuation;
import jdk.internal.vm.ContinuationScope;

/**
 * Adapts a {@link DeterministicWriter} into an {@link InputStream} with fixed memory overhead and
 * on a single thread using {@link Continuation}.
 */
public final class DeterministicWriterInputStream extends InputStream {
  // It may be tempting to use a PipedInputStream/PipedOutputStream pair in a virtual thread to
  // avoid reliance on JDK internal APIs, but this may result in livelocks if the pace of writer and
  // reader matches, with neither yielding the carrier thread.

  private final byte[] buffer;
  private final int capacity;
  private final long mask;
  private final byte[] singleByte = new byte[1];
  private final ContinuationScope scope =
      new ContinuationScope(DeterministicWriterInputStream.class.getSimpleName());
  private final Continuation continuation;

  // Invariants:
  // * writePos and readPos increase monotonically.
  // * writePos >= readPos at all times.
  // * writePos - readPos <= capacity at all times.
  private long readPos = 0;
  private long writePos = 0;

  public DeterministicWriterInputStream(DeterministicWriter writer) {
    this(writer, 8192);
  }

  public DeterministicWriterInputStream(DeterministicWriter writer, int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("Buffer capacity must be positive");
    }
    // Round to the next power of 2 to simplify the buffer wrapping logic.
    this.capacity = 1 << (32 - Integer.numberOfLeadingZeros(capacity - 1));
    this.mask = this.capacity - 1;
    this.buffer = new byte[this.capacity];
    this.continuation =
        new Continuation(
            scope,
            () -> {
              try {
                writer.writeTo(new OutPipe());
              } catch (IOException e) {
                throw new IllegalStateException(
                    "writeTo not expected to throw since OutPipe doesn't", e);
              }
            });
  }

  @Override
  public int available() {
    return (int) (writePos - readPos);
  }

  @Override
  public int read() {
    return read(singleByte, 0, 1) == -1 ? -1 : singleByte[0] & 0xFF;
  }

  @Override
  public int read(byte[] b, int off, int len) {
    Objects.requireNonNull(b);
    Objects.checkFromIndexSize(off, len, b.length);
    if (len == 0) {
      return 0;
    }

    int originalLen = len;
    while (true) {
      long rp = readPos;
      long wp = writePos;
      int bytesToRead = Math.min((int) (wp - rp), len);
      if (bytesToRead > 0) {
        // The mask rounds to a power of two, so truncation to int is safe.
        int start = (int) (rp & mask);
        int end = (int) ((rp + bytesToRead) & mask);
        if (start < end) {
          System.arraycopy(buffer, start, b, off, bytesToRead);
        } else {
          int firstChunk = capacity - start;
          System.arraycopy(buffer, start, b, off, firstChunk);
          System.arraycopy(buffer, 0, b, off + firstChunk, end);
        }

        off += bytesToRead;
        len -= bytesToRead;
        readPos = rp + bytesToRead;

        if (len == 0) {
          return originalLen;
        }
      }

      // Need more data from the writer if available.
      if (continuation.isDone()) {
        int written = originalLen - len;
        return written == 0 ? -1 : written;
      }
      continuation.run();
    }
  }

  @Override
  public long skip(long n) {
    if (n < 0) {
      return 0;
    }

    long remaining = n;
    while (true) {
      long rp = readPos;
      long wp = writePos;
      // wp - rp <= capacity, which is always an int.
      int bytesToSkip = (int) Math.min(wp - rp, remaining);
      if (bytesToSkip > 0) {
        remaining -= bytesToSkip;
        readPos = rp + bytesToSkip;

        if (remaining == 0) {
          return n;
        }
      }

      // Need more data from the writer if available.
      if (continuation.isDone()) {
        return n - remaining;
      }
      continuation.run();
    }
  }

  @Override
  public void close() {}

  private final class OutPipe extends OutputStream {

    @Override
    public void write(int b) {
      singleByte[0] = (byte) b;
      write(singleByte, 0, 1);
    }

    @Override
    public void write(byte[] b, int off, int len) {
      Objects.requireNonNull(b);
      Objects.checkFromIndexSize(off, len, b.length);
      if (len == 0) {
        return;
      }

      while (true) {
        long rp = readPos;
        long wp = writePos;
        int bytesToWrite = Math.min(capacity - (int) (wp - rp), len);
        if (bytesToWrite > 0) {
          int start = (int) (wp & mask);
          int end = (int) ((wp + bytesToWrite) & mask);
          if (start < end) {
            System.arraycopy(b, off, buffer, start, bytesToWrite);
          } else {
            int firstChunk = capacity - start;
            System.arraycopy(b, off, buffer, start, firstChunk);
            System.arraycopy(b, off + firstChunk, buffer, 0, end);
          }

          off += bytesToWrite;
          len -= bytesToWrite;
          writePos = wp + bytesToWrite;

          if (len == 0) {
            return;
          }
        }

        // No more space in the buffer, yield to the reader to clear it.
        Continuation.yield(scope);
      }
    }

    @Override
    public void close() {}
  }
}
