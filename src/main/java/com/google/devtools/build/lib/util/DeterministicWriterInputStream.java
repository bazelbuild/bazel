// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.io.CountingOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicReference;

/**
 * An input stream over the contents produced by a {@link DeterministicWriter}.
 *
 * <p>Reads pull the contents through a bounded pipe from a writer thread started on the first read.
 * {@link #transferTo} on an unread stream bypasses the pipe and runs the writer on the calling
 * thread.
 */
final class DeterministicWriterInputStream extends FilterInputStream {
  private static final ThreadFactory WRITER_THREAD_FACTORY =
      Thread.ofVirtual().name("deterministic-writer-pipe-", 0).factory();

  private final DeterministicWriter writer;
  private final PipedOutputStream pipedOut;
  private final AtomicReference<Throwable> failure = new AtomicReference<>();
  // Started on the first read.
  private Thread writerThread;
  // Set by a direct transferTo.
  private boolean transferred;

  DeterministicWriterInputStream(DeterministicWriter writer, int bufferSize) {
    super(new PipedInputStream(bufferSize));
    this.writer = writer;
    try {
      this.pipedOut = new PipedOutputStream((PipedInputStream) in);
    } catch (IOException e) {
      throw new IllegalStateException("PipedOutputStream constructor is not expected to throw", e);
    }
  }

  private void ensureWriterStarted() {
    if (writerThread != null) {
      return;
    }
    writerThread =
        WRITER_THREAD_FACTORY.newThread(
            () -> {
              try (pipedOut) {
                try {
                  writer.writeTo(pipedOut);
                } catch (Throwable t) {
                  failure.set(t);
                }
              } catch (IOException e) {
                failure.compareAndSet(null, e);
              }
            });
    writerThread.start();
  }

  @Override
  public int read() throws IOException {
    if (transferred) {
      return -1;
    }
    ensureWriterStarted();
    return checkResult(in.read());
  }

  @Override
  public int read(byte[] bytes, int offset, int length) throws IOException {
    if (transferred) {
      return length == 0 ? 0 : -1;
    }
    ensureWriterStarted();
    return checkResult(in.read(bytes, offset, length));
  }

  @Override
  public long skip(long n) throws IOException {
    if (transferred) {
      return 0;
    }
    ensureWriterStarted();
    return in.skip(n);
  }

  @Override
  public long transferTo(OutputStream out) throws IOException {
    if (transferred) {
      return 0;
    }
    if (writerThread != null) {
      // Some contents may have been read already, so the rest has to come from the pipe.
      return super.transferTo(out);
    }
    // Nothing has been read yet, so the writer can write to the target directly.
    transferred = true;
    var countingOut = new CountingOutputStream(out);
    writer.writeTo(countingOut);
    return countingOut.getCount();
  }

  private int checkResult(int result) throws IOException {
    if (result == -1 && failure.get() != null) {
      throw new IOException("Failed to write stream contents", failure.get());
    }
    return result;
  }

  @Override
  public void close() throws IOException {
    try {
      super.close();
    } finally {
      if (writerThread != null) {
        writerThread.interrupt();
      }
    }
  }
}
