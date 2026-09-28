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

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicReference;

/**
 * An {@link java.io.InputStream} that reads the bytes written by a {@link DeterministicWriter}
 * through a bounded pipe fed from a virtual thread.
 *
 * <p>See {@link DeterministicWriter#getInputStream(int)} for the contract.
 */
final class DeterministicWriterInputStream extends FilterInputStream {
  private static final ThreadFactory WRITER_THREAD_FACTORY =
      Thread.ofVirtual().name("deterministic-writer-pipe-", 0).factory();

  private final Thread writerThread;
  private final AtomicReference<Throwable> failure;

  DeterministicWriterInputStream(DeterministicWriter writer, int bufferSize) {
    var pipedIn = new PipedInputStream(bufferSize);
    PipedOutputStream pipedOut;
    try {
      pipedOut = new PipedOutputStream(pipedIn);
    } catch (IOException e) {
      throw new IllegalStateException("PipedOutputStream constructor is not expected to throw", e);
    }
    var failure = new AtomicReference<Throwable>();
    // The writer only captures locals so that it can't observe this stream before its
    // construction is complete.
    var writerThread =
        WRITER_THREAD_FACTORY.newThread(
            () -> {
              try (pipedOut) {
                // Publish failures before closing the pipe, so EOF cannot race with the failure.
                try {
                  writer.writeTo(pipedOut);
                } catch (Throwable t) {
                  failure.set(t);
                }
              } catch (IOException e) {
                failure.compareAndSet(null, e);
              }
            });
    super(pipedIn);
    this.failure = failure;
    this.writerThread = writerThread;
    writerThread.start();
  }

  @Override
  public int read() throws IOException {
    return checkResult(in.read());
  }

  @Override
  public int read(byte[] bytes, int offset, int length) throws IOException {
    return checkResult(in.read(bytes, offset, length));
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
      writerThread.interrupt();
    }
  }
}
