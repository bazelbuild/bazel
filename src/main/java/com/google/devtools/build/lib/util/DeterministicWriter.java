// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.ByteString;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A {@link DeterministicWriter} writes a stream of bytes to an {@link OutputStream}.
 *
 * <p>The same stream of bytes is written on every invocation of {@link #writeTo}.
 */
public interface DeterministicWriter {
  /**
   * Writes the stream of bytes to the given {@link OutputStream}.
   *
   * <p>Every invocation of this method writes the same stream of bytes.
   *
   * <p>Implementations
   *
   * <ul>
   *   <li>must not close the given {@link OutputStream}
   *   <li>may flush the given {@link OutputStream}
   *   <li>should not wrap the given {@link OutputStream} in a buffered stream. The caller is
   *       responsible for providing a buffered stream if necessary.
   * </ul>
   *
   * @param out the {@link OutputStream} to write to
   * @throws IOException only if out throws an IOException
   */
  void writeTo(OutputStream out) throws IOException;

  /**
   * Returns the stream of bytes as a {@link ByteString}.
   *
   * <p>May be used to avoid unnecessary copying by callers that only need a {@link ByteString}.
   *
   * <p>The default implementation calls {@link #writeTo} on a fresh {@link ByteString.Output} and
   * returns the resulting {@link ByteString}. Other implementations may provide a more efficient
   * alternative.
   */
  default ByteString getBytes() throws IOException {
    ByteString.Output out = ByteString.newOutput();
    writeTo(out);
    return out.toByteString();
  }

  /**
   * Provides an {@link InputStream} that reads the contents without materializing them entirely in
   * memory. The pipe buffers at most {@code bufferSize} bytes, in addition to any memory used by
   * the writer itself.
   *
   * <p>The writer runs on a virtual thread and blocks when the buffer is full. The caller must
   * close the stream to stop the writer if it does not read to the end. Writer failures are
   * propagated to the reader as {@link IOException}s.
   */
  default InputStream getInputStream(int bufferSize) {
    var pipedIn = new PipedInputStream(bufferSize);
    PipedOutputStream pipedOut;
    try {
      pipedOut = new PipedOutputStream(pipedIn);
    } catch (IOException e) {
      throw new IllegalStateException("PipedOutputStream constructor is not expected to throw", e);
    }
    var failure = new AtomicReference<Throwable>();
    Thread writerThread =
        Thread.ofVirtual()
            .name("deterministic-writer-pipe")
            .start(
                () -> {
                  try (pipedOut) {
                    // Publish failures before closing the pipe, so EOF cannot race with the
                    // failure.
                    try {
                      writeTo(pipedOut);
                    } catch (Throwable t) {
                      failure.set(t);
                    }
                  } catch (IOException e) {
                    failure.compareAndSet(null, e);
                  }
                });
    return new FilterInputStream(pipedIn) {
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
    };
  }
}
