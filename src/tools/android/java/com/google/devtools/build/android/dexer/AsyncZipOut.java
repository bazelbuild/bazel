// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static java.util.concurrent.TimeUnit.MINUTES;

import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * {@link ZipOutputStream} wrapper that {@link #writeAsync writes} zip entries asynchronously in the
 * given order.  It's essential to eventually {@link #close} to block until writing is finished.
 */
public class AsyncZipOut implements Closeable {
  /** A single thread used to write zip entries sequentially in the given order. */
  private final ExecutorService writerThread = Executors.newSingleThreadExecutor();
  /** The first exception writing to {@link #out}, if any. */
  private final AtomicReference<Throwable> exception = new AtomicReference<>(null);

  private final Path dest; // for exception messages
  /**
   * The underlying zip output.  This field must be exclusively accessed through tasks enqueued in
   * {@link #writerThread}.
   */
  private final ZipOutputStream out;

  AsyncZipOut(Path dest, OpenOption... options) throws IOException {
    this(dest, new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(dest, options))));
  }

  private AsyncZipOut(Path dest, ZipOutputStream out) {
    this.dest = dest;
    this.out = out;
  }

  /**
   * Enqueues a zip entry to write after any already enqueued entries, unless errors have occurred
   * or {@link #finishAsync} or {@link #close} have been called.
   *
   * @throws IOException exceptions that occurred writing any previously enqueued entries
   */
  void writeAsync(ZipEntry entry, byte[] content) throws IOException {
    checkPendingException(); // fail fast
    checkArgument(entry.getMethod() == ZipEntry.STORED);
    checkArgument(entry.getSize() == content.length);
    writerThread.execute(
        () -> {
          try {
            out.putNextEntry(entry);
            out.write(content);
            out.closeEntry();
          } catch (Throwable e) {
            exception.compareAndSet(null, e);
          }
        });
  }

  /**
   * After any pending writes are done, this closes the underlying {@link ZipOutputStream}, which
   * appends the zip file's central directory to the end of the file.
   */
  void finishAsync() {
    if (writerThread.isShutdown()) {
      return;
    }
    writerThread.execute(
        () -> {
          try {
            out.close();
          } catch (Throwable e) {
            exception.compareAndSet(null, e);
          }
        });
    writerThread.shutdown();
  }

  @Override
  public void close() throws IOException {
    finishAsync();
    try {
      checkState(writerThread.awaitTermination(1, MINUTES), "Didn't finish writing in time");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException(e);
    }
    checkPendingException();
  }

  private void checkPendingException() throws IOException {
    Throwable e = exception.get();
    if (e != null) {
      writerThread.shutdownNow(); // abort pending writes, since we're failed anyways
      throw new IOException("Asynchronous exception writing " + dest, e);
    }
  }

  @Override
  protected void finalize() throws Throwable {
    close(); // Call close() so we get any exceptions, but note this may block
  }
}
