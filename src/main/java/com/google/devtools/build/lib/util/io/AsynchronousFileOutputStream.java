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
package com.google.devtools.build.lib.util.io;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Message;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.CompletionHandler;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicReference;

/**
 * An output stream supporting anynchronous writes, backed by a file.
 *
 * <p>We use an {@link AsynchronousFileChannel} to perform non-blocking writes to a file. It gets
 * tricky when it comes to {@link #closeAsync()}, as we may only complete the returned future when
 * all writes have completed (succeeded or failed). Thus, we use a field {@link #outstandingWrites}
 * to keep track of the number of writes that have not completed yet. It's incremented before a new
 * write and decremented after a write has completed. When it's {@code 0} it's safe to complete the
 * close future.
 */
@ThreadSafety.ThreadSafe
public class AsynchronousFileOutputStream extends OutputStream implements MessageOutputStream {
  private final AsynchronousFileChannel ch;
  private final WriteCompletionHandler completionHandler = new WriteCompletionHandler();
  // The offset in the file to begin the next write at.
  private long writeOffset;
  // Number of writes that haven't completed yet.
  private long outstandingWrites;
  // The future returned by closeAsync().
  private SettableFuture<Void> closeFuture;
  // To store any exception raised from the writes.
  private final AtomicReference<Throwable> exception = new AtomicReference<>();

  public AsynchronousFileOutputStream(String filename) throws IOException {
    this(
        AsynchronousFileChannel.open(
            Paths.get(filename),
            StandardOpenOption.WRITE,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING));
  }

  @VisibleForTesting
  public AsynchronousFileOutputStream(AsynchronousFileChannel ch) throws IOException {
    this.ch = ch;
  }

  public void write(String message) {
    write(message.getBytes(UTF_8));
  }

  /**
   * Writes a delimited protocol buffer message in the same format as {@link
   * MessageLite#writeDelimitedTo(java.io.OutputStream)}.
   *
   * <p>Unfortunately, {@link MessageLite#writeDelimitedTo(java.io.OutputStream)} may result in
   * multiple calls to write on the underlying stream, so we have to provide this method here
   * instead of the caller using it directly.
   */
  @Override
  public void write(Message m) {
    Preconditions.checkNotNull(m);
    final int size = m.getSerializedSize();
    ByteArrayOutputStream bos =
        new ByteArrayOutputStream(CodedOutputStream.computeRawVarint32Size(size) + size);
    try {
      m.writeDelimitedTo(bos);
    } catch (IOException e) {
      // This should never happen with an in-memory stream.
      exception.compareAndSet(null, new IllegalStateException(e.toString()));
      return;
    }
    write(bos.toByteArray());
  }

  @Override
  public void write(int b) {
    throw new UnsupportedOperationException();
  }

  /**
   * Writes the byte buffer into the file asynchronously.
   *
   * <p>The writes are guaranteed to land in the output file in the same order that they were
   * called; However, some writes may fail, leaving the file partially corrupted. In case a write
   * fails, an exception will be propagated in close, but remaining writes will be allowed to
   * continue.
   */
  @Override
  public synchronized void write(byte[] data) {
    Preconditions.checkNotNull(data);
    Preconditions.checkState(ch.isOpen());

    if (closeFuture != null) {
      throw new IllegalStateException("Attempting to write to stream after close");
    }

    outstandingWrites++;
    ch.write(ByteBuffer.wrap(data), writeOffset, null, completionHandler);
    writeOffset += data.length;
  }

  /* Returns whether the stream is open for writing. */
  public boolean isOpen() {
    return ch.isOpen();
  }

  /**
   * Closes the stream without waiting until pending writes are committed, and supressing errors.
   *
   * <p>Pending writes will still continue asynchronously, but any errors will be ignored.
   */
  @SuppressWarnings("FutureReturnValueIgnored")
  public void closeNow() {
    closeAsync();
  }

  /**
   * Closes the stream and blocks until all pending writes are completed.
   *
   * Throws an exception if any of the writes or the close itself have failed.
   */
  @Override
  public void close() throws IOException {
    try {
      closeAsync().get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("Write interrupted");
    } catch (ExecutionException e) {
      Throwable c = e.getCause();
      Throwables.throwIfUnchecked(c);
      if (c instanceof IOException) {
        throw (IOException) c;
      }
      throw new IOException("Exception within stream close: " + c);
    }
  }

  /**
   * Flushes the currently ongoing writes into the channel.
   *
   * Throws an exception if any of the writes or the close itself have failed.
   */
  @Override
  public void flush() throws IOException {
    ch.force(true);
  }

  /**
   * Closes the channel, if close was invoked and there are no outstanding writes. Should only be
   * called in a synchronized context.
   */
  private void closeIfNeeded() {
    if (closeFuture == null || outstandingWrites > 0) {
      return;
    }
    try {
      flush();
      ch.close();
    } catch (Exception e) {
      exception.compareAndSet(null, e);
    } finally {
      Throwable e = exception.get();
      if (e == null) {
        closeFuture.set(null);
      } else {
        closeFuture.setException(e);
      }
    }
  }

  /**
   * Returns a future that will close the stream when all pending writes are completed.
   *
   * Any failed writes will propagate an exception.
   */
  public synchronized ListenableFuture<Void> closeAsync() {
    if (closeFuture != null) {
      return closeFuture;
    }
    closeFuture = SettableFuture.create();
    closeIfNeeded();
    return closeFuture;
  }

  /**
   * Handler that's notified when a write completes.
   */
  private final class WriteCompletionHandler implements CompletionHandler<Integer, Void> {

    @Override
    public void completed(Integer result, Void attachment) {
      countWritesAndTryClose();
    }

    @Override
    public void failed(Throwable e, Void attachment) {
      exception.compareAndSet(null, e);
      countWritesAndTryClose();
    }

    private void countWritesAndTryClose() {
      synchronized (AsynchronousFileOutputStream.this) {
        Preconditions.checkState(outstandingWrites > 0);
        outstandingWrites--;
        closeIfNeeded();
      }
    }
  }
}
