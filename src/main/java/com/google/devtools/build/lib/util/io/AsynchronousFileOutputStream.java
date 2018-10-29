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
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Message;
import com.google.protobuf.MessageLite;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicReference;

/** An output stream supporting asynchronous writes, backed by a file. */
@ThreadSafety.ThreadSafe
public class AsynchronousFileOutputStream extends OutputStream implements MessageOutputStream {
  private static final byte[] POISON_PILL = new byte[1];

  private final Thread writerThread;
  // Maybe we should use an ArrayBlockingQueue instead, and accept that write may block if the
  // buffer is full?
  private final BlockingQueue<byte[]> queue = new LinkedBlockingDeque<>();
  // The future returned by closeAsync().
  private final SettableFuture<Void> closeFuture = SettableFuture.create();
  // To store any exception raised from the writes.
  private final AtomicReference<Throwable> exception = new AtomicReference<>();

  public AsynchronousFileOutputStream(String filename) throws IOException {
    this(
        filename,
        new BufferedOutputStream( // Use a buffer of 100 kByte, scientifically chosen at random.
            Files.newOutputStream(Paths.get(filename)), 100000));
  }

  @VisibleForTesting
  AsynchronousFileOutputStream(String name, OutputStream out) {
    writerThread =
        new Thread(
            () -> {
              try {
                byte[] data;
                while ((data = queue.take()) != POISON_PILL) {
                  out.write(data);
                }
              } catch (InterruptedException e) {
                // Exit quietly.
              } catch (Exception e) {
                exception.set(e);
                closeFuture.setException(e);
              } finally {
                try {
                  out.close();
                  closeFuture.set(null);
                } catch (Exception e) {
                  closeFuture.setException(e);
                }
              }
            },
            "async-file-writer:" + name);
    writerThread.start();
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
        new ByteArrayOutputStream(CodedOutputStream.computeUInt32SizeNoTag(size) + size);
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
  public void write(byte[] data) {
    Preconditions.checkNotNull(data);
    if (closeFuture.isDone()) {
      if (exception.get() != null) {
        // There was a write failure. Silently return without doing anything.
        return;
      } else {
        // The file was closed.
        throw new IllegalStateException();
      }
    }
    Uninterruptibles.putUninterruptibly(queue, data);
  }

  /** Returns whether the stream is open for writing. */
  public boolean isOpen() {
    return !closeFuture.isDone();
  }

  /**
   * Closes the stream without waiting until pending writes are committed, and supressing errors.
   *
   * <p>Pending writes will still continue asynchronously, but any errors will be ignored.
   */
  public void closeNow() {
    writerThread.interrupt();
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
    } catch (ExecutionException e) {
      Throwables.throwIfInstanceOf(e.getCause(), IOException.class);
      Throwables.throwIfInstanceOf(e.getCause(), RuntimeException.class);
      throw new RuntimeException(e.getCause());
    }
  }

  /**
   * Flushes the currently ongoing writes into the channel.
   *
   * Throws an exception if any of the writes or the close itself have failed.
   */
  @Override
  public void flush() throws IOException {
  }

  /**
   * Returns a future that will close the stream when all pending writes are completed.
   *
   * Any failed writes will propagate an exception.
   */
  public ListenableFuture<Void> closeAsync() {
    Uninterruptibles.putUninterruptibly(queue, POISON_PILL);
    return closeFuture;
  }
}
