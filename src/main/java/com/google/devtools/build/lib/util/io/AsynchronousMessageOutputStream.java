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

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.Message;
import com.google.protobuf.MessageLite;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicReference;

/**
 * An output stream supporting asynchronous writes of length-delimited protocol buffer messages,
 * backed by a file.
 */
@ThreadSafety.ThreadSafe
public class AsynchronousMessageOutputStream<T extends Message> implements MessageOutputStream<T> {
  private static final byte[] POISON_PILL = new byte[1];

  private final Thread writerThread;
  // Maybe we should use an ArrayBlockingQueue instead, and accept that write may block if the
  // buffer is full?
  private final BlockingQueue<byte[]> queue = new LinkedBlockingDeque<>();
  // The future returned by closeAsync().
  private final SettableFuture<Void> closeFuture = SettableFuture.create();
  // To store any exception raised from the writes.
  private final AtomicReference<Throwable> exception = new AtomicReference<>();

  public AsynchronousMessageOutputStream(Path path) throws IOException {
    this(
        path.toString(),
        new BufferedOutputStream( // Use a buffer of 100 kByte, scientifically chosen at random.
            path.getOutputStream(), 100000));
  }

  public AsynchronousMessageOutputStream(String name, OutputStream out) {
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

  /**
   * Writes a protocol buffer message in the same format as {@link
   * MessageLite#writeDelimitedTo(java.io.OutputStream)}.
   *
   * <p>The writes are guaranteed to land in the output file in the same order that they were
   * called; However, some writes may fail, leaving the file partially corrupted. In case a write
   * fails, an exception will be propagated in close, but remaining writes will be allowed to
   * continue.
   */
  @Override
  public void write(T m) {
    Preconditions.checkNotNull(m);

    if (closeFuture.isDone()) {
      if (exception.get() != null) {
        // There was a previous write failure. Silently return without doing anything.
        return;
      } else {
        // Attempted to write after closing.
        throw new IllegalStateException();
      }
    }

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

    Uninterruptibles.putUninterruptibly(queue, bos.toByteArray());
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
   * Returns a future that will close the stream when all pending writes are completed.
   *
   * Any failed writes will propagate an exception.
   */
  public ListenableFuture<Void> closeAsync() {
    Uninterruptibles.putUninterruptibly(queue, POISON_PILL);
    return closeFuture;
  }
}
