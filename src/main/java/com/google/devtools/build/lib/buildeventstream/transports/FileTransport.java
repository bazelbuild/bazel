// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream.transports;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.CompletionHandler;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Non-blocking file transport.
 *
 * <p>Implementors of this class need to implement {@link #sendBuildEvent(BuildEvent)} which
 * serializes the build event and writes it to file using {@link #writeData(byte[])}.
 */
abstract class FileTransport implements BuildEventTransport {

  /**
   * We use an {@link AsynchronousFileChannel} to perform non-blocking writes to a file. It get's
   * tricky when it comes to {@link #close()}, as we may only complete the returned future when
   * all writes have completed (succeeded or failed). Thus, we use a field
   * {@link #outstandingWrites} to keep track of the number of writes that have not completed yet.
   * It's simply incremented before a new write and decremented after a write has completed. When
   * it's {@code 0} it's safe to complete the close future.
   */

  private static final Logger log = Logger.getLogger(FileTransport.class.getName());

  @VisibleForTesting
  final AsynchronousFileChannel ch;
  private final WriteCompletionHandler completionHandler = new WriteCompletionHandler();
  protected final BuildEventConverters converters;
  // The offset in the file to begin the next write at.
  private long writeOffset;
  // Number of writes that haven't completed yet.
  private long outstandingWrites;
  // The future returned by close()
  private SettableFuture<Void> closeFuture;

  FileTransport(String path, final PathConverter pathConverter) {
    try {
      ch = AsynchronousFileChannel.open(Paths.get(path), StandardOpenOption.CREATE,
          StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    this.converters = new BuildEventConverters() {
      @Override
      public PathConverter pathConverter() {
        return pathConverter;
      }
    };
  }

  synchronized void writeData(byte[] data) {
    checkNotNull(data);
    if (!ch.isOpen()) {
      close();
      return;
    }
    if (closing()) {
      return;
    }

    outstandingWrites++;

    ch.write(ByteBuffer.wrap(data), writeOffset, null, completionHandler);

    writeOffset += data.length;
  }

  @Override
  public synchronized Future<Void> close() {
    if (closing()) {
      return closeFuture;
    }
    closeFuture = SettableFuture.create();

    if (writesComplete()) {
      doClose();
    }

    return closeFuture;
  }

  private void doClose() {
    try {
      ch.force(true);
      ch.close();
    } catch (IOException e) {
      log.log(Level.SEVERE, e.getMessage(), e);
    } finally {
      closeFuture.set(null);
    }
  }

  private boolean closing() {
    return closeFuture != null;
  }

  private boolean writesComplete() {
    return outstandingWrites == 0;
  }

  /**
   * Handler that's notified when a write completes.
   */
  private final class WriteCompletionHandler implements CompletionHandler<Integer, Void> {

    @Override
    public void completed(Integer result, Void attachment) {
      countWriteAndTryClose();
    }

    @Override
    public void failed(Throwable exc, Void attachment) {
      log.log(Level.SEVERE, exc.getMessage(), exc);
      countWriteAndTryClose();
      // There is no point in trying to continue. Close the transport.
      close();
    }

    private void countWriteAndTryClose() {
      synchronized (FileTransport.this) {
        checkState(outstandingWrites > 0);

        outstandingWrites--;

        if (closing() && writesComplete()) {
          doClose();
        }
      }
    }
  }
}
