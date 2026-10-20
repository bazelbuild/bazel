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

package com.google.devtools.build.remote.worker;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.protobuf.ByteString;
import java.io.OutputStream;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * The stdout and stderr of executing actions, readable while the actions run through ByteStream
 * reads of the stream names advertised in {@code ExecuteOperationMetadata}.
 */
final class LiveOutputStreams {
  /**
   * How long a stream stays readable after its action finished, since a fast action may finish
   * before the client's read arrives.
   */
  private static final long RETENTION_AFTER_CLOSE_SECONDS = 60;

  private final ConcurrentHashMap<String, LiveOutput> streams = new ConcurrentHashMap<>();
  private final ScheduledExecutorService cleanup =
      Executors.newSingleThreadScheduledExecutor(
          new ThreadFactoryBuilder()
              .setNameFormat("live-output-cleanup-%d")
              .setDaemon(true)
              .build());

  /**
   * Creates the stdout and stderr streams of the operation with the given name. They must be closed
   * once the action finishes.
   */
  ActionOutputs create(String operationName) {
    return new ActionOutputs(
        createStream(operationName + "/streams/stdout"),
        createStream(operationName + "/streams/stderr"));
  }

  private LiveOutput createStream(String resourceName) {
    LiveOutput output = new LiveOutput(resourceName);
    streams.put(resourceName, output);
    return output;
  }

  /** Returns the stream with the given resource name, or null if there is none (anymore). */
  @Nullable
  LiveOutput get(String resourceName) {
    return streams.get(resourceName);
  }

  /** The live stdout and stderr of one action. */
  record ActionOutputs(LiveOutput stdout, LiveOutput stderr) {
    void close() {
      stdout.close();
      stderr.close();
    }
  }

  /** An append-only buffer of an action's output that readers can follow until it is closed. */
  final class LiveOutput extends OutputStream {
    private final String resourceName;
    private ByteString data = ByteString.EMPTY;
    private boolean closed;

    private LiveOutput(String resourceName) {
      this.resourceName = resourceName;
    }

    String resourceName() {
      return resourceName;
    }

    @Override
    public void write(int b) {
      write(new byte[] {(byte) b}, 0, 1);
    }

    @Override
    public synchronized void write(byte[] b, int off, int len) {
      if (closed) {
        return;
      }
      data = data.concat(ByteString.copyFrom(b, off, len));
      notifyAll();
    }

    @Override
    public void close() {
      synchronized (this) {
        if (closed) {
          return;
        }
        closed = true;
        notifyAll();
      }
      var unused =
          cleanup.schedule(
              () -> streams.remove(resourceName, this),
              RETENTION_AFTER_CLOSE_SECONDS,
              TimeUnit.SECONDS);
    }

    /**
     * Returns up to {@code maxBytes} of output starting at {@code offset}, waiting up to {@code
     * timeoutMillis} for output to become available.
     *
     * @return the output, which is empty if none arrived in time, or null if {@code offset} is the
     *     end of the stream and the stream is closed
     * @throws IllegalArgumentException if {@code offset} is past the end of a closed stream
     */
    @Nullable
    synchronized ByteString read(long offset, int maxBytes, long timeoutMillis)
        throws InterruptedException {
      if (offset >= data.size() && !closed) {
        wait(timeoutMillis);
      }
      if (offset > data.size() && closed) {
        throw new IllegalArgumentException(
            "read_offset %d is past the end of the stream (%d bytes)"
                .formatted(offset, data.size()));
      }
      if (offset >= data.size()) {
        return closed ? null : ByteString.EMPTY;
      }
      int start = (int) offset;
      return data.substring(start, Math.min(data.size(), start + maxBytes));
    }
  }
}
