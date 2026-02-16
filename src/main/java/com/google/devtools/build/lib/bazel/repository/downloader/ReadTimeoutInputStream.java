// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import java.io.IOException;
import java.io.InputStream;
import java.net.SocketTimeoutException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import javax.annotation.WillCloseWhenClosed;

/**
 * An {@link InputStream} wrapper that enforces per-read timeouts.
 *
 * <p>{@link java.net.http.HttpClient} does not provide per-read SO_TIMEOUT like {@link
 * java.net.HttpURLConnection}. {@link RetryingInputStream} depends on catching {@link
 * SocketTimeoutException} during reads to reconnect. This class uses a shared {@link
 * ScheduledExecutorService} watchdog: before each read, a task is scheduled that closes the
 * delegate stream after the timeout. If the read completes in time, the timer is cancelled. If the
 * timer fires first, the read throws an {@link IOException} which is caught and rethrown as {@link
 * SocketTimeoutException}.
 */
final class ReadTimeoutInputStream extends InputStream {

  private final InputStream delegate;
  private final ScheduledExecutorService watchdog;
  private final long timeoutMs;

  ReadTimeoutInputStream(
      @WillCloseWhenClosed InputStream delegate,
      ScheduledExecutorService watchdog,
      long timeoutMs) {
    this.delegate = delegate;
    this.watchdog = watchdog;
    this.timeoutMs = timeoutMs;
  }

  @Override
  public int read() throws IOException {
    ScheduledFuture<?> timer = scheduleClose();
    try {
      return delegate.read();
    } catch (IOException e) {
      if (timer.isDone()) {
        throw new SocketTimeoutException("Read timed out after " + timeoutMs + "ms");
      }
      throw e;
    } finally {
      timer.cancel(false);
    }
  }

  @Override
  public int read(byte[] buffer, int offset, int length) throws IOException {
    ScheduledFuture<?> timer = scheduleClose();
    try {
      return delegate.read(buffer, offset, length);
    } catch (IOException e) {
      if (timer.isDone()) {
        throw new SocketTimeoutException("Read timed out after " + timeoutMs + "ms");
      }
      throw e;
    } finally {
      timer.cancel(false);
    }
  }

  @Override
  public int available() throws IOException {
    return delegate.available();
  }

  @Override
  public void close() throws IOException {
    delegate.close();
  }

  private ScheduledFuture<?> scheduleClose() {
    return watchdog.schedule(
        () -> {
          try {
            delegate.close();
          } catch (IOException ignored) {
            // Best effort close to unblock the read.
          }
        },
        timeoutMs,
        TimeUnit.MILLISECONDS);
  }
}
