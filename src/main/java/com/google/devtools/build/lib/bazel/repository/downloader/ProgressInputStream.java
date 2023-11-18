// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Locale;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.WillCloseWhenClosed;

/**
 * Input stream that reports progress on total bytes read as the download progresses.
 *
 * <p>This class is not thread safe, but it is safe to message pass its objects between threads.
 */
@ThreadCompatible
final class ProgressInputStream extends InputStream {

  private static final long PROGRESS_INTERVAL_MS = 200;

  /** Factory for {@link ProgressInputStream}. */
  @ThreadSafe
  static class Factory {
    private final Locale locale;
    private final Clock clock;
    private final ExtendedEventHandler eventHandler;

    Factory(Locale locale, Clock clock, ExtendedEventHandler eventHandler) {
      this.locale = locale;
      this.clock = clock;
      this.eventHandler = eventHandler;
    }

    InputStream create(
        @WillCloseWhenClosed InputStream delegate,
        URL url,
        URL originalUrl,
        OptionalLong totalBytes) {
      return new ProgressInputStream(
          locale,
          clock,
          eventHandler,
          PROGRESS_INTERVAL_MS,
          delegate,
          url,
          originalUrl,
          totalBytes);
    }
  }

  private final Locale locale;
  private final Clock clock;
  private final ExtendedEventHandler eventHandler;
  private final InputStream delegate;
  private final long intervalMs;
  private final URL url;
  private final URL originalUrl;
  private final OptionalLong totalBytes;
  private final AtomicLong toto = new AtomicLong();
  private final AtomicLong nextEvent;

  ProgressInputStream(
      Locale locale,
      Clock clock,
      ExtendedEventHandler eventHandler,
      long intervalMs,
      InputStream delegate,
      URL url,
      URL originalUrl,
      OptionalLong totalBytes) {
    Preconditions.checkArgument(intervalMs >= 0);
    this.locale = locale;
    this.clock = clock;
    this.eventHandler = eventHandler;
    this.intervalMs = intervalMs;
    this.delegate = delegate;
    this.url = url;
    this.originalUrl = originalUrl;
    this.totalBytes = totalBytes;
    this.nextEvent = new AtomicLong(clock.currentTimeMillis() + intervalMs);
    eventHandler.post(new DownloadProgressEvent(originalUrl, url, 0, totalBytes, false));
  }

  @Override
  public int read() throws IOException {
    int result = delegate.read();
    if (result != -1) {
      reportProgress(toto.incrementAndGet());
    }
    return result;
  }

  @Override
  public int read(byte[] buffer, int offset, int length) throws IOException {
    int amount = delegate.read(buffer, offset, length);
    if (amount > 0) {
      reportProgress(toto.addAndGet(amount));
    }
    return amount;
  }

  @Override
  public int available() throws IOException {
    return delegate.available();
  }

  @Override
  public void close() throws IOException {
    delegate.close();
    eventHandler.post(new DownloadProgressEvent(originalUrl, url, toto.get(), totalBytes, true));
  }

  private void reportProgress(long bytesRead) {
    long now = clock.currentTimeMillis();
    if (now < nextEvent.get()) {
      return;
    }
    String via = "";
    if (!url.getHost().equals(originalUrl.getHost())) {
      via = " via " + url.getHost();
    }
    eventHandler.post(new DownloadProgressEvent(originalUrl, url, bytesRead, totalBytes, false));
    eventHandler.handle(
        Event.progress(
            String.format(locale, "Downloading %s%s: %,d bytes", originalUrl, via, bytesRead)));
    nextEvent.set(now + intervalMs);
  }
}
