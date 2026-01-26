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

import com.google.auth.Credentials;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.FetchId;
import com.google.devtools.build.lib.buildeventstream.FetchEvent;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;

/**
 * HTTP implementation of {@link Downloader}.
 *
 * <p>This class uses {@link HttpConnectorMultiplexer} to connect to HTTP mirrors and then reads the
 * file to disk.
 *
 * <p>This class is (outside of tests) a singleton instance, living in `BazelRepositoryModule`.
 */
public class HttpDownloader implements Downloader {
  private static final Clock CLOCK = new JavaClock();
  private static final Sleeper SLEEPER = new JavaSleeper();
  private static final Locale LOCALE = Locale.getDefault();

  private final Semaphore semaphore;
  private final float timeoutScaling;
  private final int maxAttempts;
  private final Duration maxRetryTimeout;

  public HttpDownloader(
      int maxAttempts, Duration maxRetryTimeout, int maxParallelDownloads, float timeoutScaling) {
    this.maxAttempts = maxAttempts;
    this.maxRetryTimeout = maxRetryTimeout;
    semaphore = new Semaphore(maxParallelDownloads, true);
    this.timeoutScaling = timeoutScaling;
  }

  public HttpDownloader() {
    this(0, Duration.ZERO, 8, 1.0f);
  }

  @Override
  public void download(
      List<URL> urls,
      Map<String, List<String>> headers,
      Credentials credentials,
      Optional<Checksum> checksum,
      String canonicalId,
      OutputStream destination,
      @Nullable String destinationPath,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      Optional<String> type,
      String context)
      throws IOException, InterruptedException {
    HttpConnectorMultiplexer multiplexer = setUpConnectorMultiplexer(eventHandler, clientEnv);

    // Iterate over urls and download the file falling back to the next url if previous failed,
    // while reporting progress to the CLI.
    boolean success = false;

    List<IOException> ioExceptions = ImmutableList.of();

    for (URL url : urls) {
      semaphore.acquire();

      try (HttpStream payload = multiplexer.connect(url, checksum, headers, credentials, type);
          OutputStream out = destination) {
        try {
          ByteStreams.copy(payload, out);
        } catch (SocketTimeoutException e) {
          // SocketTimeoutExceptions are InterruptedIOExceptions; however they do not signify
          // an external interruption, but simply a failed download due to some server timing
          // out. So rethrow them as ordinary IOExceptions.
          throw new IOException(e);
        }
        success = true;
        break;
      } catch (InterruptedIOException e) {
        throw new InterruptedException(e.getMessage());
      } catch (IOException e) {
        if (ioExceptions.isEmpty()) {
          ioExceptions = new ArrayList<>(1);
        }
        ioExceptions.add(e);
        eventHandler.handle(
            Event.warn("Download from " + url + " failed: " + e.getClass() + " " + e.getMessage()));
        continue;
      } finally {
        semaphore.release();
        eventHandler.post(new FetchEvent(url.toString(), FetchId.Downloader.HTTP, success));
      }
    }

    if (!success) {
      // When destinationPath is null (in-memory download), don't add the "Error downloading"
      // prefix - just use the original exception message to preserve backward compatibility.
      var message =
          destinationPath != null
              ? "Error downloading %s to %s%s"
                  .formatted(
                      urls,
                      destinationPath,
                      ioExceptions.isEmpty()
                          ? ""
                          : ": " + Iterables.getLast(ioExceptions).getMessage())
              : ioExceptions.isEmpty()
                  ? "Error downloading " + urls
                  : Iterables.getLast(ioExceptions).getMessage();
      // Make it easy for callers to distinguish between not found and other IO errors.
      var exception =
          !ioExceptions.isEmpty()
                  && ioExceptions.stream().allMatch(e -> e instanceof FileNotFoundException)
              ? new FileNotFoundException(message)
              : new IOException(message);
      for (var cause : ioExceptions) {
        exception.addSuppressed(cause);
      }

      throw exception;
    }
  }

  private HttpConnectorMultiplexer setUpConnectorMultiplexer(
      ExtendedEventHandler eventHandler, Map<String, String> clientEnv) {
    ProxyHelper proxyHelper = new ProxyHelper(clientEnv);
    HttpConnector connector =
        new HttpConnector(
            LOCALE,
            eventHandler,
            proxyHelper,
            SLEEPER,
            timeoutScaling,
            maxAttempts,
            maxRetryTimeout);
    ProgressInputStream.Factory progressInputStreamFactory =
        new ProgressInputStream.Factory(LOCALE, CLOCK, eventHandler);
    HttpStream.Factory httpStreamFactory = new HttpStream.Factory(progressInputStreamFactory);
    return new HttpConnectorMultiplexer(eventHandler, connector, httpStreamFactory);
  }
}
