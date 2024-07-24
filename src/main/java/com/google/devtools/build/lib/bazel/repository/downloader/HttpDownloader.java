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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.buildeventstream.FetchEvent;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import com.google.devtools.build.lib.vfs.Path;
import java.io.ByteArrayOutputStream;
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

/**
 * HTTP implementation of {@link Downloader}.
 *
 * <p>This class uses {@link HttpConnectorMultiplexer} to connect to HTTP mirrors and then reads the
 * file to disk.
 *
 * <p>This class is (outside of tests) a singleton instance, living in `BazelRepositoryModule`.
 */
public class HttpDownloader implements Downloader {
  public static final int DEFAULT_MAX_PARALLEL_DOWNLOADS = 8;
  private static final Clock CLOCK = new JavaClock();
  private static final Sleeper SLEEPER = new JavaSleeper();
  private static final Locale LOCALE = Locale.getDefault();

  private final Semaphore semaphore;
  private float timeoutScaling = 1.0f;
  private int maxAttempts = 0;
  private int maxParallelDownloads = DEFAULT_MAX_PARALLEL_DOWNLOADS;
  private Duration maxRetryTimeout = Duration.ZERO;

  public HttpDownloader() {
    semaphore = new Semaphore(maxParallelDownloads, true);
  }

  public void setTimeoutScaling(float timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  public void setMaxAttempts(int maxAttempts) {
    this.maxAttempts = maxAttempts;
  }

  public void setMaxRetryTimeout(Duration maxRetryTimeout) {
    this.maxRetryTimeout = maxRetryTimeout;
  }

  public void setMaxParallelDownloads(int maxParallelDownloads) {
    if (maxParallelDownloads >= this.maxParallelDownloads) {
      // increase the number of possible parallel downloads
      semaphore.release(maxParallelDownloads - this.maxParallelDownloads);
    } else {
      // reduce the number of possible parallel downloads
      semaphore.acquireUninterruptibly(this.maxParallelDownloads - maxParallelDownloads);
    }
    this.maxParallelDownloads = maxParallelDownloads;
  }

  @Override
  public void download(
      List<URL> urls,
      Map<String, List<String>> headers,
      Credentials credentials,
      Optional<Checksum> checksum,
      String canonicalId,
      Path destination,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      Optional<String> type)
      throws IOException, InterruptedException {
    HttpConnectorMultiplexer multiplexer = setUpConnectorMultiplexer(eventHandler, clientEnv);

    // Iterate over urls and download the file falling back to the next url if previous failed,
    // while reporting progress to the CLI.
    boolean success = false;

    List<IOException> ioExceptions = ImmutableList.of();

    for (URL url : urls) {
      semaphore.acquire();

      try (HttpStream payload = multiplexer.connect(url, checksum, headers, credentials, type);
          OutputStream out = destination.getOutputStream()) {
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
        eventHandler.post(new FetchEvent(url.toString(), success));
      }
    }

    if (!success) {
      final IOException exception =
          new IOException(
              "Error downloading "
                  + urls
                  + " to "
                  + destination
                  + (ioExceptions.isEmpty()
                      ? ""
                      : ": " + Iterables.getLast(ioExceptions).getMessage()));

      for (IOException cause : ioExceptions) {
        exception.addSuppressed(cause);
      }

      throw exception;
    }
  }

  /** Downloads the contents of one URL and reads it into a byte array. */
  public byte[] downloadAndReadOneUrl(
      URL url,
      Credentials credentials,
      Optional<Checksum> checksum,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv)
      throws IOException, InterruptedException {
    HttpConnectorMultiplexer multiplexer = setUpConnectorMultiplexer(eventHandler, clientEnv);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    semaphore.acquire();
    try (HttpStream payload =
        multiplexer.connect(url, checksum, ImmutableMap.of(), credentials, Optional.empty())) {
      ByteStreams.copy(payload, out);
    } catch (SocketTimeoutException e) {
      // SocketTimeoutExceptions are InterruptedIOExceptions; however they do not signify
      // an external interruption, but simply a failed download due to some server timing
      // out. So rethrow them as ordinary IOExceptions.
      throw new IOException(e);
    } catch (InterruptedIOException e) {
      throw new InterruptedException(e.getMessage());
    } finally {
      semaphore.release();
      // TODO(wyv): Do we need to report any event here?
    }
    return out.toByteArray();
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
