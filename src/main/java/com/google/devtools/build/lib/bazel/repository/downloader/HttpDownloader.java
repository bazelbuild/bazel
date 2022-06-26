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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
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
import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Semaphore;

/**
 * HTTP implementation of {@link Downloader}.
 *
 * <p>This class uses {@link HttpConnectorMultiplexer} to connect to HTTP mirrors and then reads the
 * file to disk.
 */
public class HttpDownloader implements Downloader {
  private static final int MAX_PARALLEL_DOWNLOADS = 8;
  private static final Semaphore SEMAPHORE = new Semaphore(MAX_PARALLEL_DOWNLOADS, true);
  private static final Clock CLOCK = new JavaClock();
  private static final Sleeper SLEEPER = new JavaSleeper();
  private static final Locale LOCALE = Locale.getDefault();

  private float timeoutScaling = 1.0f;

  public HttpDownloader() {}

  public void setTimeoutScaling(float timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  @Override
  public void download(
      List<URL> urls,
      Map<URI, Map<String, String>> authHeaders,
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
      SEMAPHORE.acquire();

      try (HttpStream payload = multiplexer.connect(url, checksum, authHeaders, type);
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
        SEMAPHORE.release();
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
      URL url, ExtendedEventHandler eventHandler, Map<String, String> clientEnv)
      throws IOException, InterruptedException {
    HttpConnectorMultiplexer multiplexer = setUpConnectorMultiplexer(eventHandler, clientEnv);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    SEMAPHORE.acquire();
    try (HttpStream payload = multiplexer.connect(url, Optional.absent())) {
      ByteStreams.copy(payload, out);
    } catch (SocketTimeoutException e) {
      // SocketTimeoutExceptions are InterruptedIOExceptions; however they do not signify
      // an external interruption, but simply a failed download due to some server timing
      // out. So rethrow them as ordinary IOExceptions.
      throw new IOException(e);
    } catch (InterruptedIOException e) {
      throw new InterruptedException(e.getMessage());
    } finally {
      SEMAPHORE.release();
      // TODO(wyv): Do we need to report any event here?
    }
    return out.toByteArray();
  }

  private HttpConnectorMultiplexer setUpConnectorMultiplexer(
      ExtendedEventHandler eventHandler, Map<String, String> clientEnv) {
    ProxyHelper proxyHelper = new ProxyHelper(clientEnv);
    HttpConnector connector =
        new HttpConnector(LOCALE, eventHandler, proxyHelper, SLEEPER, timeoutScaling);
    ProgressInputStream.Factory progressInputStreamFactory =
        new ProgressInputStream.Factory(LOCALE, CLOCK, eventHandler);
    HttpStream.Factory httpStreamFactory = new HttpStream.Factory(progressInputStreamFactory);
    return new HttpConnectorMultiplexer(eventHandler, connector, httpStreamFactory);
  }
}
