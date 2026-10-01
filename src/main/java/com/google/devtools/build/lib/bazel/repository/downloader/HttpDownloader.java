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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.auth.Credentials;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.FetchId;
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
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Semaphore;
import java.util.function.Function;

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
      List<URI> urls,
      Map<String, List<String>> headers,
      Credentials credentials,
      Optional<Checksum> checksum,
      String canonicalId,
      Path destination,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      Optional<String> type,
      String context)
      throws IOException, InterruptedException {
    // Stream the payload straight to the destination file (without buffering a potentially large
    // archive in memory) and report progress to the CLI. On total failure, wrap every per-URL
    // failure in a single new IOException naming the URLs and the destination.
    var unused =
        this.<Void>downloadFromUrls(
            urls,
            headers,
            credentials,
            checksum,
            type,
            eventHandler,
            clientEnv,
            /* reportProgressEvents= */ true,
            payload -> {
              try (OutputStream out = destination.getOutputStream()) {
                payload.transferTo(out);
              }
              return null;
            },
            ioExceptions -> aggregatedDownloadException(urls, destination, ioExceptions));
  }

  /**
   * Downloads the contents of the first of the given URLs that succeeds.
   *
   * <p>As with {@link #download}, the URLs are tried in order, falling back to the next one if the
   * previous one fails; callers typically pass the rewritten URLs of a single logical resource
   * (e.g. a mirror/proxy URL followed by the direct URL as a fallback). If all of them fail, the
   * last failure is rethrown with its original type preserved (so callers can still distinguish
   * e.g. {@link java.io.FileNotFoundException}), and the earlier failures are attached to it as
   * suppressed exceptions.
   */
  public byte[] downloadAndRead(
      List<URI> urls,
      Credentials credentials,
      Optional<Checksum> checksum,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv)
      throws IOException, InterruptedException {
    checkArgument(!urls.isEmpty(), "Cannot download from an empty list of URLs");
    // Buffer the (small) registry file in memory and return the bytes. Unlike download(), this
    // posts no progress events and preserves the type of the failing exception for the caller.
    return downloadFromUrls(
        urls,
        ImmutableMap.of(),
        credentials,
        checksum,
        Optional.empty(),
        eventHandler,
        clientEnv,
        /* reportProgressEvents= */ false, // TODO(wyv): Do we need to report any event here?
        payload -> {
          ByteArrayOutputStream out = new ByteArrayOutputStream();
          payload.transferTo(out);
          return out.toByteArray();
        },
        HttpDownloader::lastDownloadException);
  }

  /** Consumes a successfully connected {@link HttpStream}, producing the download's result. */
  @FunctionalInterface
  private interface StreamProcessor<T> {
    T process(HttpStream payload) throws IOException;
  }

  /**
   * Connects to each of {@code urls} in order and hands the first one that succeeds to {@code
   * processor}, returning its result. On failure it falls back to the next URL; if every URL fails,
   * it throws the exception that {@code onAllUrlsFailed} builds from the collected per-URL
   * failures.
   *
   * <p>This is the shared engine behind {@link #download} (which streams the payload to a file) and
   * {@link #downloadAndRead} (which reads it into memory). They differ only in how the payload is
   * consumed ({@code processor}), what is thrown when every URL fails ({@code onAllUrlsFailed}),
   * and whether per-URL progress is reported ({@code reportProgressEvents}).
   */
  private <T> T downloadFromUrls(
      List<URI> urls,
      Map<String, List<String>> headers,
      Credentials credentials,
      Optional<Checksum> checksum,
      Optional<String> type,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      boolean reportProgressEvents,
      StreamProcessor<T> processor,
      Function<List<IOException>, IOException> onAllUrlsFailed)
      throws IOException, InterruptedException {
    HttpConnectorMultiplexer multiplexer = setUpConnectorMultiplexer(eventHandler, clientEnv);

    // Iterate over the URLs, attempting each in turn and falling back to the next one if the
    // previous one failed, until one succeeds.
    List<IOException> ioExceptions = ImmutableList.of();

    for (URI url : urls) {
      semaphore.acquire();
      boolean success = false;

      try (HttpStream payload = multiplexer.connect(url, checksum, headers, credentials, type)) {
        T result = processor.process(payload);
        success = true;
        return result;
      } catch (SocketTimeoutException e) {
        // SocketTimeoutExceptions are InterruptedIOExceptions; however they do not signify an
        // external interruption, but simply a failed download due to some server timing out. So
        // treat them as ordinary IOExceptions and fall back to the next URL.
        ioExceptions =
            recordFailure(
                ioExceptions, new IOException(e), url, eventHandler, reportProgressEvents);
      } catch (InterruptedIOException e) {
        throw new InterruptedException(e.getMessage());
      } catch (IOException e) {
        ioExceptions = recordFailure(ioExceptions, e, url, eventHandler, reportProgressEvents);
      } finally {
        semaphore.release();
        if (reportProgressEvents) {
          eventHandler.post(new FetchEvent(url.toString(), FetchId.Downloader.HTTP, success));
        }
      }
    }

    throw onAllUrlsFailed.apply(ioExceptions);
  }

  /**
   * Records a failed download attempt for {@code url}, lazily allocating the (initially empty and
   * immutable) {@code ioExceptions} list on first use and returning it for the caller to reassign.
   * When {@code reportProgressEvents} is set, the failure is also surfaced to the CLI as a warning.
   */
  private static List<IOException> recordFailure(
      List<IOException> ioExceptions,
      IOException failure,
      URI url,
      ExtendedEventHandler eventHandler,
      boolean reportProgressEvents) {
    if (ioExceptions.isEmpty()) {
      ioExceptions = new ArrayList<>(1);
    }
    ioExceptions.add(failure);
    if (reportProgressEvents) {
      eventHandler.handle(
          Event.warn(
              "Download from "
                  + url
                  + " failed: "
                  + failure.getClass()
                  + " "
                  + failure.getMessage()));
    }
    return ioExceptions;
  }

  /**
   * Builds the exception thrown by {@link #download} when every URL failed: a new {@link
   * IOException} naming the URLs and destination, with each per-URL failure attached as a
   * suppressed exception.
   */
  private static IOException aggregatedDownloadException(
      List<URI> urls, Path destination, List<IOException> ioExceptions) {
    IOException exception =
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
    return exception;
  }

  /**
   * Builds the exception thrown by {@link #downloadAndRead} when every URL failed: the last failure
   * itself, so its original type is preserved (letting callers distinguish e.g. {@link
   * java.io.FileNotFoundException}), with the earlier failures attached as suppressed exceptions.
   * The list is non-empty: {@link #downloadAndRead} rejects an empty URL list, and the loop records
   * a failure for every URL it attempts.
   */
  private static IOException lastDownloadException(List<IOException> ioExceptions) {
    IOException last = Iterables.getLast(ioExceptions);
    for (IOException e : ioExceptions.subList(0, ioExceptions.size() - 1)) {
      last.addSuppressed(e);
    }
    return last;
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
