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

import com.google.common.base.Ascii;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Sleeper;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.Authenticator;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.PasswordAuthentication;
import java.net.ProxySelector;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.UnknownHostException;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.channels.UnresolvedAddressException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import javax.annotation.Nullable;
import javax.annotation.WillClose;
import javax.net.ssl.SSLException;

/**
 * Class for establishing connections to HTTP servers for downloading files.
 *
 * <p>This class must be used in conjunction with {@link HttpConnectorMultiplexer}.
 *
 * <p>Instances are thread safe and can be reused.
 */
@ThreadSafe
class HttpConnector {

  private static final int MAX_ATTEMPTS = 8;
  private static final int MAX_REDIRECTS = 40;
  private static final int MIN_RETRY_DELAY_MS = 100;
  private static final int MIN_CONNECT_TIMEOUT_MS = 1000;
  private static final int MAX_CONNECT_TIMEOUT_MS = 10000;
  private static final int READ_TIMEOUT_MS = 20000;
  private static final ImmutableSet<String> COMPRESSED_EXTENSIONS =
      ImmutableSet.of("bz2", "gz", "jar", "tgz", "war", "xz", "zip");
  private static final String USER_AGENT_VALUE =
      "bazel/" + BlazeVersionInfo.instance().getVersion();

  // Shared executor for HttpClient instances to avoid thread pool proliferation.
  private static final Executor HTTP_CLIENT_EXECUTOR =
      Executors.newCachedThreadPool(
          r -> {
            Thread t = new Thread(r, "http-client-worker");
            t.setDaemon(true);
            return t;
          });

  // Shared watchdog for ReadTimeoutInputStream.
  private static final ScheduledExecutorService READ_TIMEOUT_WATCHDOG =
      Executors.newSingleThreadScheduledExecutor(
          r -> {
            Thread t = new Thread(r, "read-timeout-watchdog");
            t.setDaemon(true);
            return t;
          });

  private final Locale locale;
  private final EventHandler eventHandler;
  private final ProxyHelper proxyHelper;
  private final Sleeper sleeper;
  private final float timeoutScaling;
  private final int maxAttempts;
  private final Duration maxRetryTimeout;

  HttpConnector(
      Locale locale,
      EventHandler eventHandler,
      ProxyHelper proxyHelper,
      Sleeper sleeper,
      float timeoutScaling,
      int maxAttempts,
      Duration maxRetryTimeout) {
    this.locale = locale;
    this.eventHandler = eventHandler;
    this.proxyHelper = proxyHelper;
    this.sleeper = sleeper;
    this.timeoutScaling = timeoutScaling;
    this.maxAttempts = maxAttempts > 0 ? maxAttempts : MAX_ATTEMPTS;
    this.maxRetryTimeout = maxRetryTimeout;
  }

  HttpConnector(
      Locale locale,
      EventHandler eventHandler,
      ProxyHelper proxyHelper,
      Sleeper sleeper,
      float timeoutScaling) {
    this(locale, eventHandler, proxyHelper, sleeper, timeoutScaling, 0, Duration.ZERO);
  }

  HttpConnector(
      Locale locale, EventHandler eventHandler, ProxyHelper proxyHelper, Sleeper sleeper) {
    this(locale, eventHandler, proxyHelper, sleeper, 1.0f);
  }

  private int scale(int unscaled) {
    return Math.round(unscaled * timeoutScaling);
  }

  DownloadResponse connect(
      URI originalUrl, Function<URI, ImmutableMap<String, List<String>>> requestHeaders)
      throws IOException {

    if (Thread.interrupted()) {
      throw new InterruptedIOException();
    }
    URI url = originalUrl;
    if (HttpUtils.isProtocol(url, "file")) {
      InputStream body = Files.newInputStream(Path.of(url));
      return new DownloadResponse(url, Collections.emptyMap(), body);
    }
    List<Throwable> suppressions = new ArrayList<>();
    int retries = 0;
    int redirects = 0;
    int connectTimeout = scale(MIN_CONNECT_TIMEOUT_MS);
    while (true) {
      InputStream responseBody = null;
      try {
        ProxyInfo proxyInfo = proxyHelper.createProxyIfNeeded(url);

        // Build HttpClient per retry iteration (needed for connect timeout scaling).
        HttpClient.Builder clientBuilder =
            HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(connectTimeout))
                .followRedirects(HttpClient.Redirect.NEVER)
                .version(HttpClient.Version.HTTP_2)
                .executor(HTTP_CLIENT_EXECUTOR);

        if (proxyInfo.proxy().type() == java.net.Proxy.Type.HTTP) {
          InetSocketAddress proxyAddress =
              (InetSocketAddress) proxyInfo.proxy().address();
          clientBuilder.proxy(
              ProxySelector.of(proxyAddress));
          if (proxyInfo.hasCredentials()) {
            String proxyAuthHeader = proxyInfo.getProxyAuthorizationHeader();
            clientBuilder.authenticator(
                new Authenticator() {
                  @Override
                  protected PasswordAuthentication getPasswordAuthentication() {
                    if (getRequestorType() == RequestorType.PROXY) {
                      // Extract username/password from ProxyInfo's auth header.
                      // The ProxyHelper already decodes credentials, but Authenticator
                      // needs them split. We parse from the Base64 header.
                      return decodeBasicAuth(proxyAuthHeader);
                    }
                    return null;
                  }
                });
          }
        }

        HttpClient client = clientBuilder.build();

        boolean isAlreadyCompressed =
            COMPRESSED_EXTENSIONS.contains(HttpUtils.getExtension(url.getPath()))
                || COMPRESSED_EXTENSIONS.contains(HttpUtils.getExtension(originalUrl.getPath()));

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
            .uri(url)
            .GET()
            .timeout(Duration.ofMillis(scale(READ_TIMEOUT_MS)));

        boolean hasUserAgent = false;
        for (Map.Entry<String, List<String>> entry : requestHeaders.apply(url).entrySet()) {
          if (isAlreadyCompressed && Ascii.equalsIgnoreCase(entry.getKey(), "Accept-Encoding")) {
            // We're not going to ask for compression if we're downloading a file that already
            // appears to be compressed.
            continue;
          }
          String key = entry.getKey();
          if (Ascii.equalsIgnoreCase(key, "User-Agent")) {
            hasUserAgent = true;
          }
          for (String value : entry.getValue()) {
            requestBuilder.header(key, value);
          }
        }
        if (!hasUserAgent) {
          requestBuilder.header("User-Agent", USER_AGENT_VALUE);
        }
        // For HTTP connections through authenticated proxies, set the Proxy-Authorization header.
        if (proxyInfo.hasCredentials()) {
          requestBuilder.header("Proxy-Authorization", proxyInfo.getProxyAuthorizationHeader());
        }

        HttpRequest request = requestBuilder.build();

        HttpResponse<InputStream> response;
        try {
          response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());
        } catch (HttpConnectTimeoutException e) {
          // HttpConnectTimeoutException is thrown by HttpClient on connect timeout.
          // Convert to SocketTimeoutException so the retry logic handles it.
          SocketTimeoutException wrapper = new SocketTimeoutException(e.getMessage());
          wrapper.initCause(e);
          throw wrapper;
        } catch (ConnectException e) {
          if (hasCause(e, UnresolvedAddressException.class)) {
            // HttpClient throws ConnectException wrapping UnresolvedAddressException
            // for unresolvable hosts, instead of UnknownHostException.
            throw new UnknownHostException(url.getHost());
          }
          throw e;
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new InterruptedIOException();
        }

        int code = response.statusCode();
        responseBody = response.body();

        // 206 means partial content and only happens if caller specified Range. See RFC7233 § 4.1.
        if (code == 200 || code == 206) {
          InputStream wrappedBody =
              new ReadTimeoutInputStream(responseBody, READ_TIMEOUT_WATCHDOG, scale(READ_TIMEOUT_MS));
          DownloadResponse result =
              new DownloadResponse(url, response.headers().map(), wrappedBody);
          responseBody = null; // ownership transferred
          return result;
        } else if (code == 301 || code == 302 || code == 303 || code == 307) {
          readAllBytesAndClose(responseBody);
          responseBody = null;
          if (++redirects == MAX_REDIRECTS) {
            eventHandler.handle(Event.progress("Redirect loop detected in " + originalUrl));
            throw new UnrecoverableHttpException("Redirect loop detected");
          }
          url = HttpUtils.getLocation(url,
              response.headers().firstValue("Location").orElse(null));
          if (code == 301) {
            originalUrl = url;
          }
        } else if (code == 403) {
          // jart@ has noticed BitBucket + Amazon AWS downloads frequently flake with this code.
          throw new IOException(describeHttpResponse(code));
        } else if (code == 408) {
          throw new IOException(describeHttpResponse(code));
        } else if (code == 429) {
          throw new IOException(describeHttpResponse(code));
        } else if (code < 500          // 4xx means client seems to have erred quoth RFC7231 § 6.5
                    || code == 501     // Server doesn't support function quoth RFC7231 § 6.6.2
                    || code == 505) {  // Server refuses to support version quoth RFC7231 § 6.6.6
          // This is a permanent error so we're not going to retry.
          readAllBytesAndClose(responseBody);
          responseBody = null;
          if (code == 404 || code == 410) {
            throw new FileNotFoundException(describeHttpResponse(code));
          }
          throw new UnrecoverableHttpException(describeHttpResponse(code));
        } else {
          // However we will retry on some 5xx errors, particularly 500, 502 and 503.
          throw new IOException(describeHttpResponse(code));
        }
      } catch (SSLException e) {
        if (e.getMessage() != null
            && (e.getMessage().contains("certificate")
                || e.getMessage().contains("CertPathValidatorException"))) {
          String message = "TLS error: " + e.getMessage();
          eventHandler.handle(Event.progress(message));
          IOException httpException = new UnrecoverableHttpException(message);
          httpException.addSuppressed(e);
          throw httpException;
        }
        throw e;
      } catch (UnknownHostException e) {
        String message = "Unknown host: " + e.getMessage();
        eventHandler.handle(Event.progress(message));
        IOException httpException = new UnrecoverableHttpException(message);
        httpException.addSuppressed(e);
        throw httpException;
      } catch (UnrecoverableHttpException | FileNotFoundException e) {
        throw e;
      } catch (IllegalArgumentException e) {
        throw new UnrecoverableHttpException(e.getMessage());
      } catch (IOException e) {
        if (responseBody != null) {
          try {
            responseBody.close();
          } catch (IOException ignored) {
            // Best effort cleanup.
          }
        }
        // We don't respect the Retry-After header (RFC7231 § 7.1.3) because it's rarely used and
        // tends to be too conservative when it is. We're already being good citizens by using
        // exponential backoff with jitter. Furthermore RFC law didn't use the magic word "MUST".
        double rawTimeout = Math.scalb((double) MIN_RETRY_DELAY_MS, retries);
        if (!maxRetryTimeout.isZero()) {
          rawTimeout = Math.min(rawTimeout, (double) maxRetryTimeout.toMillis());
        }
        int timeout = (int) ((0.75 + Math.random() / 2) * rawTimeout);
        if (e instanceof SocketTimeoutException) {
          eventHandler.handle(Event.progress("Timeout connecting to " + url));
          connectTimeout = Math.min(connectTimeout * 2, scale(MAX_CONNECT_TIMEOUT_MS));
          // If we got connect timeout, we're already doing exponential backoff, so no point
          // in sleeping too.
          timeout = 1;
        } else if (e instanceof InterruptedIOException) {
          // Please note that SocketTimeoutException is a subtype of InterruptedIOException.
          throw e;
        }
        if (++retries == maxAttempts) {
          if (e instanceof SocketTimeoutException) {
            // SocketTimeoutExceptions are InterruptedIOExceptions; however they do not signify
            // an external interruption, but simply a failed download due to some server timing
            // out. So rethrow them as ordinary IOExceptions.
            e = new IOException(e.getMessage(), e);
          } else {
            eventHandler
                .handle(Event.progress(format("Error connecting to %s: %s", url, e.getMessage())));
          }
          for (Throwable suppressed : suppressions) {
            e.addSuppressed(suppressed);
          }
          throw e;
        }
        // Java 7 allows us to create a tree of all errors that led to the ultimate failure.
        suppressions.add(e);
        eventHandler.handle(
            Event.progress(format("Failed to connect to %s trying again in %,dms", url, timeout)));
        url = originalUrl;
        try {
          sleeper.sleepMillis(timeout);
        } catch (InterruptedException translated) {
          throw new InterruptedIOException();
        }
      } catch (RuntimeException e) {
        if (responseBody != null) {
          try {
            responseBody.close();
          } catch (IOException ignored) {
            // Best effort cleanup.
          }
        }
        eventHandler.handle(Event.progress(format("Unknown error connecting to %s: %s", url, e)));
        throw e;
      }
    }
  }

  private String describeHttpResponse(int statusCode) {
    return format("GET returned %d", statusCode);
  }

  private String format(String format, Object... args) {
    return String.format(locale, format, args);
  }

  // Exhausts all bytes in an HTTP response to make it easier for Java infrastructure to reuse
  // sockets.
  private static void readAllBytesAndClose(
      @WillClose @Nullable InputStream stream)
          throws IOException {
    if (stream != null) {
      ByteStreams.exhaust(stream);
      stream.close();
    }
  }

  private static boolean hasCause(Throwable t, Class<? extends Throwable> causeType) {
    for (Throwable cause = t.getCause(); cause != null; cause = cause.getCause()) {
      if (causeType.isInstance(cause)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Decodes a Basic authentication header value into username/password.
   * Expected format: "Basic base64encoded"
   */
  @Nullable
  private static PasswordAuthentication decodeBasicAuth(String authHeader) {
    if (authHeader == null || !authHeader.startsWith("Basic ")) {
      return null;
    }
    String decoded =
        new String(
            java.util.Base64.getDecoder().decode(authHeader.substring(6)),
            java.nio.charset.StandardCharsets.UTF_8);
    int colon = decoded.indexOf(':');
    if (colon < 0) {
      return null;
    }
    return new PasswordAuthentication(
        decoded.substring(0, colon), decoded.substring(colon + 1).toCharArray());
  }
}
