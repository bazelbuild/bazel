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

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.nullToEmpty;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.math.IntMath;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.Proxy;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import javax.annotation.Nullable;

/** Utility class for connecting to HTTP servers for downloading files. */
final class HttpConnector {

  private static final int MAX_RETRIES = 8;
  private static final int MAX_REDIRECTS = 20;
  private static final int MIN_RETRY_DELAY_MS = 100;
  private static final int CONNECT_TIMEOUT_MS = 1000;
  private static final int MAX_CONNECT_TIMEOUT_MS = 10000;
  private static final int READ_TIMEOUT_MS = 20000;

  /**
   * Connects to HTTP (or file) URL with GET request and lazily returns payload.
   *
   * <p>This routine supports gzip, redirects, retries, and exponential backoff. It's designed to
   * recover fast from transient errors. However please note that this this reliability magic only
   * applies to the connection and header reading phase.
   *
   * @param url URL to download, which can be file, http, or https
   * @param proxy HTTP proxy to use or {@link Proxy#NO_PROXY} if none is desired
   * @param eventHandler Bazel event handler for reporting real-time progress on retries
   * @throws IOException if response returned ≥400 after max retries or ≥300 after max redirects
   * @throws InterruptedException if thread is being cast into oblivion
   */
  static InputStream connect(
      URL url, Proxy proxy, EventHandler eventHandler)
          throws IOException, InterruptedException {
    checkNotNull(proxy);
    checkNotNull(eventHandler);
    if (isProtocol(url, "file")) {
      return url.openConnection().getInputStream();
    }
    if (!isHttp(url)) {
      throw new IOException("Protocol must be http, https, or file");
    }
    List<Throwable> suppressions = new ArrayList<>();
    int retries = 0;
    int redirects = 0;
    int connectTimeout = CONNECT_TIMEOUT_MS;
    while (true) {
      HttpURLConnection connection = null;
      try {
        connection = (HttpURLConnection) url.openConnection(proxy);
        connection.setRequestProperty("Accept-Encoding", "gzip");
        connection.setConnectTimeout(connectTimeout);
        connection.setReadTimeout(READ_TIMEOUT_MS);
        int code;
        try {
          connection.connect();
          code = connection.getResponseCode();
        } catch (FileNotFoundException ignored) {
          code = connection.getResponseCode();
        } catch (IllegalArgumentException e) {
          // This will happen if the user does something like specify a port greater than 2^16-1.
          throw new UnrecoverableHttpException(e.getMessage());
        } catch (IOException e) {
          if (!e.getMessage().startsWith("Server returned")) {
            throw e;
          }
          code = connection.getResponseCode();
        }
        if (code == 200) {
          return getInputStream(connection);
        } else if (code == 301 || code == 302) {
          readAllBytesAndClose(connection.getInputStream());
          if (++redirects == MAX_REDIRECTS) {
            throw new UnrecoverableHttpException("Redirect loop detected");
          }
          url = getLocation(connection);
        } else if (code < 500) {
          readAllBytesAndClose(connection.getErrorStream());
          throw new UnrecoverableHttpException(describeHttpResponse(connection));
        } else {
          throw new IOException(describeHttpResponse(connection));
        }
      } catch (InterruptedIOException e) {
        throw new InterruptedException();
      } catch (UnrecoverableHttpException e) {
        throw e;
      } catch (UnknownHostException e) {
        throw new IOException("Unknown host: " + e.getMessage());
      } catch (IOException e) {
        if (connection != null) {
          connection.disconnect();
        }
        if (e instanceof SocketTimeoutException) {
          connectTimeout = Math.min(connectTimeout * 2, MAX_CONNECT_TIMEOUT_MS);
        }
        if (++retries == MAX_RETRIES) {
          for (Throwable suppressed : suppressions) {
            e.addSuppressed(suppressed);
          }
          throw e;
        }
        suppressions.add(e);
        int timeout = IntMath.pow(2, retries) * MIN_RETRY_DELAY_MS;
        eventHandler.handle(Event.progress(
            String.format("Failed to connect to %s trying again in %,dms: %s",
                url, timeout, e)));
        TimeUnit.MILLISECONDS.sleep(timeout);
      } catch (RuntimeException e) {
        if (connection != null) {
          connection.disconnect();
        }
        throw e;
      }
    }
  }

  private static String describeHttpResponse(HttpURLConnection connection) throws IOException {
    return String.format(
        "%s returned %s %s",
        connection.getRequestMethod(),
        connection.getResponseCode(),
        nullToEmpty(connection.getResponseMessage()));
  }

  private static void readAllBytesAndClose(@Nullable InputStream stream) throws IOException {
    if (stream != null) {
      // TODO: Replace with ByteStreams#exhaust when Guava 20 comes out.
      byte[] buf = new byte[8192];
      while (stream.read(buf) != -1) {}
      stream.close();
    }
  }

  private static InputStream getInputStream(HttpURLConnection connection) throws IOException {
    // See RFC2616 § 3.5 and § 14.11
    switch (firstNonNull(connection.getContentEncoding(), "identity")) {
      case "identity":
        return connection.getInputStream();
      case "gzip":
      case "x-gzip":
        return new GZIPInputStream(connection.getInputStream());
      default:
        throw new UnrecoverableHttpException(
            "Unsupported and unrequested Content-Encoding: " + connection.getContentEncoding());
    }
  }

  @VisibleForTesting
  static URL getLocation(HttpURLConnection connection) throws IOException {
    String newLocation = connection.getHeaderField("Location");
    if (newLocation == null) {
      throw new IOException("Remote redirect missing Location.");
    }
    URL result = mergeUrls(URI.create(newLocation), connection.getURL());
    if (!isHttp(result)) {
      throw new IOException("Bad Location: " + newLocation);
    }
    return result;
  }

  private static URL mergeUrls(URI preferred, URL original) throws IOException {
    // If the Location value provided in a 3xx (Redirection) response does not have a fragment
    // component, a user agent MUST process the redirection as if the value inherits the fragment
    // component of the URI reference used to generate the request target (i.e., the redirection
    // inherits the original reference's fragment, if any). Quoth RFC7231 § 7.1.2
    String protocol = firstNonNull(preferred.getScheme(), original.getProtocol());
    String userInfo = preferred.getUserInfo();
    String host = preferred.getHost();
    int port;
    if (host == null) {
      host = original.getHost();
      port = original.getPort();
      userInfo = original.getUserInfo();
    } else {
      port = preferred.getPort();
      if (userInfo == null
          && host.equals(original.getHost())
          && port == original.getPort()) {
        userInfo = original.getUserInfo();
      }
    }
    String path = preferred.getPath();
    String query = preferred.getQuery();
    String fragment = preferred.getFragment();
    if (fragment == null) {
      fragment = original.getRef();
    }
    URL result;
    try {
      result = new URI(protocol, userInfo, host, port, path, query, fragment).toURL();
    } catch (URISyntaxException | MalformedURLException e) {
      throw new IOException("Could not merge " + preferred + " into " + original);
    }
    return result;
  }

  private static boolean isHttp(URL url) {
    return isProtocol(url, "http") || isProtocol(url, "https");
  }

  private static boolean isProtocol(URL url, String protocol) {
    // An implementation should accept uppercase letters as equivalent to lowercase in scheme names
    // (e.g., allow "HTTP" as well as "http") for the sake of robustness. Quoth RFC3986 § 3.1
    return Ascii.equalsIgnoreCase(protocol, url.getProtocol());
  }

  private static final class UnrecoverableHttpException extends IOException {
    UnrecoverableHttpException(String message) {
      super(message);
    }
  }

  private HttpConnector() {}
}
