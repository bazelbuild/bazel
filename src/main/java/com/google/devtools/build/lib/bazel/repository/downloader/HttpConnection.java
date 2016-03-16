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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.io.ByteStreams;
import com.google.common.net.MediaType;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.Proxy;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

/**
 * Represents a connection over HTTP.
 */
class HttpConnection implements Closeable {
  private static final int MAX_REDIRECTS = 20;
  private final InputStream inputStream;
  private final int contentLength;

  private HttpConnection(InputStream inputStream, int contentLength) {
    this.inputStream = inputStream;
    this.contentLength = contentLength;
  }

  public InputStream getInputStream() {
    return inputStream;
  }

  /**
   * @return The length of the response, or -1 if unknown.
   */
  public int getContentLength() {
    return contentLength;
  }

  @Override
  public void close() throws IOException {
    inputStream.close();
  }

  private static int parseContentLength(HttpURLConnection connection) {
    String length;
    try {
      length = connection.getHeaderField("Content-Length");
      if (length == null) {
        return -1;
      }
      return Integer.parseInt(length);
    } catch (NumberFormatException e) {
      return -1;
    }
  }

  public static HttpConnection createAndConnect(URL url) throws IOException {
    int retries = MAX_REDIRECTS;
    Proxy proxy = ProxyHelper.createProxyIfNeeded(url.toString());
    do {
      HttpURLConnection connection = (HttpURLConnection) url.openConnection(proxy);
      try {
        connection.connect();
      } catch (IllegalArgumentException e) {
        throw new IOException("Failed to connect to " + url + " : " + e.getMessage(), e);
      }

      int statusCode = connection.getResponseCode();
      switch (statusCode) {
        case HttpURLConnection.HTTP_OK:
          return new HttpConnection(connection.getInputStream(), parseContentLength(connection));
        case HttpURLConnection.HTTP_MOVED_PERM:
        case HttpURLConnection.HTTP_MOVED_TEMP:
          url = tryGetLocation(statusCode, connection);
          connection.disconnect();
          break;
        case -1:
          throw new IOException("An HTTP error occured");
        default:
          throw new IOException(String.format("%s %s: %s",
              connection.getResponseCode(),
              connection.getResponseMessage(),
              readBody(connection)));
      }
    } while (retries-- > 0);
    throw new IOException("Maximum redirects (" + MAX_REDIRECTS + ") exceeded");
  }

  private static URL tryGetLocation(int statusCode, HttpURLConnection connection)
      throws IOException {
    String newLocation = connection.getHeaderField("Location");
    if (newLocation == null) {
      throw new IOException(
          "Remote returned " + statusCode + " but did not return location header.");
    }

    URL newUrl;
    try {
      newUrl = new URL(newLocation);
    } catch (MalformedURLException e) {
      throw new IOException("Remote returned invalid location header: " + newLocation);
    }

    String newProtocol = newUrl.getProtocol();
    if (!("http".equals(newProtocol) || "https".equals(newProtocol))) {
      throw new IOException(
          "Remote returned invalid location header: " + newLocation);
    }

    return newUrl;
  }

  /**
   * Attempts to detect the encoding the HTTP reponse is using.
   *
   * <p>This attempts to read the Content-Encoding header, then the Content-Type header,
   * then just falls back to UTF-8.</p>
   *
   * @throws IOException If something goes wrong (the encoding isn't parsable or is, but isn't
   * supported by the system).
   */
  @VisibleForTesting
  static Charset getEncoding(HttpURLConnection connection) throws IOException {
    String encoding = connection.getContentEncoding();
    if (encoding != null) {
      if (Charset.availableCharsets().containsKey(encoding)) {
        try {
          return Charset.forName(encoding);
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
          throw new IOException(
              "Got invalid encoding from " + connection.getURL() + ": " + encoding);
        }
      } else {
        throw new IOException(
            "Got unavailable encoding from " + connection.getURL() + ": " + encoding);
      }
    }
    encoding = connection.getContentType();
    if (encoding == null) {
      return StandardCharsets.UTF_8;
    }
    try {
      MediaType mediaType = MediaType.parse(encoding);
      if (mediaType == null) {
        return StandardCharsets.UTF_8;
      }
      Optional<Charset> charset = mediaType.charset();
      if (charset.isPresent()) {
        return charset.get();
      }
    } catch (IllegalArgumentException | IllegalStateException e) {
      throw new IOException(
          "Got invalid encoding from " + connection.getURL() + ": " + encoding);
    }
    return StandardCharsets.UTF_8;
  }

  private static String readBody(HttpURLConnection connection) throws IOException {
    InputStream errorStream = connection.getErrorStream();
    Charset encoding = getEncoding(connection);
    if (errorStream != null) {
      return new String(ByteStreams.toByteArray(errorStream), encoding);
    }

    InputStream responseStream = connection.getInputStream();
    if (responseStream != null) {
      return new String(ByteStreams.toByteArray(responseStream), encoding);
    }

    return null;
  }
}
