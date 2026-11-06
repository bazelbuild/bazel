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
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Input stream that reconnects on read timeouts and errors.
 *
 * <p>This class is not thread safe, but it is safe to message pass between threads.
 */
@ThreadCompatible
class RetryingInputStream extends InputStream {

  private static final int MAX_RESUMES = 3;

  /** Lambda for establishing a connection. */
  interface Reconnector {
    /** Establishes a connection with the same parameters as what was passed to us initially. */
    URLConnection connect(Throwable cause, ImmutableMap<String, List<String>> extraHeaders)
        throws IOException;
  }

  private InputStream delegate;
  private final Reconnector reconnector;
  @Nullable private final String contentEncoding;
  @Nullable private final String contentLength;
  private long toto;
  private int resumes;
  private final ArrayList<Throwable> suppressed = new ArrayList<>();

  /**
   * @param delegate the stream of the initial response
   * @param connection the connection of the initial response; its {@code Content-Encoding} and
   *     {@code Content-Length} are used to validate full-body responses to range requests
   * @param reconnector used to reconnect after read errors
   */
  RetryingInputStream(InputStream delegate, URLConnection connection, Reconnector reconnector) {
    this.delegate = delegate;
    this.reconnector = reconnector;
    this.contentEncoding = connection.getContentEncoding();
    this.contentLength = connection.getHeaderField("Content-Length");
  }

  @Override
  public int read() throws IOException {
    while (true) {
      try {
        int result = delegate.read();
        if (result != -1) {
          toto++;
        }
        return result;
      } catch (IOException e) {
        tryAgainIfPossible(e);
      }
    }
  }

  @Override
  public int read(byte[] buffer, int offset, int length) throws IOException {
    while (true) {
      try {
        int amount = delegate.read(buffer, offset, length);
        if (amount != -1) {
          toto += amount;
        }
        return amount;
      } catch (IOException e) {
        tryAgainIfPossible(e);
      }
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

  private void tryAgainIfPossible(IOException cause) throws IOException {
    if (cause instanceof InterruptedIOException && !(cause instanceof SocketTimeoutException)) {
      throw cause;
    }
    if (resumes >= MAX_RESUMES) {
      propagate(cause);
    }
    resumes++;
    try {
      delegate.close();
    } catch (Exception ignored) {
      // We know this connection failed so if it reminds us we're going to ignore it.
    }
    suppressed.add(cause);
    reconnectWhereWeLeftOff(cause);
  }

  private void reconnectWhereWeLeftOff(IOException cause) throws IOException {
    try {
      URLConnection connection;
      long amountRead = toto;
      if (amountRead == 0) {
        connection = reconnector.connect(cause, ImmutableMap.of());
      } else {
        connection =
            reconnector.connect(
                cause,
                ImmutableMap.of("Range", ImmutableList.of(String.format("bytes=%d-", amountRead))));
        // Servers and caches may ignore the Range header and send the full body. See RFC 9110
        // § 14.2.
        boolean fullBody =
            connection instanceof HttpURLConnection httpConnection
                ? httpConnection.getResponseCode() == HttpURLConnection.HTTP_OK
                : connection.getHeaderField("Content-Range") == null;
        if (fullBody) {
          // We count bytes as sent on the wire, so skipping only lands at the right offset if the
          // server sent the same representation as before.
          if (!Ascii.equalsIgnoreCase(
                  Strings.nullToEmpty(contentEncoding),
                  Strings.nullToEmpty(connection.getContentEncoding()))
              || (contentLength != null
                  && connection.getHeaderField("Content-Length") != null
                  && !contentLength.equals(connection.getHeaderField("Content-Length")))) {
            throw new IOException(
                String.format(
                    "Tried to reconnect at offset %,d but server sent a different representation",
                    amountRead));
          }
          delegate = new InterruptibleInputStream(connection.getInputStream());
          ByteStreams.skipFully(delegate, amountRead);
          return;
        }
        if (!Strings.nullToEmpty(connection.getHeaderField("Content-Range"))
            .startsWith(String.format("bytes %d-", amountRead))) {
          throw new IOException(
              String.format(
                  "Tried to reconnect at offset %,d but server didn't support it", amountRead));
        }
      }
      delegate = new InterruptibleInputStream(connection.getInputStream());
    } catch (InterruptedIOException e) {
      throw e;
    } catch (IOException e) {
      propagate(e);
    }
  }

  private <T extends Throwable> void propagate(T error) throws T {
    for (Throwable e : suppressed) {
      error.addSuppressed(e);
    }
    throw error;
  }
}
