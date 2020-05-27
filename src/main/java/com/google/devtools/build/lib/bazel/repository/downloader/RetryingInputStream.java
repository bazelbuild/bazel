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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.SocketTimeoutException;
import java.net.URLConnection;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

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
    URLConnection connect(
        Throwable cause, ImmutableMap<String, String> extraHeaders)
            throws IOException;
  }

  private volatile InputStream delegate;
  private final Reconnector reconnector;
  private final AtomicLong toto = new AtomicLong();
  private final AtomicInteger resumes = new AtomicInteger();
  private final Vector<Throwable> suppressed = new Vector<>();

  RetryingInputStream(InputStream delegate, Reconnector reconnector) {
    this.delegate = delegate;
    this.reconnector = reconnector;
  }

  @Override
  public int read() throws IOException {
    while (true) {
      try {
        int result = delegate.read();
        if (result != -1) {
          toto.incrementAndGet();
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
          toto.addAndGet(amount);
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
    if (resumes.incrementAndGet() > MAX_RESUMES) {
      propagate(cause);
    }
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
      long amountRead = toto.get();
      if (amountRead == 0) {
        connection = reconnector.connect(cause, ImmutableMap.<String, String>of());
      } else {
        connection =
            reconnector.connect(
                cause, ImmutableMap.of("Range", String.format("bytes=%d-", amountRead)));
        if (!Strings.nullToEmpty(connection.getHeaderField("Content-Range"))
                .startsWith(String.format("bytes %d-", amountRead))) {
          throw new IOException(String.format(
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
