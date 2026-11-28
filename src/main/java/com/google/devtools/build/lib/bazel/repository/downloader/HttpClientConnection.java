// Copyright 2026 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSet;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.ProxySelector;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.UnknownHostException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.ByteBuffer;
import java.nio.channels.UnresolvedAddressException;
import java.time.Duration;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Flow;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/**
 * An {@link HttpURLConnection} backed by {@link HttpClient}, which speaks HTTP/2 to servers that
 * support it.
 *
 * <p>Connections to the same server through the same proxy and with the same connect timeout share
 * a pool for the lifetime of the Bazel server. The timeouts have to be set before connecting.
 */
final class HttpClientConnection extends HttpURLConnection {

  static {
    // Lets Proxy-Authorization headers reach the proxy when tunneling HTTPS. The client only reads
    // this property once.
    System.getProperties().putIfAbsent("jdk.http.auth.tunneling.disabledSchemes", "");
  }

  private record ClientKey(@Nullable InetSocketAddress proxy, int connectTimeoutMillis) {}

  private static final ConcurrentHashMap<ClientKey, HttpClient> clients = new ConcurrentHashMap<>();

  // HttpClient rejects requests that set these headers.
  private static final ImmutableSet<String> RESTRICTED_HEADERS =
      ImmutableSet.of("connection", "content-length", "expect", "host", "upgrade");

  private final URI uri;
  @Nullable private final InetSocketAddress proxy;
  @Nullable private HttpResponse<Body> response;

  /**
   * @param proxy the address of an HTTP proxy, or null to connect directly
   */
  HttpClientConnection(URI uri, @Nullable InetSocketAddress proxy) throws IOException {
    super(uri.toURL());
    this.uri = uri;
    this.proxy = proxy;
  }

  @Override
  public void connect() throws IOException {
    if (response != null) {
      return;
    }
    HttpRequest.Builder request = HttpRequest.newBuilder(uri);
    if (HttpUtils.isProtocol(uri, "http")) {
      // The client would otherwise attempt an h2c upgrade, which some servers and proxies choke on.
      request.version(HttpClient.Version.HTTP_1_1);
    }
    if (getConnectTimeout() + getReadTimeout() > 0) {
      request.timeout(Duration.ofMillis(getConnectTimeout() + getReadTimeout()));
    }
    if (getRequestProperty("Accept") == null) {
      request.header("Accept", "*/*");
    }
    for (Map.Entry<String, List<String>> header : getRequestProperties().entrySet()) {
      if (RESTRICTED_HEADERS.contains(Ascii.toLowerCase(header.getKey()))) {
        continue;
      }
      for (String value : header.getValue()) {
        request.header(header.getKey(), value);
      }
    }
    HttpClient client =
        clients.computeIfAbsent(
            new ClientKey(proxy, getConnectTimeout()),
            key -> {
              HttpClient.Builder builder =
                  HttpClient.newBuilder()
                      .followRedirects(HttpClient.Redirect.NEVER)
                      .proxy(
                          key.proxy() == null
                              ? HttpClient.Builder.NO_PROXY
                              : ProxySelector.of(key.proxy()));
              if (key.connectTimeoutMillis() > 0) {
                builder.connectTimeout(Duration.ofMillis(key.connectTimeoutMillis()));
              }
              return builder.build();
            });
    HttpResponse.BodyHandler<Body> bodyHandler = unused -> new Body(getReadTimeout());
    try {
      try {
        response = client.send(request.build(), bodyHandler);
      } catch (IOException e) {
        // The server allows fewer concurrent streams on its HTTP/2 connection than there are
        // parallel downloads, so let this one use a separate HTTP/1.1 connection.
        if (!"too many concurrent streams".equals(e.getMessage())) {
          throw e;
        }
        response = client.send(request.version(HttpClient.Version.HTTP_1_1).build(), bodyHandler);
      }
    } catch (HttpTimeoutException e) {
      throw (IOException) new SocketTimeoutException(e.getMessage()).initCause(e);
    } catch (ConnectException e) {
      if (Throwables.getRootCause(e) instanceof UnresolvedAddressException) {
        throw (IOException) new UnknownHostException(uri.getHost()).initCause(e);
      }
      throw e;
    } catch (InterruptedException e) {
      throw (IOException) new InterruptedIOException().initCause(e);
    }
    connected = true;
  }

  @Override
  public int getResponseCode() throws IOException {
    connect();
    return response.statusCode();
  }

  /**
   * Returns the standard reason phrase for the status code since HTTP/2 has no reason phrases and
   * {@link HttpClient} doesn't expose those of HTTP/1.1.
   */
  @Override
  @Nullable
  public String getResponseMessage() throws IOException {
    return switch (getResponseCode()) {
      case 400 -> "Bad Request";
      case 401 -> "Unauthorized";
      case 403 -> "Forbidden";
      case 404 -> "Not Found";
      case 407 -> "Proxy Authentication Required";
      case 408 -> "Request Timeout";
      case 410 -> "Gone";
      case 429 -> "Too Many Requests";
      case 500 -> "Internal Server Error";
      case 501 -> "Not Implemented";
      case 502 -> "Bad Gateway";
      case 503 -> "Service Unavailable";
      case 504 -> "Gateway Timeout";
      case 505 -> "HTTP Version Not Supported";
      default -> null;
    };
  }

  @Override
  @Nullable
  public String getHeaderField(String name) {
    try {
      connect();
    } catch (IOException e) {
      return null;
    }
    return response.headers().firstValue(name).orElse(null);
  }

  @Override
  public Map<String, List<String>> getHeaderFields() {
    try {
      connect();
    } catch (IOException e) {
      return Map.of();
    }
    return response.headers().map();
  }

  @Override
  public InputStream getInputStream() throws IOException {
    connect();
    return response.body();
  }

  @Override
  @Nullable
  public InputStream getErrorStream() {
    return response != null && response.statusCode() >= 400 ? response.body() : null;
  }

  @Override
  public void disconnect() {
    if (response != null) {
      response.body().close();
    }
  }

  @Override
  public boolean usingProxy() {
    return proxy != null;
  }

  /**
   * The response body as a stream with a read timeout, which {@link
   * HttpResponse.BodySubscribers#ofInputStream} lacks.
   */
  private static final class Body extends InputStream
      implements HttpResponse.BodySubscriber<Body> {
    private static final List<ByteBuffer> EOF = List.of();

    private final int readTimeoutMillis;
    // Holds a List<ByteBuffer> per chunk, followed by either EOF or a Throwable.
    private final LinkedBlockingQueue<Object> chunks = new LinkedBlockingQueue<>();
    @Nullable private volatile Flow.Subscription subscription;
    private volatile boolean closed;
    private Iterator<ByteBuffer> buffers = EOF.iterator();
    @Nullable private ByteBuffer buffer;
    private boolean done;

    Body(int readTimeoutMillis) {
      this.readTimeoutMillis = readTimeoutMillis;
    }

    @Override
    public CompletionStage<Body> getBody() {
      return CompletableFuture.completedStage(this);
    }

    @Override
    public void onSubscribe(Flow.Subscription subscription) {
      this.subscription = subscription;
      if (closed) {
        subscription.cancel();
      } else {
        subscription.request(1);
      }
    }

    @Override
    public void onNext(List<ByteBuffer> item) {
      chunks.add(item);
    }

    @Override
    public void onError(Throwable throwable) {
      chunks.add(throwable);
    }

    @Override
    public void onComplete() {
      chunks.add(EOF);
    }

    @Override
    public int read() throws IOException {
      byte[] b = new byte[1];
      return read(b, 0, 1) == -1 ? -1 : b[0] & 0xff;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
      if (len == 0) {
        return 0;
      }
      while (buffer == null || !buffer.hasRemaining()) {
        if (closed) {
          throw new IOException("closed");
        }
        if (done) {
          return -1;
        }
        if (buffers.hasNext()) {
          buffer = buffers.next();
          continue;
        }
        Object chunk;
        try {
          chunk =
              readTimeoutMillis > 0
                  ? chunks.poll(readTimeoutMillis, TimeUnit.MILLISECONDS)
                  : chunks.take();
        } catch (InterruptedException e) {
          close();
          throw (IOException) new InterruptedIOException().initCause(e);
        }
        switch (chunk) {
          case null -> {
            close();
            throw new SocketTimeoutException("Read timed out");
          }
          case Throwable t -> {
            close();
            throw new IOException(t.getMessage(), t);
          }
          case List<?> list when list == EOF -> done = true;
          case List<?> list -> {
            @SuppressWarnings("unchecked")
            var byteBuffers = (List<ByteBuffer>) list;
            buffers = byteBuffers.iterator();
            subscription.request(1);
          }
          default -> throw new IllegalStateException(chunk.toString());
        }
      }
      int read = Math.min(len, buffer.remaining());
      buffer.get(b, off, read);
      return read;
    }

    @Override
    public void close() {
      closed = true;
      Flow.Subscription subscription = this.subscription;
      if (subscription != null) {
        subscription.cancel();
      }
    }
  }
}
