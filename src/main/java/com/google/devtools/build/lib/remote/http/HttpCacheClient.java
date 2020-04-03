// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.http;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.auth.Credentials;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashingOutputStream;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.pool.ChannelPool;
import io.netty.channel.pool.ChannelPoolHandler;
import io.netty.channel.pool.FixedChannelPool;
import io.netty.channel.pool.SimpleChannelPool;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequestEncoder;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpResponseDecoder;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.ssl.OpenSsl;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslHandler;
import io.netty.handler.ssl.SslProvider;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.handler.timeout.ReadTimeoutException;
import io.netty.handler.timeout.WriteTimeoutException;
import io.netty.util.concurrent.Future;
import io.netty.util.concurrent.Promise;
import io.netty.util.internal.PlatformDependent;
import java.io.FileInputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.URI;
import java.util.List;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;
import javax.net.ssl.SSLEngine;

/**
 * Implementation of {@link RemoteCacheClient} that can talk to a HTTP/1.1 backend.
 *
 * <p>Blobs (Binary large objects) are uploaded using the {@code PUT} method. Action cache blobs are
 * stored under the path {@code /ac/base16-key}. CAS (Content Addressable Storage) blobs are stored
 * under the path {@code /cas/base16-key}. Valid status codes for a successful upload are 200 (OK),
 * 201 (CREATED), 202 (ACCEPTED) and 204 (NO CONTENT). It's recommended to return 200 (OK) on
 * success. The other status codes are supported to be compatibility with the nginx webdav module
 * and may be removed in the future.
 *
 * <p>Blobs are downloaded using the {@code GET} method at the paths they were stored at. A status
 * code of 200 should be followed by the content of the blob. The status codes 404 (NOT FOUND) and
 * 204 (NO CONTENT) indicate that no cache entry exists. It's recommended to return 404 (NOT FOUND)
 * as the 204 (NO CONTENT) status code is only supported for compatibility with the nginx webdav
 * module.
 *
 * <p>TLS is supported and enabled automatically when using HTTPS as the URI scheme.
 *
 * <p>Uploads do not use {@code Expect: 100-CONTINUE} headers, as this would incur an additional
 * roundtrip for every upload and with little practical value as we would expect most uploads to be
 * accepted.
 *
 * <p>The implementation currently does not support transfer encoding chunked.
 */
public final class HttpCacheClient implements RemoteCacheClient {

  public static final String AC_PREFIX = "ac/";
  public static final String CAS_PREFIX = "cas/";
  private static final Pattern INVALID_TOKEN_ERROR =
      Pattern.compile("\\s*error\\s*=\\s*\"?invalid_token\"?");

  private final ConcurrentHashMap<String, Boolean> storedBlobs = new ConcurrentHashMap<>();

  private final EventLoopGroup eventLoop;
  private final ChannelPool channelPool;
  private final URI uri;
  private final int timeoutSeconds;
  private final ImmutableList<Entry<String, String>> extraHttpHeaders;
  private final boolean useTls;
  private final boolean verifyDownloads;
  private final DigestUtil digestUtil;

  private final Object closeLock = new Object();

  @GuardedBy("closeLock")
  private boolean isClosed;

  private final Object credentialsLock = new Object();

  @GuardedBy("credentialsLock")
  private final Credentials creds;

  @GuardedBy("credentialsLock")
  private long lastRefreshTime;

  public static HttpCacheClient create(
      URI uri,
      int timeoutSeconds,
      int remoteMaxConnections,
      boolean verifyDownloads,
      ImmutableList<Entry<String, String>> extraHttpHeaders,
      DigestUtil digestUtil,
      @Nullable final Credentials creds)
      throws Exception {
    return new HttpCacheClient(
        NioEventLoopGroup::new,
        NioSocketChannel.class,
        uri,
        timeoutSeconds,
        remoteMaxConnections,
        verifyDownloads,
        extraHttpHeaders,
        digestUtil,
        creds,
        null);
  }

  public static HttpCacheClient create(
      DomainSocketAddress domainSocketAddress,
      URI uri,
      int timeoutSeconds,
      int remoteMaxConnections,
      boolean verifyDownloads,
      ImmutableList<Entry<String, String>> extraHttpHeaders,
      DigestUtil digestUtil,
      @Nullable final Credentials creds)
      throws Exception {

    if (KQueue.isAvailable()) {
      return new HttpCacheClient(
          KQueueEventLoopGroup::new,
          KQueueDomainSocketChannel.class,
          uri,
          timeoutSeconds,
          remoteMaxConnections,
          verifyDownloads,
          extraHttpHeaders,
          digestUtil,
          creds,
          domainSocketAddress);
    } else if (Epoll.isAvailable()) {
      return new HttpCacheClient(
          EpollEventLoopGroup::new,
          EpollDomainSocketChannel.class,
          uri,
          timeoutSeconds,
          remoteMaxConnections,
          verifyDownloads,
          extraHttpHeaders,
          digestUtil,
          creds,
          domainSocketAddress);
    } else {
      throw new Exception("Unix domain sockets are unsupported on this platform");
    }
  }

  private HttpCacheClient(
      Function<Integer, EventLoopGroup> newEventLoopGroup,
      Class<? extends Channel> channelClass,
      URI uri,
      int timeoutSeconds,
      int remoteMaxConnections,
      boolean verifyDownloads,
      ImmutableList<Entry<String, String>> extraHttpHeaders,
      DigestUtil digestUtil,
      @Nullable final Credentials creds,
      @Nullable SocketAddress socketAddress)
      throws Exception {
    useTls = uri.getScheme().equals("https");
    if (uri.getPort() == -1) {
      int port = useTls ? 443 : 80;
      uri =
          new URI(
              uri.getScheme(),
              uri.getUserInfo(),
              uri.getHost(),
              port,
              uri.getPath(),
              uri.getQuery(),
              uri.getFragment());
    }
    this.uri = uri;
    if (socketAddress == null) {
      socketAddress = new InetSocketAddress(uri.getHost(), uri.getPort());
    }

    final SslContext sslCtx;
    if (useTls) {
      // OpenSsl gives us a > 2x speed improvement on fast networks, but requires netty tcnative
      // to be there which is not available on all platforms and environments.
      SslProvider sslProvider = OpenSsl.isAvailable() ? SslProvider.OPENSSL : SslProvider.JDK;
      sslCtx = SslContextBuilder.forClient().sslProvider(sslProvider).build();
    } else {
      sslCtx = null;
    }
    final int port = uri.getPort();
    final String hostname = uri.getHost();
    this.eventLoop = newEventLoopGroup.apply(2);
    Bootstrap clientBootstrap =
        new Bootstrap()
            .channel(channelClass)
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 1000 * timeoutSeconds)
            .group(eventLoop)
            .remoteAddress(socketAddress);

    ChannelPoolHandler channelPoolHandler =
        new ChannelPoolHandler() {
          @Override
          public void channelReleased(Channel ch) {}

          @Override
          public void channelAcquired(Channel ch) {}

          @Override
          public void channelCreated(Channel ch) {
            ChannelPipeline p = ch.pipeline();
            if (sslCtx != null) {
              SSLEngine engine = sslCtx.newEngine(ch.alloc(), hostname, port);
              engine.setUseClientMode(true);
              p.addFirst("ssl-handler", new SslHandler(engine));
            }
          }
        };
    if (remoteMaxConnections > 0) {
      channelPool = new FixedChannelPool(clientBootstrap, channelPoolHandler, remoteMaxConnections);
    } else {
      channelPool = new SimpleChannelPool(clientBootstrap, channelPoolHandler);
    }
    this.creds = creds;
    this.timeoutSeconds = timeoutSeconds;
    this.extraHttpHeaders = extraHttpHeaders;
    this.verifyDownloads = verifyDownloads;
    this.digestUtil = digestUtil;
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private Channel acquireUploadChannel() throws InterruptedException {
    Promise<Channel> channelReady = eventLoop.next().newPromise();
    channelPool
        .acquire()
        .addListener(
            (Future<Channel> channelAcquired) -> {
              if (!channelAcquired.isSuccess()) {
                channelReady.setFailure(channelAcquired.cause());
                return;
              }

              try {
                Channel ch = channelAcquired.getNow();
                ChannelPipeline p = ch.pipeline();

                if (!isChannelPipelineEmpty(p)) {
                  channelReady.setFailure(
                      new IllegalStateException("Channel pipeline is not empty."));
                  return;
                }

                p.addFirst(
                    "timeout-handler",
                    new IdleTimeoutHandler(timeoutSeconds, WriteTimeoutException.INSTANCE));
                p.addLast(new HttpResponseDecoder());
                // The 10KiB limit was chosen arbitrarily. We only expect HTTP servers to respond
                // with an error message in the body, and that should always be less than 10KiB. If
                // the response is larger than 10KiB, HttpUploadHandler will catch the
                // TooLongFrameException that HttpObjectAggregator throws and convert it to an
                // IOException.
                p.addLast(new HttpObjectAggregator(10 * 1024));
                p.addLast(new HttpRequestEncoder());
                p.addLast(new ChunkedWriteHandler());
                synchronized (credentialsLock) {
                  p.addLast(new HttpUploadHandler(creds, extraHttpHeaders));
                }

                if (!ch.eventLoop().inEventLoop()) {
                  // If addLast is called outside an event loop, then it doesn't complete until the
                  // event loop is run again. In that case, a message sent to the last handler gets
                  // delivered to the last non-pending handler, which will most likely end up
                  // throwing UnsupportedMessageTypeException. Therefore, we only complete the
                  // promise in the event loop.
                  ch.eventLoop().execute(() -> channelReady.setSuccess(ch));
                } else {
                  channelReady.setSuccess(ch);
                }
              } catch (Throwable t) {
                channelReady.setFailure(t);
              }
            });

    try {
      return channelReady.get();
    } catch (ExecutionException e) {
      PlatformDependent.throwException(e.getCause());
      return null;
    }
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private void releaseUploadChannel(Channel ch) {
    if (ch.isOpen()) {
      try {
        ch.pipeline().remove(IdleTimeoutHandler.class);
        ch.pipeline().remove(HttpResponseDecoder.class);
        ch.pipeline().remove(HttpObjectAggregator.class);
        ch.pipeline().remove(HttpRequestEncoder.class);
        ch.pipeline().remove(ChunkedWriteHandler.class);
        ch.pipeline().remove(HttpUploadHandler.class);
      } catch (NoSuchElementException e) {
        // If the channel is in the process of closing but not yet closed, some handlers could have
        // been removed and would cause NoSuchElement exceptions to be thrown. Because handlers are
        // removed in reverse-order, if we get a NoSuchElement exception, the following handlers
        // should have been removed.
      }
    }
    channelPool.release(ch);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private Future<Channel> acquireDownloadChannel() {
    Promise<Channel> channelReady = eventLoop.next().newPromise();
    channelPool
        .acquire()
        .addListener(
            (Future<Channel> channelAcquired) -> {
              if (!channelAcquired.isSuccess()) {
                channelReady.setFailure(channelAcquired.cause());
                return;
              }

              try {
                Channel ch = channelAcquired.getNow();
                ChannelPipeline p = ch.pipeline();

                if (!isChannelPipelineEmpty(p)) {
                  channelReady.setFailure(
                      new IllegalStateException("Channel pipeline is not empty."));
                  return;
                }
                p.addFirst(
                    "timeout-handler",
                    new IdleTimeoutHandler(timeoutSeconds, ReadTimeoutException.INSTANCE));
                p.addLast(new HttpClientCodec());
                synchronized (credentialsLock) {
                  p.addLast(new HttpDownloadHandler(creds, extraHttpHeaders));
                }

                if (!ch.eventLoop().inEventLoop()) {
                  // If addLast is called outside an event loop, then it doesn't complete until the
                  // event loop is run again. In that case, a message sent to the last handler gets
                  // delivered to the last non-pending handler, which will most likely end up
                  // throwing UnsupportedMessageTypeException. Therefore, we only complete the
                  // promise in the event loop.
                  ch.eventLoop().execute(() -> channelReady.setSuccess(ch));
                } else {
                  channelReady.setSuccess(ch);
                }
              } catch (Throwable t) {
                channelReady.setFailure(t);
              }
            });

    return channelReady;
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private void releaseDownloadChannel(Channel ch) {
    if (ch.isOpen()) {
      // The channel might have been closed due to an error, in which case its pipeline
      // has already been cleared. Closed channels can't be reused.
      try {
        ch.pipeline().remove(IdleTimeoutHandler.class);
        ch.pipeline().remove(HttpClientCodec.class);
        ch.pipeline().remove(HttpDownloadHandler.class);
      } catch (NoSuchElementException e) {
        // If the channel is in the process of closing but not yet closed, some handlers could have
        // been removed and would cause NoSuchElement exceptions to be thrown. Because handlers are
        // removed in reverse-order, if we get a NoSuchElement exception, the following handlers
        // should have been removed.
      }
    }
    channelPool.release(ch);
  }

  private boolean isChannelPipelineEmpty(ChannelPipeline pipeline) {
    return (pipeline.first() == null)
        || (useTls
            && "ssl-handler".equals(pipeline.firstContext().name())
            && pipeline.first() == pipeline.last());
  }

  @Override
  public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    final HashingOutputStream hashOut =
        verifyDownloads ? digestUtil.newHashingOutputStream(out) : null;
    return Futures.transformAsync(
        get(digest, hashOut != null ? hashOut : out, /* casDownload= */ true),
        (v) -> {
          try {
            if (hashOut != null) {
              Utils.verifyBlobContents(
                  digest.getHash(), DigestUtil.hashCodeToString(hashOut.hash()));
            }
            out.flush();
            return Futures.immediateFuture(null);
          } catch (IOException e) {
            return Futures.immediateFailedFuture(e);
          }
        },
        MoreExecutors.directExecutor());
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private ListenableFuture<Void> get(Digest digest, final OutputStream out, boolean casDownload) {
    final AtomicBoolean dataWritten = new AtomicBoolean();
    OutputStream wrappedOut =
        new OutputStream() {
          // OutputStream.close() does nothing, which is what we want to ensure that the
          // OutputStream can't be closed somewhere in the Netty pipeline, so that we can support
          // retries. The OutputStream is closed in the finally block below.

          @Override
          public void write(byte[] b, int offset, int length) throws IOException {
            dataWritten.set(true);
            out.write(b, offset, length);
          }

          @Override
          public void write(int b) throws IOException {
            dataWritten.set(true);
            out.write(b);
          }

          @Override
          public void flush() throws IOException {
            out.flush();
          }
        };
    DownloadCommand downloadCmd = new DownloadCommand(uri, casDownload, digest, wrappedOut);
    SettableFuture<Void> outerF = SettableFuture.create();
    acquireDownloadChannel()
        .addListener(
            (Future<Channel> chP) -> {
              if (!chP.isSuccess()) {
                outerF.setException(chP.cause());
                return;
              }

              Channel ch = chP.getNow();
              ch.writeAndFlush(downloadCmd)
                  .addListener(
                      (f) -> {
                        try {
                          if (f.isSuccess()) {
                            outerF.set(null);
                          } else {
                            Throwable cause = f.cause();
                            // cause can be of type HttpException, because Netty uses
                            // Unsafe.throwException to
                            // re-throw a checked exception that hasn't been declared in the method
                            // signature.
                            if (cause instanceof HttpException) {
                              HttpResponse response = ((HttpException) cause).response();
                              if (!dataWritten.get() && authTokenExpired(response)) {
                                // The error is due to an auth token having expired. Let's try
                                // again.
                                refreshCredentials();
                                getAfterCredentialRefresh(downloadCmd, outerF);
                                return;
                              } else if (cacheMiss(response.status())) {
                                outerF.setException(new CacheNotFoundException(digest));
                                return;
                              }
                            }
                            outerF.setException(cause);
                          }
                        } finally {
                          releaseDownloadChannel(ch);
                        }
                      });
            });
    return outerF;
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private void getAfterCredentialRefresh(DownloadCommand cmd, SettableFuture<Void> outerF) {
    acquireDownloadChannel()
        .addListener(
            (Future<Channel> chP) -> {
              if (!chP.isSuccess()) {
                outerF.setException(chP.cause());
                return;
              }

              Channel ch = chP.getNow();
              ch.writeAndFlush(cmd)
                  .addListener(
                      (f) -> {
                        try {
                          if (f.isSuccess()) {
                            outerF.set(null);
                          } else {
                            Throwable cause = f.cause();
                            if (cause instanceof HttpException) {
                              HttpResponse response = ((HttpException) cause).response();
                              if (cacheMiss(response.status())) {
                                outerF.setException(new CacheNotFoundException(cmd.digest()));
                                return;
                              }
                            }
                            outerF.setException(cause);
                          }
                        } finally {
                          releaseDownloadChannel(ch);
                        }
                      });
            });
  }

  @Override
  public ListenableFuture<ActionResult> downloadActionResult(
      ActionKey actionKey, boolean inlineOutErr) {
    return Utils.downloadAsActionResult(
        actionKey, (digest, out) -> get(digest, out, /* casDownload= */ false));
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private void uploadBlocking(String key, long length, InputStream in, boolean casUpload)
      throws IOException, InterruptedException {
    InputStream wrappedIn =
        new FilterInputStream(in) {
          @Override
          public void close() {
            // Ensure that the InputStream can't be closed somewhere in the Netty
            // pipeline, so that we can support retries. The InputStream is closed in
            // the finally block below.
          }
        };
    UploadCommand upload = new UploadCommand(uri, casUpload, key, wrappedIn, length);
    Channel ch = null;
    boolean success = false;
    if (storedBlobs.putIfAbsent((casUpload ? CAS_PREFIX : AC_PREFIX) + key, true) == null) {
      try {
        ch = acquireUploadChannel();
        ChannelFuture uploadFuture = ch.writeAndFlush(upload);
        uploadFuture.sync();
        success = true;
      } catch (Exception e) {
        // e can be of type HttpException, because Netty uses Unsafe.throwException to re-throw a
        // checked exception that hasn't been declared in the method signature.
        if (e instanceof HttpException) {
          HttpResponse response = ((HttpException) e).response();
          if (authTokenExpired(response)) {
            refreshCredentials();
            // The error is due to an auth token having expired. Let's try again.
            if (!reset(in)) {
              // The InputStream can't be reset and thus we can't retry as most likely
              // bytes have already been read from the InputStream.
              throw e;
            }
            putAfterCredentialRefresh(upload);
            success = true;
            return;
          }
        }
        throw e;
      } finally {
        if (!success) {
          storedBlobs.remove(key);
        }
        in.close();
        if (ch != null) {
          releaseUploadChannel(ch);
        }
      }
    }
  }

  @Override
  public ListenableFuture<Void> uploadFile(Digest digest, Path file) {
    try (InputStream in = file.getInputStream()) {
      uploadBlocking(digest.getHash(), digest.getSizeBytes(), in, /* casUpload= */ true);
    } catch (IOException | InterruptedException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(Digest digest, ByteString data) {
    try (InputStream in = data.newInput()) {
      uploadBlocking(digest.getHash(), digest.getSizeBytes(), in, /* casUpload= */ true);
    } catch (IOException | InterruptedException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
    return Futures.immediateFuture(ImmutableSet.copyOf(digests));
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private void putAfterCredentialRefresh(UploadCommand cmd) throws InterruptedException {
    Channel ch = null;
    try {
      // TODO(buchgr): Look into simplifying the retry logic after a credentials refresh.
      ch = acquireUploadChannel();
      ChannelFuture uploadFuture = ch.writeAndFlush(cmd);
      uploadFuture.sync();
    } finally {
      if (ch != null) {
        releaseUploadChannel(ch);
      }
    }
  }

  private boolean reset(InputStream in) throws IOException {
    if (in.markSupported()) {
      in.reset();
      return true;
    }
    if (in instanceof FileInputStream) {
      // FileInputStream does not support reset().
      ((FileInputStream) in).getChannel().position(0);
      return true;
    }
    return false;
  }

  @Override
  public void uploadActionResult(ActionKey actionKey, ActionResult actionResult)
      throws IOException, InterruptedException {
    ByteString serialized = actionResult.toByteString();
    try (InputStream in = serialized.newInput()) {
      uploadBlocking(actionKey.getDigest().getHash(), serialized.size(), in, false);
    }
  }

  /**
   * It's safe to suppress this warning because all methods on Netty futures return {@code this}. So
   * we are not ignoring anything.
   */
  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void close() {
    synchronized (closeLock) {
      if (isClosed) {
        return;
      }

      isClosed = true;
      channelPool.close();
      eventLoop.shutdownGracefully();
    }
  }

  private boolean cacheMiss(HttpResponseStatus status) {
    // Supporting NO_CONTENT for nginx webdav compatibility.
    return status.equals(HttpResponseStatus.NOT_FOUND)
        || status.equals(HttpResponseStatus.NO_CONTENT);
  }

  /** See https://tools.ietf.org/html/rfc6750#section-3.1 */
  private boolean authTokenExpired(HttpResponse response) {
    synchronized (credentialsLock) {
      if (creds == null) {
        return false;
      }
    }
    List<String> values = response.headers().getAllAsString(HttpHeaderNames.WWW_AUTHENTICATE);
    String value = String.join(",", values);
    if (value != null && value.startsWith("Bearer")) {
      return INVALID_TOKEN_ERROR.matcher(value).find();
    } else {
      return response.status().equals(HttpResponseStatus.UNAUTHORIZED);
    }
  }

  private void refreshCredentials() throws IOException {
    synchronized (credentialsLock) {
      long now = System.currentTimeMillis();
      // Call creds.refresh() at most once per second. The one second was arbitrarily chosen, as
      // a small enough value that we don't expect to interfere with actual token lifetimes, but
      // it should just make sure that potentially hundreds of threads don't call this method
      // at the same time.
      if ((now - lastRefreshTime) > TimeUnit.SECONDS.toMillis(1)) {
        lastRefreshTime = now;
        creds.refresh();
      }
    }
  }
}
