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
package com.google.devtools.build.lib.remote.blobstore.http;

import com.google.auth.Credentials;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.pool.ChannelPoolHandler;
import io.netty.channel.pool.SimpleChannelPool;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequestEncoder;
import io.netty.handler.codec.http.HttpResponseDecoder;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslHandler;
import io.netty.handler.ssl.SslProvider;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.util.internal.PlatformDependent;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import javax.net.ssl.SSLEngine;

/**
 * Implementation of {@link SimpleBlobStore} that can talk to a HTTP/1.1 backend.
 *
 * <p>Blobs (Binary large objects) are uploaded using the {@code PUT} method. Action cache blobs are
 * stored under the path {@code /ac/base16-key}. CAS (Content Addressable Storage) blobs are stored
 * under the path {@code /cas/base16-key}. Valid status codes for a successful upload are 200 (OK),
 * 201 (CREATED), 202 (ACCEPTED) and 204 (NO CONTENT). It's recommended to return 200 (OK) on
 * success. The other status codes are supported to be compatible with the nginx webdav module and
 * may be removed in the future.
 *
 * <p>Blobs are downloaded using the {@code GET} method at the paths they were stored at. A status
 * code of 200 should be followed by the content of the blob. The status codes 404 (NOT FOUND) and
 * 204 (NO CONTENT) indicate that no cache entry exists. It's recommended to return 404 (NOT FOUND)
 * as the 204 (NO CONTENT) status code is only supported to be compatible with the nginx webdav
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
public class HttpBlobStore implements SimpleBlobStore {

  private final NioEventLoopGroup eventLoop = new NioEventLoopGroup(8);
  private final SimpleChannelPool downloadChannels;
  private final SimpleChannelPool uploadChannels;
  private final URI uri;

  public HttpBlobStore(URI uri, int timeoutMillis, @Nullable final Credentials creds)
      throws Exception {
    boolean useTls = uri.getScheme().equals("https");
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

    final SslContext sslCtx;
    if (useTls) {
      sslCtx = SslContextBuilder.forClient().sslProvider(SslProvider.OPENSSL).build();
    } else {
      sslCtx = null;
    }
    Bootstrap clientBootstrap =
        new Bootstrap()
            .channel(NioSocketChannel.class)
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, timeoutMillis)
            .option(ChannelOption.SO_TIMEOUT, timeoutMillis)
            .group(eventLoop)
            .remoteAddress(uri.getHost(), uri.getPort());
    downloadChannels =
        new SimpleChannelPool(
            clientBootstrap,
            new ChannelPoolHandler() {
              @Override
              public void channelReleased(Channel ch) throws Exception {}

              @Override
              public void channelAcquired(Channel ch) throws Exception {}

              @Override
              public void channelCreated(Channel ch) throws Exception {
                ChannelPipeline p = ch.pipeline();
                if (sslCtx != null) {
                  SSLEngine engine = sslCtx.newEngine(ch.alloc());
                  engine.setUseClientMode(true);
                  p.addFirst(new SslHandler(engine));
                }
                p.addLast(new HttpClientCodec());
                p.addLast(new HttpDownloadHandler(creds));
              }
            });
    uploadChannels =
        new SimpleChannelPool(
            clientBootstrap,
            new ChannelPoolHandler() {
              @Override
              public void channelReleased(Channel ch) throws Exception {}

              @Override
              public void channelAcquired(Channel ch) throws Exception {}

              @Override
              public void channelCreated(Channel ch) throws Exception {
                ChannelPipeline p = ch.pipeline();
                if (sslCtx != null) {
                  SSLEngine engine = sslCtx.newEngine(ch.alloc());
                  engine.setUseClientMode(true);
                  p.addFirst(new SslHandler(engine));
                }
                p.addLast(new HttpResponseDecoder());
                // The 10KiB limit was chosen at random. We only expect HTTP servers to respond with
                // an error message in the body and that should always be less than 10KiB.
                p.addLast(new HttpObjectAggregator(10 * 1024));
                p.addLast(new HttpRequestEncoder());
                p.addLast(new ChunkedWriteHandler());
                p.addLast(new HttpUploadHandler(creds));
              }
            });
  }

  @Override
  public boolean containsKey(String key) throws IOException, InterruptedException {
    throw new UnsupportedOperationException("HTTP Caching does not use this method.");
  }

  @Override
  public boolean get(String key, OutputStream out) throws IOException, InterruptedException {
    return get(key, out, true);
  }

  @Override
  public boolean getActionResult(String actionKey, OutputStream out)
      throws IOException, InterruptedException {
    return get(actionKey, out, false);
  }

  @SuppressWarnings("all")
  private boolean get(String key, OutputStream out, boolean casDownload)
      throws IOException, InterruptedException {
    final Channel ch;
    try {
      ch = downloadChannels.acquire().get();
    } catch (ExecutionException e) {
      PlatformDependent.throwException(e.getCause());
      return false;
    }
    DownloadCommand download = new DownloadCommand(uri, casDownload, key, out);
    try {
      ChannelFuture downloadFuture = ch.writeAndFlush(download);
      downloadFuture.sync();
    } catch (Exception e) {
      // e can be of type HttpException, because Netty uses Unsafe.throwException to re-throw a
      // checked exception that hasn't been declared in the method signature.
      if (e instanceof HttpException) {
        HttpResponseStatus status = ((HttpException) e).status();
        if (status.equals(HttpResponseStatus.NOT_FOUND)
            || status.equals(HttpResponseStatus.NO_CONTENT)) {
          // Cache miss. Supporting NO_CONTENT for nginx webdav compatibility.
          return false;
        }
      }
      throw e;
    } finally {
      downloadChannels.release(ch);
    }
    return true;
  }

  @Override
  public void put(String key, long length, InputStream in)
      throws IOException, InterruptedException {
    put(key, length, in, true);
  }

  @Override
  public void putActionResult(String actionKey, byte[] in)
      throws IOException, InterruptedException {
    put(actionKey, in.length, new ByteArrayInputStream(in), false);
  }

  private void put(String key, long length, InputStream in, boolean casUpload)
      throws IOException, InterruptedException {
    final Channel ch;
    try {
      ch = uploadChannels.acquire().get();
    } catch (ExecutionException e) {
      throw new IOException("Failed to obtain a channel from the pool.", e);
    }
    UploadCommand upload = new UploadCommand(uri, casUpload, key, in, length);
    try {
      ChannelFuture uploadFuture = ch.writeAndFlush(upload);
      uploadFuture.sync();
    } finally {
      uploadChannels.release(ch);
    }
  }

  @Override
  public void close() {
    downloadChannels.close();
    uploadChannels.close();
    eventLoop.shutdownGracefully();
  }
}
