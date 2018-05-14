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

import static com.google.common.truth.Truth.assertThat;
import static java.util.Collections.singletonList;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.auth.Credentials;
import com.google.common.base.Charsets;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStoreTest.NotAuthorizedHandler.ErrorType;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandler.Sharable;
import io.netty.channel.ChannelHandlerAdapter;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.ServerSocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.timeout.ReadTimeoutException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.net.ConnectException;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link HttpBlobStore}. */
@RunWith(JUnit4.class)
public class HttpBlobStoreTest {

  private ServerSocketChannel startServer(ChannelHandler handler) throws Exception {
    EventLoopGroup eventLoop = new NioEventLoopGroup(1);
    ServerBootstrap sb =
        new ServerBootstrap()
            .group(eventLoop)
            .channel(NioServerSocketChannel.class)
            .childHandler(
                new ChannelInitializer<NioSocketChannel>() {
                  @Override
                  protected void initChannel(NioSocketChannel ch) {
                    ch.pipeline().addLast(new HttpServerCodec());
                    ch.pipeline().addLast(new HttpObjectAggregator(1000));
                    ch.pipeline().addLast(handler);
                  }
                });
    return ((ServerSocketChannel) sb.bind("localhost", 0).sync().channel());
  }

  @Test(expected = ConnectException.class, timeout = 30000)
  public void timeoutShouldWork_connect() throws Exception {
    ServerSocketChannel server = startServer(new ChannelHandlerAdapter() {});
    int serverPort = server.localAddress().getPort();
    closeServerChannel(server);

    Credentials credentials = newCredentials();
    HttpBlobStore blobStore =
        new HttpBlobStore(new URI("http://localhost:" + serverPort), 5, credentials);
    blobStore.get("key", new ByteArrayOutputStream());
    fail("Exception expected");
  }

  @Test(expected = ReadTimeoutException.class, timeout = 30000)
  public void timeoutShouldWork_read() throws Exception {
    ServerSocketChannel server = null;
    try {
      server =
          startServer(
              new SimpleChannelInboundHandler<FullHttpRequest>() {
                @Override
                protected void channelRead0(
                    ChannelHandlerContext channelHandlerContext, FullHttpRequest fullHttpRequest) {
                  // Don't respond and force a client timeout.
                }
              });
      int serverPort = server.localAddress().getPort();

      Credentials credentials = newCredentials();
      HttpBlobStore blobStore =
          new HttpBlobStore(new URI("http://localhost:" + serverPort), 5, credentials);
      blobStore.get("key", new ByteArrayOutputStream());
      fail("Exception expected");
    } finally {
      closeServerChannel(server);
    }
  }

  @Test
  public void expiredAuthTokensShouldBeRetried_get() throws Exception {
    expiredAuthTokensShouldBeRetried_get(ErrorType.UNAUTHORIZED);
    expiredAuthTokensShouldBeRetried_get(ErrorType.INVALID_TOKEN);
  }

  private void expiredAuthTokensShouldBeRetried_get(ErrorType errorType) throws Exception {
    ServerSocketChannel server = null;
    try {
      server = startServer(new NotAuthorizedHandler(errorType));
      int serverPort = server.localAddress().getPort();

      Credentials credentials = newCredentials();
      HttpBlobStore blobStore =
          new HttpBlobStore(new URI("http://localhost:" + serverPort), 30, credentials);
      ByteArrayOutputStream out = Mockito.spy(new ByteArrayOutputStream());
      blobStore.get("key", out);
      assertThat(out.toString(Charsets.US_ASCII.name())).isEqualTo("File Contents");
      verify(credentials, times(1)).refresh();
      verify(credentials, times(2)).getRequestMetadata(any(URI.class));
      verify(credentials, times(2)).hasRequestMetadata();
      // The caller is responsible to the close the stream.
      verify(out, never()).close();
      verifyNoMoreInteractions(credentials);
    } finally {
      closeServerChannel(server);
    }
  }

  @Test
  public void expiredAuthTokensShouldBeRetried_put() throws Exception {
    expiredAuthTokensShouldBeRetried_put(ErrorType.UNAUTHORIZED);
    expiredAuthTokensShouldBeRetried_put(ErrorType.INVALID_TOKEN);
  }

  private void expiredAuthTokensShouldBeRetried_put(ErrorType errorType) throws Exception {
    ServerSocketChannel server = null;
    try {
      server = startServer(new NotAuthorizedHandler(errorType));
      int serverPort = server.localAddress().getPort();

      Credentials credentials = newCredentials();
      HttpBlobStore blobStore =
          new HttpBlobStore(new URI("http://localhost:" + serverPort), 30, credentials);
      byte[] data = "File Contents".getBytes(Charsets.US_ASCII);
      ByteArrayInputStream in = new ByteArrayInputStream(data);
      blobStore.put("key", data.length, in);
      verify(credentials, times(1)).refresh();
      verify(credentials, times(2)).getRequestMetadata(any(URI.class));
      verify(credentials, times(2)).hasRequestMetadata();
      verifyNoMoreInteractions(credentials);
    } finally {
      closeServerChannel(server);
    }
  }

  @Test
  public void errorCodesThatShouldNotBeRetried_get() throws InterruptedException {
    errorCodeThatShouldNotBeRetried_get(ErrorType.INSUFFICIENT_SCOPE);
    errorCodeThatShouldNotBeRetried_get(ErrorType.INVALID_REQUEST);
  }

  private void errorCodeThatShouldNotBeRetried_get(ErrorType errorType)
      throws InterruptedException {
    ServerSocketChannel server = null;
    try {
      server = startServer(new NotAuthorizedHandler(errorType));
      int serverPort = server.localAddress().getPort();

      Credentials credentials = newCredentials();
      HttpBlobStore blobStore =
          new HttpBlobStore(new URI("http://localhost:" + serverPort), 30, credentials);
      blobStore.get("key", new ByteArrayOutputStream());
      fail("Exception expected.");
    } catch (Exception e) {
      assertThat(e).isInstanceOf(HttpException.class);
      assertThat(((HttpException) e).response().status())
          .isEqualTo(HttpResponseStatus.UNAUTHORIZED);
    } finally {
      closeServerChannel(server);
    }
  }

  @Test
  public void errorCodesThatShouldNotBeRetried_put() throws InterruptedException {
    errorCodeThatShouldNotBeRetried_put(ErrorType.INSUFFICIENT_SCOPE);
    errorCodeThatShouldNotBeRetried_put(ErrorType.INVALID_REQUEST);
  }

  private void errorCodeThatShouldNotBeRetried_put(ErrorType errorType)
      throws InterruptedException {
    ServerSocketChannel server = null;
    try {
      server = startServer(new NotAuthorizedHandler(errorType));
      int serverPort = server.localAddress().getPort();

      Credentials credentials = newCredentials();
      HttpBlobStore blobStore =
          new HttpBlobStore(new URI("http://localhost:" + serverPort), 30, credentials);
      blobStore.put("key", 1, new ByteArrayInputStream(new byte[] {0}));
      fail("Exception expected.");
    } catch (Exception e) {
      assertThat(e).isInstanceOf(HttpException.class);
      assertThat(((HttpException) e).response().status())
          .isEqualTo(HttpResponseStatus.UNAUTHORIZED);
    } finally {
      closeServerChannel(server);
    }
  }

  private Credentials newCredentials() throws Exception {
    Credentials credentials = mock(Credentials.class);
    when(credentials.hasRequestMetadata()).thenReturn(true);
    Map<String, List<String>> headers = new HashMap<>();
    headers.put("Authorization", singletonList("Bearer invalidToken"));
    when(credentials.getRequestMetadata(any(URI.class))).thenReturn(headers);
    Mockito.doAnswer(
            (mock) -> {
              Map<String, List<String>> headers2 = new HashMap<>();
              headers2.put("Authorization", singletonList("Bearer validToken"));
              when(credentials.getRequestMetadata(any(URI.class))).thenReturn(headers2);
              return null;
            })
        .when(credentials)
        .refresh();
    return credentials;
  }

  /**
   * {@link ChannelHandler} that on the first request responds with a 401 UNAUTHORIZED status code,
   * which the client is expected to retry once with a new authentication token.
   */
  @Sharable
  static class NotAuthorizedHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    enum ErrorType {
      UNAUTHORIZED,
      INVALID_TOKEN,
      INSUFFICIENT_SCOPE,
      INVALID_REQUEST
    }

    private final ErrorType errorType;
    private int messageCount;

    NotAuthorizedHandler(ErrorType errorType) {
      this.errorType = errorType;
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
      if (messageCount == 0) {
        if (!"Bearer invalidToken".equals(request.headers().get(HttpHeaderNames.AUTHORIZATION))) {
          ctx.writeAndFlush(
                  new DefaultFullHttpResponse(
                      HttpVersion.HTTP_1_1, HttpResponseStatus.INTERNAL_SERVER_ERROR))
              .addListener(ChannelFutureListener.CLOSE);
          return;
        }
        final FullHttpResponse response;
        if (errorType == ErrorType.UNAUTHORIZED) {
          response =
              new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.UNAUTHORIZED);
        } else {
          response =
              new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.UNAUTHORIZED);
          response
              .headers()
              .set(
                  HttpHeaderNames.WWW_AUTHENTICATE,
                  "Bearer realm=\"localhost\","
                      + "error=\""
                      + errorType.name().toLowerCase()
                      + "\","
                      + "error_description=\"The access token expired\"");
        }
        ctx.writeAndFlush(response).addListener(ChannelFutureListener.CLOSE);
        messageCount++;
      } else if (messageCount == 1) {
        if (!"Bearer validToken".equals(request.headers().get(HttpHeaderNames.AUTHORIZATION))) {
          ctx.writeAndFlush(
                  new DefaultFullHttpResponse(
                      HttpVersion.HTTP_1_1, HttpResponseStatus.INTERNAL_SERVER_ERROR))
              .addListener(ChannelFutureListener.CLOSE);
          return;
        }
        ByteBuf content = ctx.alloc().buffer();
        content.writeCharSequence("File Contents", Charsets.US_ASCII);
        FullHttpResponse response =
            new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK, content);
        HttpUtil.setKeepAlive(response, true);
        HttpUtil.setContentLength(response, content.readableBytes());
        ctx.writeAndFlush(response);
        messageCount++;
      } else {
        // No third message expected.
        ctx.writeAndFlush(
                new DefaultFullHttpResponse(
                    HttpVersion.HTTP_1_1, HttpResponseStatus.INTERNAL_SERVER_ERROR))
            .addListener(ChannelFutureListener.CLOSE);
      }
    }
  }

  private void closeServerChannel(ServerSocketChannel server) throws InterruptedException {
    if (server != null) {
      server.close();
      server.closeFuture().sync();
      server.eventLoop().shutdownGracefully().sync();
    }
  }
}
