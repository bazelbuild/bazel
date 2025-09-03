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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static java.util.Collections.singletonList;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.RemoteRetrier;
import com.google.devtools.build.lib.remote.Retrier;
import com.google.devtools.build.lib.remote.Retrier.ResultClassifier.Result;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.remote.worker.http.InMemoryHttpCacheServerHandler;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufUtil;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandler.Sharable;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandler;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.epoll.EpollServerDomainSocketChannel;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.kqueue.KQueueServerDomainSocketChannel;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.TooLongFrameException;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.DefaultHttpContent;
import io.netty.handler.codec.http.DefaultHttpResponse;
import io.netty.handler.codec.http.DefaultLastHttpContent;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.URI;
import java.nio.channels.ClosedChannelException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.function.IntFunction;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.mockito.AdditionalAnswers;
import org.mockito.Mockito;

/** Tests for {@link HttpCacheClient}. */
@RunWith(Parameterized.class)
@SuppressWarnings("FutureReturnValueIgnored")
public class HttpCacheClientTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final Digest DIGEST = DIGEST_UTIL.computeAsUtf8("File Contents");

  private RemoteActionExecutionContext remoteActionExecutionContext;

  private static ServerChannel createServer(
      Class<? extends ServerChannel> serverChannelClass,
      IntFunction<EventLoopGroup> newEventLoopGroup,
      SocketAddress socketAddress,
      ChannelHandler handler) {
    EventLoopGroup eventLoop = newEventLoopGroup.apply(1);
    ServerBootstrap sb =
        new ServerBootstrap()
            .group(eventLoop)
            .channel(serverChannelClass)
            .childHandler(
                new ChannelInitializer<Channel>() {
                  @Override
                  protected void initChannel(Channel ch) {
                    ch.pipeline().addLast(new HttpServerCodec());
                    ch.pipeline().addLast(new HttpObjectAggregator(1000));
                    ch.pipeline().addLast(handler);
                  }
                });
    try {
      return ((ServerChannel) sb.bind(socketAddress).sync().channel());
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  private static DomainSocketAddress newDomainSocketAddress() {
    try {
      File file = File.createTempFile("bazel", ".sock", new File("/tmp"));
      file.delete();
      return new DomainSocketAddress(file.getAbsoluteFile());
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  interface TestServer {

    ServerChannel start(ChannelInboundHandler handler);

    void stop(ServerChannel serverChannel);
  }

  private static final class InetTestServer implements TestServer {

    @Override
    public ServerChannel start(ChannelInboundHandler handler) {
      return createServer(
          NioServerSocketChannel.class,
          NioEventLoopGroup::new,
          new InetSocketAddress("localhost", 0),
          handler);
    }

    @Override
    public void stop(ServerChannel serverChannel) {
      try {
        serverChannel.close();
        serverChannel.closeFuture().sync();
        serverChannel.eventLoop().shutdownGracefully().sync();
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    }
  }

  private static final class UnixDomainServer implements TestServer {

    // Note: this odd implementation is a workaround because we're unable to shut down and restart
    // KQueue backed implementations. See https://github.com/netty/netty/issues/7047.

    private final ServerChannel serverChannel;
    private ChannelInboundHandler handler = null;

    public UnixDomainServer(
        Class<? extends ServerChannel> serverChannelClass,
        IntFunction<EventLoopGroup> newEventLoopGroup) {
      EventLoopGroup eventLoop = newEventLoopGroup.apply(1);
      ServerBootstrap sb =
          new ServerBootstrap()
              .group(eventLoop)
              .channel(serverChannelClass)
              .childHandler(
                  new ChannelInitializer<Channel>() {
                    @Override
                    protected void initChannel(Channel ch) {
                      ch.pipeline().addLast(new HttpServerCodec());
                      ch.pipeline().addLast(new HttpObjectAggregator(1000));
                      ch.pipeline().addLast(Preconditions.checkNotNull(handler));
                    }
                  });
      try {
        ServerChannel actual = ((ServerChannel) sb.bind(newDomainSocketAddress()).sync().channel());
        this.serverChannel = mock(ServerChannel.class, AdditionalAnswers.delegatesTo(actual));
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    }

    @Override
    public ServerChannel start(ChannelInboundHandler handler) {
      reset(this.serverChannel);
      this.handler = handler;
      return this.serverChannel;
    }

    @Override
    public void stop(ServerChannel serverChannel) {
      // Note: In the tests, we expect that connecting to a closed server channel results
      // in a channel connection error. Netty doesn't seem to handle closing domain socket
      // addresses very well-- often connecting to a closed domain socket will result in a
      // read timeout instead of a connection timeout.
      //
      // This is a hack to ensure connection timeouts are "received" by the tests for this
      // dummy domain socket server. In particular, this lets the timeoutShouldWork_connect
      // test work for both inet and domain sockets.
      //
      // This is also part of the workaround for https://github.com/netty/netty/issues/7047.
      when(this.serverChannel.localAddress()).thenReturn(new DomainSocketAddress(""));
      this.handler = null;
    }
  }

  @Parameters
  public static List<Object[]> createInputValues() {
    ArrayList<Object[]> parameters =
        new ArrayList<Object[]>(Arrays.asList(new Object[][] {{new InetTestServer()}}));

    if (Epoll.isAvailable()) {
      parameters.add(
          new Object[] {
            new UnixDomainServer(EpollServerDomainSocketChannel.class, EpollEventLoopGroup::new)
          });
    }

    if (KQueue.isAvailable()) {
      parameters.add(
          new Object[] {
            new UnixDomainServer(KQueueServerDomainSocketChannel.class, KQueueEventLoopGroup::new)
          });
    }

    return parameters;
  }

  private final TestServer testServer;

  public HttpCacheClientTest(TestServer testServer) {
    this.testServer = testServer;
  }

  private HttpCacheClient createHttpBlobStore(
      ServerChannel serverChannel,
      int timeoutSeconds,
      boolean remoteVerifyDownloads,
      @Nullable final Credentials creds,
      AuthAndTLSOptions authAndTlsOptions,
      Optional<RemoteRetrier> optRetrier)
      throws Exception {
    SocketAddress socketAddress = serverChannel.localAddress();
    RemoteRetrier retrier =
        optRetrier.orElseGet(
            () -> {
              ListeningScheduledExecutorService retryScheduler =
                  MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
              return new RemoteRetrier(
                  () -> RemoteRetrier.RETRIES_DISABLED,
                  (e) -> Result.SUCCESS,
                  retryScheduler,
                  Retrier.ALLOW_ALL_CALLS);
            });
    if (socketAddress instanceof DomainSocketAddress domainSocketAddress) {
      URI uri = new URI("http://localhost");
      return HttpCacheClient.create(
          domainSocketAddress,
          uri,
          timeoutSeconds,
          /* remoteMaxConnections= */ 0,
          remoteVerifyDownloads,
          ImmutableList.of(),
          DIGEST_UTIL,
          retrier,
          creds,
          authAndTlsOptions);
    } else if (socketAddress instanceof InetSocketAddress inetSocketAddress) {
      URI uri = new URI("http://localhost:" + inetSocketAddress.getPort());
      return HttpCacheClient.create(
          uri,
          timeoutSeconds,
          /* remoteMaxConnections= */ 0,
          remoteVerifyDownloads,
          ImmutableList.of(),
          DIGEST_UTIL,
          retrier,
          creds,
          authAndTlsOptions);
    } else {
      throw new IllegalStateException(
          "unsupported socket address class " + socketAddress.getClass());
    }
  }

  private HttpCacheClient createHttpBlobStore(
      ServerChannel serverChannel,
      int timeoutSeconds,
      @Nullable final Credentials creds,
      AuthAndTLSOptions authAndTlsOptions)
      throws Exception {
    return createHttpBlobStore(
        serverChannel,
        timeoutSeconds,
        /* remoteVerifyDownloads= */ true,
        creds,
        authAndTlsOptions,
        Optional.empty());
  }

  @Before
  public void setUp() throws Exception {
    remoteActionExecutionContext =
        RemoteActionExecutionContext.create(
            mock(Spawn.class),
            mock(SpawnExecutionContext.class),
            TracingMetadataUtils.buildMetadata(
                "none", "none", Digest.getDefaultInstance().getHash(), null));
  }

  @Test
  public void testUpload() throws Exception {
    ServerChannel server = null;
    try {
      ConcurrentHashMap<String, byte[]> cacheContents = new ConcurrentHashMap<>();
      server = testServer.start(new InMemoryHttpCacheServerHandler(cacheContents));

      HttpCacheClient blobStore =
          createHttpBlobStore(
              server, /* timeoutSeconds= */ 1, /* creds= */ null, new AuthAndTLSOptions());

      ByteString data = ByteString.copyFrom("foo bar", StandardCharsets.UTF_8);
      Digest digest = DIGEST_UTIL.compute(data.toByteArray());
      blobStore.uploadBlob(remoteActionExecutionContext, digest, data).get();

      assertThat(cacheContents).hasSize(1);
      String cacheKey = "/cas/" + digest.getHash();
      assertThat(cacheContents).containsKey(cacheKey);
      assertThat(cacheContents.get(cacheKey)).isEqualTo(data.toByteArray());
    } finally {
      testServer.stop(server);
    }
  }

  @Test(timeout = 30000)
  public void connectTimeout() throws Exception {
    ServerChannel server = testServer.start(new ChannelInboundHandlerAdapter() {});
    testServer.stop(server);

    Credentials credentials = newCredentials();
    AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
    HttpCacheClient blobStore =
        createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
    assertThrows(
        ConnectException.class,
        () ->
            getFromFuture(
                blobStore.downloadBlob(
                    remoteActionExecutionContext, DIGEST, new ByteArrayOutputStream())));
  }

  @Test(timeout = 30000)
  public void uploadTimeout() throws Exception {
    ServerChannel server = null;
    try {
      server =
          testServer.start(
              new SimpleChannelInboundHandler<FullHttpRequest>() {
                @Override
                protected void channelRead0(
                    ChannelHandlerContext channelHandlerContext, FullHttpRequest fullHttpRequest) {
                  // Don't respond and force a client timeout.
                }
              });

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      byte[] data = "File Contents".getBytes(StandardCharsets.US_ASCII);
      assertThrows(
          UploadTimeoutException.class,
          () ->
              getFromFuture(
                  blobStore.uploadBlob(
                      remoteActionExecutionContext,
                      DIGEST_UTIL.compute(data),
                      ByteString.copyFrom(data))));
    } finally {
      testServer.stop(server);
    }
  }

  @Test(timeout = 30000)
  public void downloadTimeout() throws Exception {
    ServerChannel server = null;
    try {
      server =
          testServer.start(
              new SimpleChannelInboundHandler<FullHttpRequest>() {
                @Override
                protected void channelRead0(
                    ChannelHandlerContext channelHandlerContext, FullHttpRequest fullHttpRequest) {
                  // Don't respond and force a client timeout.
                }
              });

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      assertThrows(
          DownloadTimeoutException.class,
          () ->
              getFromFuture(
                  blobStore.downloadBlob(
                      remoteActionExecutionContext, DIGEST, new ByteArrayOutputStream())));
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void uploadResponseTooLarge() throws Exception {
    ServerChannel server = null;
    try {
      server =
          testServer.start(
              new SimpleChannelInboundHandler<FullHttpRequest>() {
                @Override
                protected void channelRead0(
                    ChannelHandlerContext channelHandlerContext, FullHttpRequest request) {
                  ByteBuf longMessage =
                      channelHandlerContext.alloc().buffer(50000).writerIndex(50000);
                  DefaultFullHttpResponse response =
                      new DefaultFullHttpResponse(
                          HttpVersion.HTTP_1_1,
                          HttpResponseStatus.INTERNAL_SERVER_ERROR,
                          longMessage);
                  channelHandlerContext
                      .writeAndFlush(response)
                      .addListener(ChannelFutureListener.CLOSE);
                }
              });

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      ByteString data = ByteString.copyFrom("File Contents", StandardCharsets.US_ASCII);
      IOException e =
          assertThrows(
              IOException.class,
              () ->
                  getFromFuture(
                      blobStore.uploadBlob(
                          remoteActionExecutionContext,
                          DIGEST_UTIL.compute(data.toByteArray()),
                          data)));
      assertThat(e.getCause()).isInstanceOf(TooLongFrameException.class);
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void testDownloadFailsOnDigestMismatch() throws Exception {
    // Test that the download fails when a blob/file has a different content hash than expected.

    ServerChannel server = null;
    try {
      server =
          testServer.start(
              new SimpleChannelInboundHandler<FullHttpRequest>() {
                @Override
                protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
                  ByteBuf data = ctx.alloc().buffer();
                  ByteBufUtil.writeUtf8(data, "bar");
                  DefaultFullHttpResponse response =
                      new DefaultFullHttpResponse(
                          HttpVersion.HTTP_1_1, HttpResponseStatus.OK, data);
                  HttpUtil.setContentLength(response, data.readableBytes());

                  ctx.writeAndFlush(response).addListener(ChannelFutureListener.CLOSE);
                }
              });

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(
              server,
              /* timeoutSeconds= */ 1,
              /* remoteVerifyDownloads= */ true,
              credentials,
              authAndTlsOptions,
              Optional.empty());
      Digest fooDigest = DIGEST_UTIL.compute("foo".getBytes(StandardCharsets.UTF_8));
      try (OutputStream out = new ByteArrayOutputStream()) {
        IOException e =
            assertThrows(
                IOException.class,
                () ->
                    getFromFuture(
                        blobStore.downloadBlob(remoteActionExecutionContext, fooDigest, out)));
        assertThat(e).hasMessageThat().contains(fooDigest.getHash());
        assertThat(e).hasMessageThat().contains(DIGEST_UTIL.computeAsUtf8("bar").getHash());
      }
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void testDisablingDigestVerification() throws Exception {
    // Test that when digest verification is disabled a corrupted download works.

    ServerChannel server = null;
    try {
      server =
          testServer.start(
              new SimpleChannelInboundHandler<FullHttpRequest>() {
                @Override
                protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
                  ByteBuf data = ctx.alloc().buffer();
                  ByteBufUtil.writeUtf8(data, "bar");
                  DefaultFullHttpResponse response =
                      new DefaultFullHttpResponse(
                          HttpVersion.HTTP_1_1, HttpResponseStatus.OK, data);
                  HttpUtil.setContentLength(response, data.readableBytes());

                  ctx.writeAndFlush(response).addListener(ChannelFutureListener.CLOSE);
                }
              });

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(
              server,
              /* timeoutSeconds= */ 1,
              /* remoteVerifyDownloads= */ false,
              credentials,
              authAndTlsOptions,
              Optional.empty());
      Digest fooDigest = DIGEST_UTIL.compute("foo".getBytes(StandardCharsets.UTF_8));
      try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
        getFromFuture(blobStore.downloadBlob(remoteActionExecutionContext, fooDigest, out));
        assertThat(out.toByteArray()).isEqualTo("bar".getBytes(StandardCharsets.UTF_8));
      }
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void partialDownloadFailsWithoutRetry() throws Exception {
    ServerChannel server = null;
    try {
      ByteBuf chunk1 = Unpooled.wrappedBuffer("File ".getBytes(StandardCharsets.US_ASCII));
      ByteBuf chunk2 = Unpooled.wrappedBuffer("Contents".getBytes(StandardCharsets.US_ASCII));
      server = testServer.start(new IntermittentFailureHandler(chunk1, chunk2));
      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);

      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      assertThrows(
          ClosedChannelException.class,
          () ->
              getFromFuture(
                  blobStore.downloadBlob(
                      remoteActionExecutionContext, DIGEST, new ByteArrayOutputStream())));
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void partialDownloadSucceedsWithRetry() throws Exception {
    ServerChannel server = null;
    try {
      ByteBuf chunk1 = Unpooled.wrappedBuffer("File ".getBytes(StandardCharsets.US_ASCII));
      // Replace first chunk to test that the client skips the redundant prefix on retry.
      ByteBuf chunk1Attempt2 = Unpooled.wrappedBuffer("abcde".getBytes(StandardCharsets.US_ASCII));
      ByteBuf chunk2 = Unpooled.wrappedBuffer("Contents".getBytes(StandardCharsets.US_ASCII));
      server = testServer.start(new IntermittentFailureHandler(chunk1, chunk1Attempt2, chunk2));
      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);

      ListeningScheduledExecutorService retryScheduler =
          MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
      RemoteRetrier retrier =
          new RemoteRetrier(
              () -> new Retrier.ZeroBackoff(1),
              (e) -> {
                return e instanceof ClosedChannelException
                    ? Result.TRANSIENT_FAILURE
                    : Result.PERMANENT_FAILURE;
              },
              retryScheduler,
              Retrier.ALLOW_ALL_CALLS);
      HttpCacheClient blobStore =
          createHttpBlobStore(
              server,
              /* timeoutSeconds= */ 1,
              /* remoteVerifyDownloads= */ false,
              credentials,
              authAndTlsOptions,
              Optional.of(retrier));

      ByteArrayOutputStream download = new ByteArrayOutputStream();
      getFromFuture(blobStore.downloadBlob(remoteActionExecutionContext, DIGEST, download));
      assertThat(download.toByteArray())
          .isEqualTo("File Contents".getBytes(StandardCharsets.US_ASCII));
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void actionResultRetryReadsFromStart() throws Exception {
    ServerChannel server = null;
    try {
      ActionResult.Builder builder1 = ActionResult.newBuilder();
      builder1
          .addOutputFilesBuilder()
          .setPath("attempt1/filename")
          .setDigest(DIGEST_UTIL.computeAsUtf8("digest1"))
          .setIsExecutable(true);
      ActionResult action1 = builder1.build();
      ByteArrayOutputStream buffer1 = new ByteArrayOutputStream();
      action1.writeTo(buffer1);
      int splitAt = buffer1.size() / 2;
      ByteBuf chunk1 = Unpooled.copiedBuffer(buffer1.toByteArray(), 0, splitAt);

      // Replace first chunk to test that the client starts a fresh ActionResult download on retry.
      ActionResult.Builder builder2 = ActionResult.newBuilder();
      builder2
          .addOutputFilesBuilder()
          .setPath("attempt2/filename")
          .setDigest(DIGEST_UTIL.computeAsUtf8("digest2"))
          .setIsExecutable(false);
      ActionResult action2 = builder2.build();
      ByteArrayOutputStream buffer2 = new ByteArrayOutputStream();
      action2.writeTo(buffer2);
      ByteBuf chunk1Attempt2 = Unpooled.copiedBuffer(buffer2.toByteArray(), 0, splitAt);
      ByteBuf chunk2 =
          Unpooled.copiedBuffer(buffer2.toByteArray(), splitAt, buffer2.size() - splitAt);

      server = testServer.start(new IntermittentFailureHandler(chunk1, chunk1Attempt2, chunk2));
      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);

      ListeningScheduledExecutorService retryScheduler =
          MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
      RemoteRetrier retrier =
          new RemoteRetrier(
              () -> new Retrier.ZeroBackoff(1),
              (e) -> {
                return e instanceof ClosedChannelException
                    ? Result.TRANSIENT_FAILURE
                    : Result.PERMANENT_FAILURE;
              },
              retryScheduler,
              Retrier.ALLOW_ALL_CALLS);
      HttpCacheClient blobStore =
          createHttpBlobStore(
              server,
              /* timeoutSeconds= */ 1,
              /* remoteVerifyDownloads= */ false,
              credentials,
              authAndTlsOptions,
              Optional.of(retrier));

      var actionResult =
          getFromFuture(
              blobStore.downloadActionResult(
                  remoteActionExecutionContext,
                  new RemoteCacheClient.ActionKey(DIGEST),
                  /* inlineOutErr= */ false,
                  /* inlineOutputFiles= */ ImmutableSet.of()));
      assertThat(actionResult).isEqualTo(action2);
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void expiredAuthTokensShouldBeRetried_get() throws Exception {
    expiredAuthTokensShouldBeRetried_get(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.UNAUTHORIZED);
    expiredAuthTokensShouldBeRetried_get(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.INVALID_TOKEN);
  }

  private void expiredAuthTokensShouldBeRetried_get(
      HttpCacheClientTest.NotAuthorizedHandler.ErrorType errorType) throws Exception {
    ServerChannel server = null;
    try {
      server = testServer.start(new NotAuthorizedHandler(errorType));

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      ByteArrayOutputStream out = Mockito.spy(new ByteArrayOutputStream());
      getFromFuture(blobStore.downloadBlob(remoteActionExecutionContext, DIGEST, out));
      assertThat(out.toString(StandardCharsets.US_ASCII.name())).isEqualTo("File Contents");
      verify(credentials, times(1)).refresh();
      verify(credentials, times(2)).getRequestMetadata(any(URI.class));
      verify(credentials, times(2)).hasRequestMetadata();
      // The caller is responsible to the close the stream.
      verify(out, never()).close();
      verifyNoMoreInteractions(credentials);
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void expiredAuthTokensShouldBeRetried_put() throws Exception {
    expiredAuthTokensShouldBeRetried_put(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.UNAUTHORIZED);
    expiredAuthTokensShouldBeRetried_put(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.INVALID_TOKEN);
  }

  private void expiredAuthTokensShouldBeRetried_put(
      HttpCacheClientTest.NotAuthorizedHandler.ErrorType errorType) throws Exception {
    ServerChannel server = null;
    try {
      server = testServer.start(new NotAuthorizedHandler(errorType));

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      byte[] data = "File Contents".getBytes(StandardCharsets.US_ASCII);
      blobStore
          .uploadBlob(
              remoteActionExecutionContext, DIGEST_UTIL.compute(data), ByteString.copyFrom(data))
          .get();
      verify(credentials, times(1)).refresh();
      verify(credentials, times(2)).getRequestMetadata(any(URI.class));
      verify(credentials, times(2)).hasRequestMetadata();
      verifyNoMoreInteractions(credentials);
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void errorCodesThatShouldNotBeRetried_get() {
    errorCodeThatShouldNotBeRetried_get(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.INSUFFICIENT_SCOPE);
    errorCodeThatShouldNotBeRetried_get(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.INVALID_REQUEST);
  }

  private void errorCodeThatShouldNotBeRetried_get(
      HttpCacheClientTest.NotAuthorizedHandler.ErrorType errorType) {
    ServerChannel server = null;
    try {
      server = testServer.start(new NotAuthorizedHandler(errorType));

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      getFromFuture(
          blobStore.downloadBlob(
              remoteActionExecutionContext, DIGEST, new ByteArrayOutputStream()));
      fail("Exception expected.");
    } catch (Exception e) {
      assertThat(e).isInstanceOf(HttpException.class);
      assertThat(((HttpException) e).response().status())
          .isEqualTo(HttpResponseStatus.UNAUTHORIZED);
    } finally {
      testServer.stop(server);
    }
  }

  @Test
  public void errorCodesThatShouldNotBeRetried_put() {
    errorCodeThatShouldNotBeRetried_put(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.INSUFFICIENT_SCOPE);
    errorCodeThatShouldNotBeRetried_put(
        HttpCacheClientTest.NotAuthorizedHandler.ErrorType.INVALID_REQUEST);
  }

  private void errorCodeThatShouldNotBeRetried_put(
      HttpCacheClientTest.NotAuthorizedHandler.ErrorType errorType) {
    ServerChannel server = null;
    try {
      server = testServer.start(new NotAuthorizedHandler(errorType));

      Credentials credentials = newCredentials();
      AuthAndTLSOptions authAndTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
      HttpCacheClient blobStore =
          createHttpBlobStore(server, /* timeoutSeconds= */ 1, credentials, authAndTlsOptions);
      byte[] oneByte = new byte[] {0};
      getFromFuture(
          blobStore.uploadBlob(
              remoteActionExecutionContext,
              DIGEST_UTIL.compute(oneByte),
              ByteString.copyFrom(oneByte)));
      fail("Exception expected.");
    } catch (Exception e) {
      assertThat(e).isInstanceOf(HttpException.class);
      assertThat(((HttpException) e).response().status())
          .isEqualTo(HttpResponseStatus.UNAUTHORIZED);
    } finally {
      testServer.stop(server);
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
        content.writeCharSequence("File Contents", StandardCharsets.US_ASCII);
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

  /**
   * {@link ChannelHandler} that on the first request returns a partial response and then closes the
   * stream, and on any further requests returns a full response.
   */
  @Sharable
  static class IntermittentFailureHandler extends SimpleChannelInboundHandler<FullHttpRequest> {
    private final ByteBuf attempt1Chunk1;
    private final ByteBuf attempt2Chunk1;
    private final ByteBuf attempt2Chunk2;
    private int messageCount;

    public IntermittentFailureHandler(
        ByteBuf attempt1Chunk1, ByteBuf attempt2Chunk1, ByteBuf attempt2Chunk2) {
      this.attempt1Chunk1 = attempt1Chunk1;
      this.attempt2Chunk1 = attempt2Chunk1;
      this.attempt2Chunk2 = attempt2Chunk2;
    }

    public IntermittentFailureHandler(ByteBuf chunk1, ByteBuf chunk2) {
      this(chunk1.copy(), chunk1, chunk2);
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
      DefaultHttpResponse response =
          new DefaultHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK);
      response.headers().set(HttpHeaderNames.TRANSFER_ENCODING, HttpHeaderValues.CHUNKED);
      ctx.write(response);
      if (messageCount == 0) {
        ctx.writeAndFlush(new DefaultHttpContent(attempt1Chunk1))
            .addListener(ChannelFutureListener.CLOSE);
      } else {
        ctx.writeAndFlush(new DefaultHttpContent(attempt2Chunk1));
        ctx.writeAndFlush(new DefaultLastHttpContent(attempt2Chunk2))
            .addListener(ChannelFutureListener.CLOSE);
      }
      ++messageCount;
    }
  }
}
