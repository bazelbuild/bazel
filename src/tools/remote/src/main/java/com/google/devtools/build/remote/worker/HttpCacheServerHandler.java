package com.google.devtools.build.remote.worker;

import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.util.CharsetUtil;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class HttpCacheServerHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

  final ConcurrentMap<String, byte[]> cache = new ConcurrentHashMap<>();

  @Override
  protected void channelRead0(ChannelHandlerContext ctx,
      FullHttpRequest request) {
    if (!request.decoderResult().isSuccess()) {
      sendError(ctx, request, HttpResponseStatus.BAD_REQUEST);
      return;
    }

    if (request.method() == HttpMethod.GET) {
      handleGet(ctx, request);
    } else if (request.method() == HttpMethod.PUT) {
      handlePut(ctx, request);
    } else {
      sendError(ctx, request, HttpResponseStatus.METHOD_NOT_ALLOWED);
    }
  }

  private void handleGet(ChannelHandlerContext ctx, FullHttpRequest request) {
    byte[] contents = cache.get(request.uri());

    if (contents == null) {
      sendError(ctx, request, HttpResponseStatus.NOT_FOUND);
      return;
    }

    FullHttpResponse response = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1,
        HttpResponseStatus.OK, Unpooled.wrappedBuffer(contents));
    HttpUtil.setContentLength(response, contents.length);
    response.headers().set(HttpHeaderNames.CONTENT_TYPE, "application/octet-stream");
    ChannelFuture lastContentFuture = ctx.writeAndFlush(response);

    if (!HttpUtil.isKeepAlive(request)) {
      lastContentFuture.addListener(ChannelFutureListener.CLOSE);
    }
  }

  private void handlePut(ChannelHandlerContext ctx, FullHttpRequest request) {
    if (!request.decoderResult().isSuccess()) {
      sendError(ctx, request, HttpResponseStatus.INTERNAL_SERVER_ERROR);
      return;
    }

    byte[] contentBytes = new byte[request.content().readableBytes()];
    request.content().readBytes(contentBytes);
    cache.putIfAbsent(request.uri(), contentBytes);

    FullHttpResponse response = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1,
        HttpResponseStatus.NO_CONTENT);
    ChannelFuture lastContentFuture = ctx.writeAndFlush(response);

    if (!HttpUtil.isKeepAlive(request)) {
      lastContentFuture.addListener(ChannelFutureListener.CLOSE);
    }
  }

  private static void sendError(ChannelHandlerContext ctx,
      FullHttpRequest request, HttpResponseStatus status) {
    FullHttpResponse response = new DefaultFullHttpResponse(
        HttpVersion.HTTP_1_1, status,
        Unpooled.copiedBuffer("Failure: " + status + "\r\n", CharsetUtil.UTF_8));
    response.headers().set(HttpHeaderNames.CONTENT_TYPE, "text/plain; charset=UTF-8");
    ChannelFuture future = ctx.writeAndFlush(response);

    if (!HttpUtil.isKeepAlive(request)) {
      future.addListener(ChannelFutureListener.CLOSE);
    }
  }
}
