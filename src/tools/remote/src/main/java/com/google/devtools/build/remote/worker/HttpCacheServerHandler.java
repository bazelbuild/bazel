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

package com.google.devtools.build.remote.worker;

import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler.Sharable;
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

/** A simple HTTP REST in-memory cache used during testing the LRE. */
@Sharable
public class HttpCacheServerHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

  final ConcurrentMap<String, byte[]> cache = new ConcurrentHashMap<>();

  @Override
  protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
    if (!request.decoderResult().isSuccess()) {
      sendError(ctx, request, HttpResponseStatus.BAD_REQUEST);
      return;
    }

    if (request.method().equals(HttpMethod.GET)) {
      handleGet(ctx, request);
    } else if (request.method().equals(HttpMethod.PUT)) {
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

    FullHttpResponse response =
        new DefaultFullHttpResponse(
            HttpVersion.HTTP_1_1, HttpResponseStatus.OK, Unpooled.wrappedBuffer(contents));
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

    FullHttpResponse response =
        new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.NO_CONTENT);
    ChannelFuture lastContentFuture = ctx.writeAndFlush(response);

    if (!HttpUtil.isKeepAlive(request)) {
      lastContentFuture.addListener(ChannelFutureListener.CLOSE);
    }
  }

  private static void sendError(
      ChannelHandlerContext ctx, FullHttpRequest request, HttpResponseStatus status) {
    FullHttpResponse response =
        new DefaultFullHttpResponse(
            HttpVersion.HTTP_1_1,
            status,
            Unpooled.copiedBuffer("Failure: " + status + "\r\n", CharsetUtil.UTF_8));
    response.headers().set(HttpHeaderNames.CONTENT_TYPE, "text/plain; charset=UTF-8");
    ChannelFuture future = ctx.writeAndFlush(response);

    if (!HttpUtil.isKeepAlive(request)) {
      future.addListener(ChannelFutureListener.CLOSE);
    }
  }
}
