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

import static com.google.common.base.Preconditions.checkState;

import com.google.auth.Credentials;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPromise;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.HttpContent;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.LastHttpContent;
import io.netty.util.internal.StringUtil;
import java.io.IOException;
import java.io.OutputStream;

/** ChannelHandler for downloads. */
final class HttpDownloadHandler extends AbstractHttpHandler<HttpObject> {

  private long contentLength = -1;
  private long bytesReceived;
  private OutputStream out;
  private boolean keepAlive = HttpVersion.HTTP_1_1.isKeepAliveDefault();

  public HttpDownloadHandler(Credentials credentials) {
    super(credentials);
  }

  @Override
  protected void channelRead0(ChannelHandlerContext ctx, HttpObject msg) throws Exception {
    if (!msg.decoderResult().isSuccess()) {
      failAndResetUserPromise(new IOException("Failed to parse the HTTP response."));
      return;
    }
    checkState(userPromise != null, "response before request");
    if (msg instanceof HttpResponse) {
      HttpResponse response = (HttpResponse) msg;
      keepAlive = HttpUtil.isKeepAlive((HttpResponse) msg);
      if (HttpUtil.isContentLengthSet(response)) {
        contentLength = HttpUtil.getContentLength(response);
      }
      if (!response.status().equals(HttpResponseStatus.OK)) {
        failAndReset(
            new HttpException(response, "Download failed with status: " + response.status(), null),
            ctx);
      }
    } else if (msg instanceof HttpContent) {
      ByteBuf content = ((HttpContent) msg).content();
      bytesReceived += content.readableBytes();
      content.readBytes(out, content.readableBytes());
      if (bytesReceived == contentLength || msg instanceof LastHttpContent) {
        succeedAndReset(ctx);
      }
    } else {
      failAndReset(
          new IllegalArgumentException(
              "Unsupported message type: " + StringUtil.simpleClassName(msg)),
          ctx);
    }
  }

  @Override
  public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise)
      throws Exception {
    checkState(userPromise == null, "handler can't be shared between pipelines.");
    userPromise = promise;
    if (!(msg instanceof DownloadCommand)) {
      failAndResetUserPromise(
          new IllegalArgumentException(
              "Unsupported message type: " + StringUtil.simpleClassName(msg)));
      return;
    }
    out = ((DownloadCommand) msg).out();
    HttpRequest request = buildRequest((DownloadCommand) msg);
    addCredentialHeaders(request, ((DownloadCommand) msg).uri());
    ctx.writeAndFlush(request)
        .addListener(
            (f) -> {
              if (!f.isSuccess()) {
                failAndReset(f.cause(), ctx);
              }
            });
  }

  private HttpRequest buildRequest(DownloadCommand request) {
    HttpRequest httpRequest =
        new DefaultFullHttpRequest(
            HttpVersion.HTTP_1_1,
            HttpMethod.GET,
            constructPath(request.uri(), request.hash(), request.casDownload()));
    httpRequest.headers().set(HttpHeaderNames.HOST, constructHost(request.uri()));
    httpRequest.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
    httpRequest.headers().set(HttpHeaderNames.ACCEPT, "*/*");
    return httpRequest;
  }

  private void succeedAndReset(ChannelHandlerContext ctx) throws IOException {
    try {
      succeedAndResetUserPromise();
    } finally {
      reset(ctx);
    }
  }

  private void failAndReset(Throwable t, ChannelHandlerContext ctx) throws IOException {
    try {
      failAndResetUserPromise(t);
    } finally {
      reset(ctx);
    }
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  private void reset(ChannelHandlerContext ctx) throws IOException {
    try {
      if (!keepAlive) {
        ctx.close();
      }
    } finally {
      contentLength = -1;
      bytesReceived = 0;
      out = null;
      keepAlive = HttpVersion.HTTP_1_1.isKeepAliveDefault();
    }
  }
}
