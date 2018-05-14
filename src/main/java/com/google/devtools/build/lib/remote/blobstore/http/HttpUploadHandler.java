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
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPromise;
import io.netty.handler.codec.http.DefaultHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpChunkedInput;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.stream.ChunkedStream;
import io.netty.util.internal.StringUtil;
import java.io.IOException;

/** ChannelHandler for uploads. */
final class HttpUploadHandler extends AbstractHttpHandler<FullHttpResponse> {

  public HttpUploadHandler(Credentials credentials) {
    super(credentials);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  protected void channelRead0(ChannelHandlerContext ctx, FullHttpResponse response) {
    if (!response.decoderResult().isSuccess()) {
      failAndClose(new IOException("Failed to parse the HTTP response."), ctx);
      return;
    }
    try {
      checkState(userPromise != null, "response before request");
      if (!response.status().equals(HttpResponseStatus.OK)
          && !response.status().equals(HttpResponseStatus.ACCEPTED)
          && !response.status().equals(HttpResponseStatus.CREATED)
          && !response.status().equals(HttpResponseStatus.NO_CONTENT)) {
        // Supporting more than OK status to be compatible with nginx webdav.
        String errorMsg = response.status().toString();
        if (response.content().readableBytes() > 0) {
          byte[] data = new byte[response.content().readableBytes()];
          response.content().readBytes(data);
          errorMsg += "\n" + new String(data, HttpUtil.getCharset(response));
        }
        failAndResetUserPromise(new HttpException(response, errorMsg, null));
      } else {
        succeedAndResetUserPromise();
      }
    } finally {
      if (!HttpUtil.isKeepAlive(response)) {
        ctx.close();
      }
    }
  }

  @Override
  public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise)
      throws Exception {
    checkState(userPromise == null, "handler can't be shared between pipelines.");
    userPromise = promise;
    if (!(msg instanceof UploadCommand)) {
      failAndResetUserPromise(
          new IllegalArgumentException(
              "Unsupported message type: " + StringUtil.simpleClassName(msg)));
      return;
    }
    HttpRequest request = buildRequest((UploadCommand) msg);
    addCredentialHeaders(request, ((UploadCommand) msg).uri());
    HttpChunkedInput body = buildBody((UploadCommand) msg);
    ctx.writeAndFlush(request)
        .addListener(
            (f) -> {
              if (f.isSuccess()) {
                return;
              }
              failAndClose(f.cause(), ctx);
            });
    ctx.writeAndFlush(body)
        .addListener(
            (f) -> {
              if (f.isSuccess()) {
                return;
              }
              failAndClose(f.cause(), ctx);
            });
  }

  private HttpRequest buildRequest(UploadCommand msg) {
    HttpRequest request =
        new DefaultHttpRequest(
            HttpVersion.HTTP_1_1,
            HttpMethod.PUT,
            constructPath(msg.uri(), msg.hash(), msg.casUpload()));
    request.headers().set(HttpHeaderNames.HOST, constructHost(msg.uri()));
    request.headers().set(HttpHeaderNames.ACCEPT, "*/*");
    request.headers().set(HttpHeaderNames.CONTENT_LENGTH, msg.contentLength());
    request.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
    return request;
  }

  private HttpChunkedInput buildBody(UploadCommand msg) {
    return new HttpChunkedInput(new ChunkedStream(msg.data()));
  }


  @SuppressWarnings("FutureReturnValueIgnored")
  private void failAndClose(Throwable t, ChannelHandlerContext ctx) {
    try {
      failAndResetUserPromise(t);
    } finally {
      ctx.close();
    }
  }
}
