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
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelOutboundHandler;
import io.netty.channel.ChannelPromise;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpRequest;
import java.io.IOException;
import java.net.SocketAddress;
import java.net.URI;
import java.util.List;
import java.util.Map;

abstract class AbstractHttpHandler<T extends HttpObject> extends SimpleChannelInboundHandler<T>
    implements ChannelOutboundHandler {

  private final Credentials credentials;

  public AbstractHttpHandler(Credentials credentials) {
    this.credentials = credentials;
  }

  protected ChannelPromise userPromise;

  protected void failAndResetUserPromise(Throwable t) {
    if (userPromise != null && !userPromise.isDone()) {
      userPromise.setFailure(t);
    }
    userPromise = null;
  }

  protected void succeedAndResetUserPromise() {
    userPromise.setSuccess();
    userPromise = null;
  }

  protected void addCredentialHeaders(HttpRequest request, URI uri) throws IOException {
    if (credentials == null || !credentials.hasRequestMetadata()) {
      return;
    }
    Map<String, List<String>> authHeaders = credentials.getRequestMetadata(uri);
    if (authHeaders == null || authHeaders.isEmpty()) {
      return;
    }
    for (Map.Entry<String, List<String>> entry : authHeaders.entrySet()) {
      String name = entry.getKey();
      for (String value : entry.getValue()) {
        request.headers().add(name, value);
      }
    }
  }

  protected String constructPath(URI uri, String hash, boolean isCas) {
    StringBuilder builder = new StringBuilder();
    builder.append(uri.getPath());
    if (!uri.getPath().endsWith("/")) {
      builder.append("/");
    }
    if (isCas) {
      builder.append("cas/");
    } else {
      builder.append("ac/");
    }
    builder.append(hash);
    return builder.toString();
  }

  protected String constructHost(URI uri) {
    return uri.getHost() + ":" + uri.getPort();
  }

  @Override
  public void exceptionCaught(ChannelHandlerContext channelHandlerContext, Throwable throwable)
      throws Exception {
    failAndResetUserPromise(throwable);
  }

  @Override
  public void bind(ChannelHandlerContext ctx, SocketAddress localAddress, ChannelPromise promise)
      throws Exception {
    ctx.bind(localAddress, promise);
  }

  @Override
  public void connect(
      ChannelHandlerContext ctx,
      SocketAddress remoteAddress,
      SocketAddress localAddress,
      ChannelPromise promise)
      throws Exception {
    ctx.connect(remoteAddress, localAddress, promise);
  }

  @Override
  public void disconnect(ChannelHandlerContext ctx, ChannelPromise promise) throws Exception {
    ctx.disconnect(promise);
  }

  @Override
  public void close(ChannelHandlerContext ctx, ChannelPromise promise) throws Exception {
    ctx.close(promise);
  }

  @Override
  public void deregister(ChannelHandlerContext ctx, ChannelPromise promise) throws Exception {
    ctx.deregister(promise);
  }

  @Override
  public void read(ChannelHandlerContext ctx) throws Exception {
    ctx.read();
  }

  @Override
  public void flush(ChannelHandlerContext ctx) throws Exception {
    ctx.flush();
  }
}
