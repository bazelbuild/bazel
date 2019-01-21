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

import com.google.common.base.Charsets;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelOutboundHandler;
import io.netty.channel.ChannelPromise;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpRequest;
import java.io.IOException;
import java.net.SocketAddress;
import java.net.URI;
import java.nio.channels.ClosedChannelException;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Common functionality shared by concrete classes. */
abstract class AbstractHttpHandler<T extends HttpObject> extends SimpleChannelInboundHandler<T>
    implements ChannelOutboundHandler {

  @Nullable
  private final AuthHeadersProvider authHeadersProvider;

  public AbstractHttpHandler(AuthHeadersProvider authHeadersProvider) {
    this.authHeadersProvider = authHeadersProvider;
  }

  protected ChannelPromise userPromise;

  @SuppressWarnings("FutureReturnValueIgnored")
  protected void failAndResetUserPromise(Throwable t) {
    if (userPromise != null && !userPromise.isDone()) {
      userPromise.setFailure(t);
    }
    userPromise = null;
  }

  @SuppressWarnings("FutureReturnValueIgnored") 
  protected void succeedAndResetUserPromise() {  
    userPromise.setSuccess();
    userPromise = null;
  }

  protected void addCredentialHeaders(HttpRequest request, URI uri) throws IOException {
    // TODO(buchgr): implement basic auth as a AuthHeadersProvider
    String userInfo = uri.getUserInfo();
    if (userInfo != null) {
      String value = BaseEncoding.base64Url().encode(userInfo.getBytes(Charsets.UTF_8));
      request.headers().set(HttpHeaderNames.AUTHORIZATION, "Basic " + value);
      return;
    }
    if (authHeadersProvider == null) {
      return;
    }
    Map<String, List<String>> authHeaders = authHeadersProvider.getRequestHeaders(uri);
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
    builder.append(isCas ? "cas/" : "ac/");
    builder.append(hash);
    return builder.toString();
  }

  protected String constructHost(URI uri) {
    boolean includePort =
        (uri.getPort() > 0)
            && ((uri.getScheme().equals("http") && uri.getPort() != 80)
                || (uri.getScheme().equals("https") && uri.getPort() != 443));
    return uri.getHost() + (includePort ? ":" + uri.getPort() : "");
  }

  @Override
  public void exceptionCaught(ChannelHandlerContext ctx, Throwable t) {
    failAndResetUserPromise(t);
    ctx.fireExceptionCaught(t);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void bind(ChannelHandlerContext ctx, SocketAddress localAddress, ChannelPromise promise) {
    ctx.bind(localAddress, promise);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void connect(
      ChannelHandlerContext ctx,
      SocketAddress remoteAddress,
      SocketAddress localAddress,
      ChannelPromise promise) {
    ctx.connect(remoteAddress, localAddress, promise);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void disconnect(ChannelHandlerContext ctx, ChannelPromise promise) {
    failAndResetUserPromise(new ClosedChannelException());
    ctx.disconnect(promise);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void close(ChannelHandlerContext ctx, ChannelPromise promise) {
    failAndResetUserPromise(new ClosedChannelException());
    ctx.close(promise);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void deregister(ChannelHandlerContext ctx, ChannelPromise promise) {
    failAndResetUserPromise(new ClosedChannelException());
    ctx.deregister(promise);
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void read(ChannelHandlerContext ctx) {
    ctx.read();
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  @Override
  public void flush(ChannelHandlerContext ctx) {
    ctx.flush();
  }

  @Override
  public void channelInactive(ChannelHandlerContext ctx) {
    failAndResetUserPromise(new ClosedChannelException());
    ctx.fireChannelInactive();
  }

  @Override
  public void handlerRemoved(ChannelHandlerContext ctx) {
    failAndResetUserPromise(new IOException("handler removed"));
  }

  @Override
  public void channelUnregistered(ChannelHandlerContext ctx) {
    failAndResetUserPromise(new ClosedChannelException());
    ctx.fireChannelUnregistered();
  }
}
