// Copyright 2026 The Bazel Authors. All rights reserved.
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

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelPipeline;
import io.netty.handler.proxy.HttpProxyHandler;
import io.netty.handler.proxy.ProxyHandler;
import io.netty.resolver.NoopAddressResolverGroup;
import io.netty.util.concurrent.Future;
import io.netty.util.concurrent.Promise;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.URI;
import javax.annotation.Nullable;

/**
 * Tunnels {@link HttpCacheClient} connections through an HTTP CONNECT proxy, if configured.
 */
final class HttpConnectProxy {

  private static final String SSL_HANDLER = "ssl-handler";

  private final InetSocketAddress proxyAddress;

  private HttpConnectProxy(InetSocketAddress proxyAddress) {
    this.proxyAddress = proxyAddress;
  }

  /** Returns a proxy for {@code proxyAddress}, or {@code null} when no proxy is configured. */
  @Nullable
  static HttpConnectProxy create(@Nullable InetSocketAddress proxyAddress) {
    return proxyAddress == null ? null : new HttpConnectProxy(proxyAddress);
  }

  /** The proxy connects to the server, so leave the host unresolved and let the proxy resolve it. */
  SocketAddress remoteAddress(URI uri) {
    return InetSocketAddress.createUnresolved(uri.getHost(), uri.getPort());
  }

  void configureBootstrap(Bootstrap bootstrap) {
    // DNS is resolved on the proxy-side rather than client side.
    bootstrap.resolver(NoopAddressResolverGroup.INSTANCE);
  }

  void addProxyHandler(ChannelPipeline pipeline) {
    pipeline.addLast("http-proxy-handler", new HttpProxyHandler(proxyAddress));
  }

  /**
   * Runs {@code onReady} once any HTTP CONNECT proxy handshake on the channel has completed. The
   * proxy handler installs transient codec handlers while the tunnel is being established, so
   * per-request handlers must not be added until then. Once the handshake succeeds the proxy
   * handlers have done their job and are removed so the pipeline is clean before {@code onReady}
   * adds the request handlers. If no proxy is configured (or it was already removed on a reused
   * channel), {@code onReady} runs immediately.
   */
  @SuppressWarnings("FutureReturnValueIgnored")
  static void awaitHandshake(Channel channel, Promise<Channel> channelReady, Runnable onReady) {
    ProxyHandler proxyHandler = channel.pipeline().get(ProxyHandler.class);
    if (proxyHandler == null) {
      onReady.run();
      return;
    }
    proxyHandler
        .connectFuture()
        .addListener(
            (Future<Channel> handshake) -> {
              if (!handshake.isSuccess()) {
                channelReady.setFailure(handshake.cause());
                return;
              }
              // The listener fires from within the proxy handler's own channelRead, so defer the
              // cleanup and handler setup to the event loop to avoid reentrant pipeline mutation.
              channel
                  .eventLoop()
                  .execute(
                      () -> {
                        removeProxyHandlers(channel.pipeline());
                        onReady.run();
                      });
            });
  }

  /**
   * Strips the handlers an HTTP CONNECT proxy leaves behind once the tunnel is established: the
   * proxy handler itself and the now-inert codec wrapper that Netty does not fully detach. These
   * sit ahead of the persistent {@code ssl-handler}, so removing from the front until that handler
   * (or an empty pipeline) is reached leaves the pipeline ready for per-request handlers.
   */
  private static void removeProxyHandlers(ChannelPipeline pipeline) {
    while (pipeline.first() != null && !SSL_HANDLER.equals(pipeline.firstContext().name())) {
      pipeline.removeFirst();
    }
  }
}
