// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.remote.grpc.ChannelConnectionFactory;
import io.grpc.ClientInterceptor;
import io.reactivex.rxjava3.core.Single;
import java.util.List;

/**
 * A {@link ChannelConnectionFactory} which creates {@link ChannelConnection} using {@link
 * ChannelFactory}.
 */
public class GoogleChannelConnectionFactory implements ChannelConnectionFactory {
  private final ChannelFactory channelFactory;
  private final String target;
  private final String proxy;
  private final AuthAndTLSOptions options;
  private final List<ClientInterceptor> interceptors;
  private final int maxConcurrency;

  public GoogleChannelConnectionFactory(
      ChannelFactory channelFactory,
      String target,
      String proxy,
      AuthAndTLSOptions options,
      List<ClientInterceptor> interceptors,
      int maxConcurrency) {
    this.channelFactory = channelFactory;
    this.target = target;
    this.proxy = proxy;
    this.options = options;
    this.interceptors = interceptors;
    this.maxConcurrency = maxConcurrency;
  }

  @Override
  public Single<ChannelConnection> create() {
    return Single.fromCallable(
        () ->
            new ChannelConnection(channelFactory.newChannel(target, proxy, options, interceptors)));
  }

  @Override
  public int maxConcurrency() {
    return maxConcurrency;
  }
}
