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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.DigestFunction.Value;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.remote.RemoteServerCapabilities.ServerCapabilitiesRequirement;
import com.google.devtools.build.lib.remote.grpc.ChannelConnectionFactory;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.RxFutures;
import com.google.devtools.build.lib.remote.util.Utils;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * A {@link ChannelConnectionFactory} which creates {@link ChannelConnection} using {@link
 * ChannelFactory}.
 */
public class GoogleChannelConnectionFactory
    implements ChannelConnectionWithServerCapabilitiesFactory {
  private final AtomicBoolean getAndVerifyServerCapabilitiesStarted = new AtomicBoolean(false);
  private final SettableFuture<ServerCapabilities> serverCapabilities = SettableFuture.create();

  private final ChannelFactory channelFactory;
  private final String target;
  private final String proxy;
  private final AuthAndTLSOptions options;
  private final List<ClientInterceptor> interceptors;
  private final int maxConcurrency;
  private final boolean verboseFailures;
  private final Reporter reporter;
  @Nullable private final RemoteServerCapabilities remoteServerCapabilities;
  private final RemoteOptions remoteOptions;
  private final DigestFunction.Value digestFunction;
  private final ServerCapabilitiesRequirement requirement;

  public GoogleChannelConnectionFactory(
      ChannelFactory channelFactory,
      String target,
      String proxy,
      RemoteOptions remoteOptions,
      AuthAndTLSOptions options,
      List<ClientInterceptor> interceptors,
      int maxConcurrency,
      boolean verboseFailures,
      Reporter reporter,
      @Nullable RemoteServerCapabilities remoteServerCapabilities,
      Value digestFunction,
      ServerCapabilitiesRequirement requirement) {
    if (requirement != ServerCapabilitiesRequirement.NONE) {
      checkNotNull(remoteServerCapabilities);
    }

    this.channelFactory = channelFactory;
    this.target = target;
    this.proxy = proxy;
    this.options = options;
    this.interceptors = interceptors;
    this.maxConcurrency = maxConcurrency;
    this.verboseFailures = verboseFailures;
    this.reporter = reporter;
    this.remoteServerCapabilities = remoteServerCapabilities;
    this.remoteOptions = remoteOptions;
    this.digestFunction = digestFunction;
    this.requirement = requirement;
  }

  @Override
  public Single<ChannelConnectionWithServerCapabilities> create() {
    return Single.fromCallable(
            () -> channelFactory.newChannel(target, proxy, options, interceptors))
        .flatMap(
            channel -> {
              var serverCapabilitiesSingle =
                  RxFutures.toSingle(
                      () -> getAndVerifyServerCapabilities(channel), directExecutor());

              // Don't issue GetCapabilities calls if the requirement is NONE because the endpoint,
              // e.g. Remote Asset API, might not implement the API. See #20342.
              if (requirement == ServerCapabilitiesRequirement.NONE) {
                return Single.just(
                    new ChannelConnectionWithServerCapabilities(channel, serverCapabilitiesSingle));
              }

              return serverCapabilitiesSingle.map(
                  sc -> new ChannelConnectionWithServerCapabilities(channel, Single.just(sc)));
            });
  }

  private ListenableFuture<ServerCapabilities> getAndVerifyServerCapabilities(
      ManagedChannel channel) {
    if (getAndVerifyServerCapabilitiesStarted.compareAndSet(false, true)) {
      var s = Profiler.instance().profile("getAndVerifyServerCapabilities");
      var future =
          Futures.transformAsync(
              checkNotNull(remoteServerCapabilities).get(channel),
              serverCapabilities -> {
                var result =
                    RemoteServerCapabilities.checkClientServerCompatibility(
                        serverCapabilities, remoteOptions, digestFunction, requirement);
                for (String warning : result.getWarnings()) {
                  reporter.handle(Event.warn(warning));
                }
                List<String> errors = result.getErrors();
                for (int i = 0; i < errors.size() - 1; ++i) {
                  reporter.handle(Event.error(errors.get(i)));
                }
                if (!errors.isEmpty()) {
                  String lastErrorMessage = errors.get(errors.size() - 1);
                  return immediateFailedFuture(new IOException(lastErrorMessage));
                }
                return immediateFuture(serverCapabilities);
              },
              directExecutor());
      future.addListener(s::close, directExecutor());
      Futures.addCallback(
          future,
          new FutureCallback<ServerCapabilities>() {
            @Override
            public void onSuccess(ServerCapabilities result) {
              serverCapabilities.set(result);
            }

            @Override
            public void onFailure(Throwable error) {
              String message =
                  "Failed to query remote execution capabilities: "
                      + Utils.grpcAwareErrorMessage(error, verboseFailures);
              reporter.handle(Event.error(message));

              IOException exception;
              if (error instanceof IOException) {
                exception = (IOException) error;
              } else {
                exception = new IOException(error);
              }
              serverCapabilities.setException(exception);
            }
          },
          directExecutor());
    }
    return serverCapabilities;
  }

  @Override
  public int maxConcurrency() {
    return maxConcurrency;
  }
}
