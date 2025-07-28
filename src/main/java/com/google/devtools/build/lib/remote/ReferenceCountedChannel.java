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
package com.google.devtools.build.lib.remote;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.remote.ChannelConnectionWithServerCapabilitiesFactory.ChannelConnectionWithServerCapabilities;
import com.google.devtools.build.lib.remote.grpc.DynamicConnectionPool;
import com.google.devtools.build.lib.remote.grpc.SharedConnectionFactory.SharedConnection;
import com.google.devtools.build.lib.remote.util.RxFutures;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import io.grpc.Channel;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import io.reactivex.rxjava3.annotations.CheckReturnValue;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleObserver;
import io.reactivex.rxjava3.core.SingleSource;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.functions.Function;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 * A wrapper around a {@link DynamicConnectionPool} exposing {@link Channel} and a reference count.
 * When instantiated the reference count is 1. {@link DynamicConnectionPool#close()} will be called
 * on the wrapped channel when the reference count reaches 0.
 *
 * <p>See {@link ReferenceCounted} for more information about reference counting.
 */
public class ReferenceCountedChannel implements ReferenceCounted {
  private final DynamicConnectionPool dynamicConnectionPool;
  private final AbstractReferenceCounted referenceCounted =
      new AbstractReferenceCounted() {
        @Override
        protected void deallocate() {
          try {
            dynamicConnectionPool.close();
          } catch (IOException e) {
            throw new AssertionError(e.getMessage(), e);
          }
        }

        @Override
        public ReferenceCounted touch(Object o) {
          return this;
        }
      };

  public ReferenceCountedChannel(ChannelConnectionWithServerCapabilitiesFactory connectionFactory) {
    this(connectionFactory, /* maxConnections= */ 0);
  }

  public ReferenceCountedChannel(
      ChannelConnectionWithServerCapabilitiesFactory connectionFactory, int maxConnections) {
    this.dynamicConnectionPool =
        new DynamicConnectionPool(
            connectionFactory, connectionFactory.maxConcurrency(), maxConnections);
  }

  public ServerCapabilities getServerCapabilities() throws IOException {
    try (var s = Profiler.instance().profile("getServerCapabilities")) {
      return blockingGet(
          withChannelConnection(ChannelConnectionWithServerCapabilities::getServerCapabilities));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException(e);
    }
  }

  public boolean isShutdown() {
    return dynamicConnectionPool.isClosed();
  }

  /**
   * A specialized {@link Function} that can only throw {@link IOException} and {@link
   * InterruptedException}.
   */
  @FunctionalInterface
  public interface IOFunction<T, R> extends Function<T, R> {
    @Override
    R apply(T t) throws IOException, InterruptedException;
  }

  @CheckReturnValue
  public <T> ListenableFuture<T> withChannelFuture(
      IOFunction<Channel, ? extends ListenableFuture<T>> source) {
    return RxFutures.toListenableFuture(
        withChannel(channel -> RxFutures.toSingle(() -> source.apply(channel), directExecutor())));
  }

  public <T> T withChannelBlocking(IOFunction<Channel, T> source)
      throws IOException, InterruptedException {
    return blockingGet(withChannel(channel -> Single.just(source.apply(channel))));
  }

  // prevents rxjava silent possible wrap of RuntimeException and misinterpretation
  private <T> T blockingGet(Single<T> single) throws IOException, InterruptedException {
    SettableFuture<T> future = SettableFuture.create();
    single.subscribe(
        new SingleObserver<T>() {
          @Override
          public void onError(Throwable t) {
            future.setException(t);
          }

          @Override
          public void onSuccess(T t) {
            future.set(t);
          }

          @Override
          public void onSubscribe(Disposable d) {
            future.addListener(
                () -> {
                  if (future.isCancelled()) {
                    d.dispose();
                  }
                },
                directExecutor());
          }
        });

    try {
      return future.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      Throwables.throwIfInstanceOf(cause, IOException.class);
      Throwables.throwIfInstanceOf(cause, InterruptedException.class);
      Throwables.throwIfUnchecked(cause);
      throw new IllegalStateException("Unexpected exception type", cause);
    }
  }

  @CheckReturnValue
  public <T> Single<T> withChannel(Function<Channel, ? extends SingleSource<? extends T>> source) {
    return withChannelConnection(channelConnection -> source.apply(channelConnection.getChannel()));
  }

  private <T> Single<T> withChannelConnection(
      Function<ChannelConnectionWithServerCapabilities, ? extends SingleSource<? extends T>>
          source) {
    return dynamicConnectionPool
        .create()
        .flatMap(
            sharedConnection ->
                Single.using(
                    () -> sharedConnection,
                    conn -> {
                      var connection =
                          (ChannelConnectionWithServerCapabilities)
                              sharedConnection.getUnderlyingConnection();
                      return source.apply(connection);
                    },
                    SharedConnection::close));
  }

  @Override
  public int refCnt() {
    return referenceCounted.refCnt();
  }

  @CanIgnoreReturnValue
  @Override
  public ReferenceCountedChannel retain() {
    referenceCounted.retain();
    return this;
  }

  @CanIgnoreReturnValue
  @Override
  public ReferenceCountedChannel retain(int increment) {
    referenceCounted.retain(increment);
    return this;
  }

  @CanIgnoreReturnValue
  @Override
  public ReferenceCounted touch() {
    referenceCounted.touch();
    return this;
  }

  @CanIgnoreReturnValue
  @Override
  public ReferenceCounted touch(Object hint) {
    referenceCounted.touch(hint);
    return this;
  }

  @Override
  public boolean release() {
    return referenceCounted.release();
  }

  @Override
  public boolean release(int decrement) {
    return referenceCounted.release(decrement);
  }
}
