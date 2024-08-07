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
package com.google.devtools.build.lib.remote.grpc;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.ForwardingClientCall.SimpleForwardingClientCall;
import io.grpc.ForwardingClientCallListener.SimpleForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import io.netty.channel.unix.Errors;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.functions.Action;
import io.reactivex.rxjava3.subjects.AsyncSubject;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * A {@link ConnectionPool} that creates one connection using provided {@link ConnectionFactory} and
 * shares the connection upto {@code maxConcurrency}.
 *
 * <p>This is useful if underlying connection maintains a connection pool internally. (such as
 * {@code Channel} in gRPC)
 *
 * <p>Connections must be closed with {@link Connection#close()} in order to be reused later.
 */
@ThreadSafe
public class SharedConnectionFactory implements ConnectionPool {
  private final TokenBucket<Integer> tokenBucket;
  private final ConnectionFactory factory;

  @Nullable
  @GuardedBy("this")
  private AsyncSubject<Connection> connectionAsyncSubject = null;

  private final AtomicReference<Disposable> connectionCreationDisposable =
      new AtomicReference<>(null);

  public SharedConnectionFactory(ConnectionFactory factory, int maxConcurrency) {
    this.factory = factory;

    List<Integer> initialTokens = new ArrayList<>(maxConcurrency);
    for (int i = 0; i < maxConcurrency; ++i) {
      initialTokens.add(i);
    }
    this.tokenBucket = new TokenBucket<>(initialTokens);
  }

  @Override
  public void close() throws IOException {
    tokenBucket.close();

    Disposable d = connectionCreationDisposable.getAndSet(null);
    if (d != null && !d.isDisposed()) {
      d.dispose();
    }

    synchronized (this) {
      if (connectionAsyncSubject != null) {
        Connection connection = connectionAsyncSubject.getValue();
        if (connection != null) {
          connection.close();
        }

        // If it still has observers, it means the subject hasn't completed. Complete it now.
        if (connectionAsyncSubject.hasObservers()) {
          connectionAsyncSubject.onError(new IllegalStateException("closed"));
        }
      }
    }
  }

  private AsyncSubject<Connection> createUnderlyingConnectionIfNot() {
    synchronized (this) {
      if (connectionAsyncSubject == null || connectionAsyncSubject.hasThrowable()) {
        connectionAsyncSubject =
            factory
                .create()
                .doOnSubscribe(connectionCreationDisposable::set)
                .toObservable()
                .subscribeWith(AsyncSubject.create());
      }

      return connectionAsyncSubject;
    }
  }

  private Single<? extends Connection> acquireConnection() {
    return Single.fromObservable(createUnderlyingConnectionIfNot());
  }

  /**
   * Reuses the underlying {@link Connection} and wait for it to be released if is exceeding {@code
   * maxConcurrency}.
   */
  @Override
  public Single<SharedConnection> create() {
    return tokenBucket
        .acquireToken()
        .flatMap(
            token ->
                acquireConnection()
                    .doOnError(ignored -> tokenBucket.addToken(token))
                    .doOnDispose(() -> tokenBucket.addToken(token))
                    .map(
                        conn ->
                            new SharedConnection(
                                conn,
                                /* onClose= */ () -> tokenBucket.addToken(token),
                                /* onFatalError= */ () -> {
                                  synchronized (this) {
                                    connectionAsyncSubject = null;
                                  }
                                })));
  }

  /** Returns current number of available connections. */
  public int numAvailableConnections() {
    return tokenBucket.size();
  }

  /** A {@link Connection} which wraps an underlying connection and is shared between consumers. */
  public static class SharedConnection implements Connection {
    private final Connection connection;
    private final Action onClose;
    private final Runnable onFatalError;

    public SharedConnection(Connection connection, Action onClose, Runnable onFatalError) {
      this.connection = connection;
      this.onClose = onClose;
      this.onFatalError = onFatalError;
    }

    @Override
    public <ReqT, RespT> ClientCall<ReqT, RespT> call(
        MethodDescriptor<ReqT, RespT> method, CallOptions options) {
      return new SimpleForwardingClientCall<>(connection.call(method, options)) {
        @Override
        public void start(Listener<RespT> responseListener, Metadata headers) {
          super.start(
              new SimpleForwardingClientCallListener<>(responseListener) {
                @Override
                public void onClose(Status status, Metadata trailers) {
                  if (isFatalError(status.getCause())) {
                    onFatalError.run();
                  }
                  super.onClose(status, trailers);
                }
              },
              headers);
        }
      };
    }

    @Override
    public void close() throws IOException {
      try {
        onClose.run();
      } catch (Throwable t) {
        throw new IOException(t);
      }
    }

    /** Returns the underlying connection this shared connection built on */
    public Connection getUnderlyingConnection() {
      return connection;
    }

    private static boolean isFatalError(@Nullable Throwable t) {
      // A low-level netty error indicates that the connection is fundamentally broken
      // and should not be reused for retries.
      return t instanceof Errors.NativeIoException;
    }
  }
}
