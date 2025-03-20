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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.remote.grpc.SharedConnectionFactory.SharedConnection;
import com.google.devtools.build.lib.remote.util.RxNoGlobalErrorsRule;
import io.grpc.CallOptions;
import io.grpc.ClientCall;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.observers.TestObserver;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Queue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests for {@link SharedConnectionFactory}. */
@RunWith(JUnit4.class)
public class SharedConnectionFactoryTest {
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();
  @Rule public final RxNoGlobalErrorsRule rxNoGlobalErrorsRule = new RxNoGlobalErrorsRule();

  @Mock private Connection connection;
  @Mock private ConnectionFactory connectionFactory;

  @Before
  public void setUp() {
    when(connectionFactory.create()).thenAnswer(invocation -> Single.just(connection));
  }

  @Test
  public void create_smoke() {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    assertThat(factory.numAvailableConnections()).isEqualTo(1);

    TestObserver<SharedConnection> observer = factory.create().test();

    observer.assertValue(conn -> conn.getUnderlyingConnection() == connection).assertComplete();
    verify(connectionFactory, times(1)).create();
    assertThat(factory.numAvailableConnections()).isEqualTo(0);
  }

  @Test
  public void create_noConnectionCreationBeforeSubscription() {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);

    factory.create();

    verify(connectionFactory, times(0)).create();
  }

  @Test
  public void create_exceedingMaxConcurrency_waiting() {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    TestObserver<SharedConnection> observer1 = factory.create().test();
    assertThat(factory.numAvailableConnections()).isEqualTo(0);
    observer1.assertValue(conn -> conn.getUnderlyingConnection() == connection).assertComplete();

    TestObserver<SharedConnection> observer2 = factory.create().test();
    observer2.assertEmpty();
  }

  @Test
  public void create_afterConnectionClosed_shareConnections() throws IOException {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    TestObserver<SharedConnection> observer1 = factory.create().test();
    assertThat(factory.numAvailableConnections()).isEqualTo(0);
    observer1.assertValue(conn -> conn.getUnderlyingConnection() == connection).assertComplete();
    TestObserver<SharedConnection> observer2 = factory.create().test();

    observer1.values().get(0).close();

    observer2.assertValue(conn -> conn.getUnderlyingConnection() == connection).assertComplete();
    assertThat(factory.numAvailableConnections()).isEqualTo(0);
  }

  @Test
  public void create_belowMaxConcurrency_shareConnections() {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 2);

    TestObserver<SharedConnection> observer1 = factory.create().test();
    assertThat(factory.numAvailableConnections()).isEqualTo(1);
    observer1.assertValue(conn -> conn.getUnderlyingConnection() == connection).assertComplete();

    TestObserver<SharedConnection> observer2 = factory.create().test();
    observer2.assertValue(conn -> conn.getUnderlyingConnection() == connection).assertComplete();
    assertThat(factory.numAvailableConnections()).isEqualTo(0);
  }

  @Test
  public void create_concurrentCreate_shareConnections() throws InterruptedException {
    int maxConcurrency = 10;
    SharedConnectionFactory factory =
        new SharedConnectionFactory(connectionFactory, maxConcurrency);
    AtomicReference<Throwable> error = new AtomicReference<>(null);
    Runnable runnable =
        () -> {
          try {
            TestObserver<SharedConnection> observer = factory.create().test();

            observer
                .assertNoErrors()
                .assertValue(conn -> conn.getUnderlyingConnection() == connection)
                .assertComplete();
          } catch (Throwable e) {
            error.set(e);
          }
        };
    Thread[] threads = new Thread[maxConcurrency];
    for (int i = 0; i < threads.length; ++i) {
      threads[i] = new Thread(runnable);
    }

    for (Thread thread : threads) {
      thread.start();
    }
    for (Thread thread : threads) {
      thread.join();
    }

    assertThat(error.get()).isNull();
    verify(connectionFactory, times(1)).create();
  }

  private static final class FatalIOException extends IOException {
    FatalIOException() {
      super("fatal");
    }
  }

  @SuppressWarnings({"unchecked", "CannotMockFinalClass"})
  @Test
  public void create_belowMaxConcurrency_fatalErrorPreventsReuse() throws IOException {
    Connection brokenConnection =
        new Connection() {
          @Override
          public <ReqT, RespT> ClientCall<ReqT, RespT> call(
              MethodDescriptor<ReqT, RespT> method, CallOptions options) {
            var call = mock(ClientCall.class);
            doAnswer(
                    invocationOnMock -> {
                      ((ClientCall.Listener) invocationOnMock.getArgument(0))
                          .onClose(Status.fromThrowable(new FatalIOException()), new Metadata());
                      return null;
                    })
                .when(call)
                .start(any(), any());
            return call;
          }

          @Override
          public void close() {}
        };
    Connection newConnection = mock(Connection.class);
    Queue<Connection> connectionsToCreate =
        new ArrayDeque<>(ImmutableList.of(brokenConnection, newConnection));
    when(connectionFactory.create())
        .thenAnswer(invocation -> Single.just(connectionsToCreate.remove()));

    SharedConnectionFactory factory =
        new SharedConnectionFactory(connectionFactory, 2, t -> t instanceof FatalIOException);

    TestObserver<SharedConnection> observer1 = factory.create().test();
    assertThat(factory.numAvailableConnections()).isEqualTo(1);
    observer1
        .assertValue(conn -> conn.getUnderlyingConnection() == brokenConnection)
        .assertComplete();

    // Submit a call on the first connection and have it fail.
    MethodDescriptor.Marshaller<byte[]> nullMarshaller =
        new MethodDescriptor.Marshaller<>() {
          @Override
          public InputStream stream(byte[] bytes) {
            return null;
          }

          @Override
          public byte[] parse(InputStream inputStream) {
            return null;
          }
        };
    try (Connection firstConnection = observer1.values().getFirst()) {
      var call =
          firstConnection.call(
              MethodDescriptor.newBuilder(nullMarshaller, nullMarshaller)
                  .setType(MethodDescriptor.MethodType.CLIENT_STREAMING)
                  .setFullMethodName("testMethod")
                  .build(),
              CallOptions.DEFAULT);
      ClientCall.Listener<byte[]> listener = new ClientCall.Listener<>() {};
      call.start(listener, new Metadata());
      listener.onClose(Status.fromThrowable(new FatalIOException()), new Metadata());
    }

    // Validate that the connection is not reused.
    TestObserver<SharedConnection> observer2 = factory.create().test();
    observer2.assertValue(conn -> conn.getUnderlyingConnection() == newConnection).assertComplete();
    assertThat(factory.numAvailableConnections()).isEqualTo(1);
  }

  @Test
  public void create_afterLastFailed_success() {
    AtomicInteger times = new AtomicInteger(0);
    ConnectionFactory connectionFactory = mock(ConnectionFactory.class);
    when(connectionFactory.create())
        .thenAnswer(
            invocation -> {
              if (times.getAndIncrement() == 0) {
                return Single.error(new IllegalStateException("error"));
              }

              return Single.just(connection);
            });
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    Single<SharedConnection> connectionSingle = factory.create();

    connectionSingle
        .test()
        .assertError(IllegalStateException.class)
        .assertError(e -> e.getMessage().contains("error"));
    assertThat(factory.numAvailableConnections()).isEqualTo(1);
    connectionSingle
        .test()
        .assertValue(conn -> conn.getUnderlyingConnection() == connection)
        .assertComplete();

    assertThat(times.get()).isEqualTo(2);
    assertThat(factory.numAvailableConnections()).isEqualTo(0);
  }

  @Test
  public void create_disposeWhenWaitingForConnectionCreation_doNotCancelCreation()
      throws InterruptedException {
    AtomicBoolean canceled = new AtomicBoolean(false);
    AtomicBoolean finished = new AtomicBoolean(false);
    Semaphore disposed = new Semaphore(0);
    Semaphore terminated = new Semaphore(0);
    ConnectionFactory connectionFactory = mock(ConnectionFactory.class);
    when(connectionFactory.create())
        .thenAnswer(
            invocation ->
                Single.create(
                        emitter ->
                            new Thread(
                                    () -> {
                                      try {
                                        disposed.acquire();
                                        finished.set(true);
                                        emitter.onSuccess(connection);
                                      } catch (InterruptedException e) {
                                        emitter.onError(e);
                                      }
                                      terminated.release();
                                    })
                                .start())
                    .doOnDispose(() -> canceled.set(true)));
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    TestObserver<SharedConnection> observer = factory.create().test();
    assertThat(factory.numAvailableConnections()).isEqualTo(0);

    observer.assertEmpty().dispose();
    disposed.release();

    terminated.acquire();
    assertThat(canceled.get()).isFalse();
    assertThat(finished.get()).isTrue();
    assertThat(factory.numAvailableConnections()).isEqualTo(1);
  }

  @Test
  public void create_interrupt_terminate() throws InterruptedException {
    AtomicBoolean finished = new AtomicBoolean(false);
    AtomicBoolean interrupted = new AtomicBoolean(true);
    Semaphore threadTerminatedSemaphore = new Semaphore(0);
    Semaphore connectionCreationSemaphore = new Semaphore(0);
    ConnectionFactory connectionFactory = mock(ConnectionFactory.class);
    when(connectionFactory.create())
        .thenAnswer(
            invocation ->
                Single.create(
                    emitter ->
                        new Thread(
                                () -> {
                                  try {
                                    Thread.sleep(Integer.MAX_VALUE);
                                    finished.set(true);
                                    emitter.onSuccess(connectionFactory);
                                  } catch (InterruptedException e) {
                                    emitter.onError(e);
                                  }
                                })
                            .start()));
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 2);
    factory.create().test().assertEmpty();
    Thread t =
        new Thread(
            () -> {
              try {
                TestObserver<SharedConnection> observer = factory.create().test();
                connectionCreationSemaphore.release();
                observer.await();
              } catch (InterruptedException e) {
                interrupted.set(true);
              }

              threadTerminatedSemaphore.release();
            });
    t.start();

    connectionCreationSemaphore.acquire();
    t.interrupt();
    threadTerminatedSemaphore.acquire();

    assertThat(finished.get()).isFalse();
    assertThat(interrupted.get()).isTrue();
  }

  @Test
  public void closeConnection_connectionBecomeAvailable() throws IOException {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    TestObserver<SharedConnection> observer = factory.create().test();
    observer.assertComplete();
    SharedConnection conn = observer.values().get(0);
    assertThat(factory.numAvailableConnections()).isEqualTo(0);

    conn.close();

    assertThat(factory.numAvailableConnections()).isEqualTo(1);
    verify(connection, times(0)).close();
  }

  @Test
  public void closeFactory_closeUnderlyingConnection() throws IOException {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    TestObserver<SharedConnection> observer = factory.create().test();
    observer.assertComplete();

    factory.close();

    verify(connection, times(1)).close();
  }

  @Test
  public void closeFactory_noNewConnectionAllowed() throws IOException {
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    factory.close();

    TestObserver<SharedConnection> observer = factory.create().test();

    observer
        .assertError(IllegalStateException.class)
        .assertError(e -> e.getMessage().contains("closed"));
  }

  @Test
  public void closeFactory_pendingConnectionCreation_closedError()
      throws IOException, InterruptedException {
    AtomicBoolean canceled = new AtomicBoolean(false);
    AtomicBoolean finished = new AtomicBoolean(false);
    Semaphore terminated = new Semaphore(0);
    ConnectionFactory connectionFactory = mock(ConnectionFactory.class);
    when(connectionFactory.create())
        .thenAnswer(
            invocation ->
                Single.create(
                        emitter -> {
                          Thread t =
                              new Thread(
                                  () -> {
                                    try {
                                      Thread.sleep(Integer.MAX_VALUE);
                                      finished.set(true);
                                      emitter.onSuccess(connection);
                                    } catch (InterruptedException ignored) {
                                      /* no-op */
                                    }

                                    terminated.release();
                                  });
                          t.start();

                          emitter.setCancellable(t::interrupt);
                        })
                    .doOnDispose(() -> canceled.set(true)));
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 1);
    TestObserver<SharedConnection> observer = factory.create().test();
    observer.assertEmpty();

    assertThat(canceled.get()).isFalse();
    factory.close();

    terminated.acquire();
    observer
        .assertError(IllegalStateException.class)
        .assertError(e -> e.getMessage().contains("closed"));
    assertThat(canceled.get()).isTrue();
    assertThat(finished.get()).isFalse();
  }
}
