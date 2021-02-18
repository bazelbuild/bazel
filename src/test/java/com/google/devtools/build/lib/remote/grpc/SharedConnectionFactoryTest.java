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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.remote.grpc.SharedConnectionFactory.SharedConnection;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.observers.TestObserver;
import io.reactivex.rxjava3.plugins.RxJavaPlugins;
import java.io.IOException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.After;
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

  private final AtomicReference<Throwable> rxGlobalThrowable = new AtomicReference<>(null);

  @Mock private Connection connection;
  @Mock private ConnectionFactory connectionFactory;

  @Before
  public void setUp() {
    RxJavaPlugins.setErrorHandler(rxGlobalThrowable::set);

    when(connectionFactory.create()).thenAnswer(invocation -> Single.just(connection));
  }

  @After
  public void tearDown() throws Throwable {
    // Make sure rxjava didn't receive global errors
    Throwable t = rxGlobalThrowable.getAndSet(null);
    if (t != null) {
      throw t;
    }
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
    SharedConnectionFactory factory = new SharedConnectionFactory(connectionFactory, 2);
    Semaphore semaphore = new Semaphore(0);
    AtomicBoolean finished = new AtomicBoolean(false);
    Thread t =
        new Thread(
            () -> {
              factory
                  .create()
                  .doOnSuccess(
                      conn -> {
                        assertThat(conn.getUnderlyingConnection()).isEqualTo(connection);
                        semaphore.release();
                        Thread.sleep(Integer.MAX_VALUE);
                        finished.set(true);
                      })
                  .blockingSubscribe();

              finished.set(true);
            });
    t.start();
    semaphore.acquire();

    TestObserver<SharedConnection> observer = factory.create().test();

    observer.assertValue(conn -> conn.getUnderlyingConnection() == connection).assertComplete();
    assertThat(finished.get()).isFalse();
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
