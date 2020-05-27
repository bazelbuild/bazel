// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.io.ByteStreams.toByteArray;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.repository.downloader.DownloaderTestUtils.sendLines;
import static com.google.devtools.build.lib.bazel.repository.downloader.HttpParser.readHttpRequest;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.util.Arrays.asList;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.Sleeper;
import java.io.IOException;
import java.net.InetAddress;
import java.net.Proxy;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URL;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Phaser;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.Timeout;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Black box integration tests for {@link HttpConnectorMultiplexer}. */
@RunWith(JUnit4.class)
public class HttpConnectorMultiplexerIntegrationTest {

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Rule public final Timeout globalTimeout = new Timeout(20000);

  private final ExecutorService executor = Executors.newFixedThreadPool(3);
  private final ProxyHelper proxyHelper = mock(ProxyHelper.class);
  private final ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);
  private final ManualClock clock = new ManualClock();
  private final Sleeper sleeper = mock(Sleeper.class);
  private final Locale locale = Locale.US;
  private final HttpConnector connector =
      new HttpConnector(locale, eventHandler, proxyHelper, sleeper, 0.15f);
  private final ProgressInputStream.Factory progressInputStreamFactory =
      new ProgressInputStream.Factory(locale, clock, eventHandler);
  private final HttpStream.Factory httpStreamFactory =
      new HttpStream.Factory(progressInputStreamFactory);
  private final HttpConnectorMultiplexer multiplexer =
      new HttpConnectorMultiplexer(eventHandler, connector, httpStreamFactory, clock, sleeper);

  private static final Optional<Checksum> HELLO_SHA256 =
      Optional.of(
          Checksum.fromString(
              KeyType.SHA256, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"));

  @Before
  public void before() throws Exception {
    when(proxyHelper.createProxyIfNeeded(any(URL.class))).thenReturn(Proxy.NO_PROXY);
  }

  @After
  public void after() throws Exception {
    executor.shutdown();
  }

  @Test
  public void normalRequest() throws Exception {
    final Phaser phaser = new Phaser(3);
    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      for (final ServerSocket server : asList(server1, server2)) {
        @SuppressWarnings("unused")
        Future<?> possiblyIgnoredError =
            executor.submit(
                new Callable<Object>() {
                  @Override
                  public Object call() throws Exception {
                    for (String status : asList("503 MELTDOWN", "500 ERROR", "200 OK")) {
                      phaser.arriveAndAwaitAdvance();
                      try (Socket socket = server.accept()) {
                        readHttpRequest(socket.getInputStream());
                        sendLines(
                            socket,
                            "HTTP/1.1 " + status,
                            "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                            "Connection: close",
                            "",
                            "hello");
                      }
                    }
                    return null;
                  }
                });
      }
      phaser.arriveAndAwaitAdvance();
      phaser.arriveAndDeregister();
      try (HttpStream stream =
          multiplexer.connect(
              ImmutableList.of(
                  new URL(String.format("http://localhost:%d", server1.getLocalPort())),
                  new URL(String.format("http://localhost:%d", server2.getLocalPort()))),
              HELLO_SHA256)) {
        assertThat(toByteArray(stream)).isEqualTo("hello".getBytes(US_ASCII));
      }
    }
  }

  @Test
  public void captivePortal_isAvoided() throws Exception {
    final CyclicBarrier barrier = new CyclicBarrier(2);
    doAnswer(
        new Answer<Void>() {
          @Override
          public Void answer(InvocationOnMock invocation) throws Throwable {
            barrier.await();
            return null;
          }
        }).when(sleeper).sleepMillis(anyLong());
    try (final ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        final ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server1.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Warning: https://youtu.be/rJ6O5sTPn1k",
                        "Connection: close",
                        "",
                        "Und wird die Welt auch in Flammen stehen",
                        "Wir werden wieder auferstehen");
                  }
                  barrier.await();
                  return null;
                }
              });
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError1 =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server2.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "",
                        "hello");
                  }
                  return null;
                }
              });
      try (HttpStream stream =
          multiplexer.connect(
              ImmutableList.of(
                  new URL(String.format("http://localhost:%d", server1.getLocalPort())),
                  new URL(String.format("http://localhost:%d", server2.getLocalPort()))),
              HELLO_SHA256)) {
        assertThat(toByteArray(stream)).isEqualTo("hello".getBytes(US_ASCII));
      }
    }
  }

  @Test
  public void allMirrorsDown_throwsIOException() throws Exception {
    final CyclicBarrier barrier = new CyclicBarrier(4);
    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server3 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      for (final ServerSocket server : asList(server1, server2, server3)) {
        Future<?> unused = executor.submit(
            new Callable<Object>() {
              @Override
              public Object call() throws Exception {
                barrier.await();
                while (true) {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(socket,
                        "HTTP/1.1 503 MELTDOWN",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Warning: https://youtu.be/6M6samPEMpM",
                        "Connection: close",
                        "",
                        "");
                  }
                }
              }
            });
      }
      barrier.await();
      thrown.expect(IOException.class);
      thrown.expectMessage("All mirrors are down: [GET returned 503 MELTDOWN]");
      multiplexer.connect(
          ImmutableList.of(
              new URL(String.format("http://localhost:%d", server1.getLocalPort())),
              new URL(String.format("http://localhost:%d", server2.getLocalPort())),
              new URL(String.format("http://localhost:%d", server3.getLocalPort()))),
          HELLO_SHA256);
    }
  }

  @Test
  public void firstUrlSocketTimeout_secondOk() throws Exception {

    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {

      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
                try (Socket socket = server1.accept()) {
                  // Do nothing to cause SocketTimeoutException on client side.
                }
                return null;
              });

      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError2 =
          executor.submit(
              () -> {
                try (Socket socket = server2.accept()) {
                  readHttpRequest(socket.getInputStream());
                  sendLines(
                      socket,
                      "HTTP/1.1 200 OK",
                      "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                      "Connection: close",
                      "",
                      "hello");
                }
                return null;
              });

      try (HttpStream stream =
          multiplexer.connect(
              ImmutableList.of(
                  new URL(String.format("http://localhost:%d", server1.getLocalPort())),
                  new URL(String.format("http://localhost:%d", server2.getLocalPort()))),
              HELLO_SHA256)) {
        assertThat(toByteArray(stream)).isEqualTo("hello".getBytes(US_ASCII));
      }
    }
  }
}
