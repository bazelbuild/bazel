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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.base.Optional;
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
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Black box integration tests for {@link HttpConnectorMultiplexer}. */
@RunWith(JUnit4.class)
public class HttpConnectorMultiplexerIntegrationTest {

  @Rule public final Timeout globalTimeout = new Timeout(20000);

  private final ExecutorService executor = Executors.newSingleThreadExecutor();
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
      new HttpConnectorMultiplexer(eventHandler, connector, httpStreamFactory);

  private static Optional<Checksum> makeChecksum(String string) {
    try {
      return Optional.of(Checksum.fromString(KeyType.SHA256, string));
    } catch (Checksum.InvalidChecksumException e) {
      throw new IllegalStateException(e);
    }
  }

  private static final Optional<Checksum> HELLO_SHA256 =
      makeChecksum("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");

  @Before
  public void before() throws Exception {
    when(proxyHelper.createProxyIfNeeded(any(URL.class))).thenReturn(Proxy.NO_PROXY);
  }

  @After
  public void after() throws Exception {
    executor.shutdown();
  }

  @Test
  public void successWithRetry() throws Exception {
    CyclicBarrier barrier = new CyclicBarrier(2);
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> unused =
          executor.submit(
              () -> {
                barrier.await();
                for (String status : asList("503 MELTDOWN", "500 ERROR", "200 OK")) {
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
              });
      barrier.await();
      try (HttpStream stream =
          multiplexer.connect(
              new URL(String.format("http://localhost:%d", server.getLocalPort())), HELLO_SHA256)) {
        assertThat(toByteArray(stream)).isEqualTo("hello".getBytes(US_ASCII));
      }
    }
  }

  @Test
  public void captivePortal_isAvoided() throws Exception {
    CyclicBarrier barrier = new CyclicBarrier(2);
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> unused =
          executor.submit(
              () -> {
                barrier.await();
                try (Socket socket = server.accept()) {
                  readHttpRequest(socket.getInputStream());
                  sendLines(
                      socket,
                      "HTTP/1.1 200 OK",
                      "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                      "Connection: close",
                      "",
                      "Never gonna give you up etc.");
                }
                return null;
              });
      barrier.await();
      IOException e =
          assertThrows(
              IOException.class,
              () ->
                  multiplexer.connect(
                      new URL(String.format("http://localhost:%d", server.getLocalPort())),
                      HELLO_SHA256));
      assertThat(e).hasMessageThat().containsMatch("Checksum was .+ but wanted");
    }
  }

  @Test
  public void retryButKeepsFailing() throws Exception {
    CyclicBarrier barrier = new CyclicBarrier(2);
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> unused =
          executor.submit(
              () -> {
                barrier.await();
                while (true) {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 503 MELTDOWN",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "",
                        "");
                  }
                }
              });
      barrier.await();
      IOException e =
          assertThrows(
              IOException.class,
              () ->
                  multiplexer.connect(
                      new URL(String.format("http://localhost:%d", server.getLocalPort())),
                      HELLO_SHA256));
      assertThat(e).hasMessageThat().contains("GET returned 503 MELTDOWN");
    }
  }
}
