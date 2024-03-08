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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.repository.downloader.DownloaderTestUtils.sendLines;
import static com.google.devtools.build.lib.bazel.repository.downloader.HttpParser.readHttpRequest;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.ManualSleeper;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.InetAddress;
import java.net.Proxy;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.net.URLConnection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.Timeout;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpConnector}. */
@RunWith(JUnit4.class)
public class HttpConnectorTest {

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Rule
  public final TemporaryFolder testFolder = new TemporaryFolder();

  @Rule
  public final Timeout globalTimeout = new Timeout(10000);

  private final ExecutorService executor = Executors.newFixedThreadPool(2);
  private final ManualClock clock = new ManualClock();
  private final ManualSleeper sleeper = new ManualSleeper(clock);
  /** Scale timeouts down to make tests fast. */
  private final float timeoutScaling = 0.05f;

  private final EventHandler eventHandler = mock(EventHandler.class);
  private final ProxyHelper proxyHelper = mock(ProxyHelper.class);
  private final HttpConnector connector =
      new HttpConnector(Locale.US, eventHandler, proxyHelper, sleeper, timeoutScaling);

  @Before
  public void before() throws Exception {
    when(proxyHelper.createProxyIfNeeded(any(URL.class))).thenReturn(Proxy.NO_PROXY);
  }

  @After
  public void after() throws Exception {
    executor.shutdown();
  }

  @Test
  public void localFileDownload() throws Exception {
    byte[] fileContents = "this is a test".getBytes(UTF_8);
    assertThat(
            ByteStreams.toByteArray(
                connector
                    .connect(createTempFile(fileContents).toURI().toURL(), url -> ImmutableMap.of())
                    .getInputStream()))
        .isEqualTo(fileContents);
  }

  @Test
  public void badHost_throwsIOException() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Unknown host: bad.example");
    connector.connect(new URL("http://bad.example"), url -> ImmutableMap.of());
  }

  @Test
  public void normalRequest() throws Exception {
    final Map<String, List<String>> headers = new ConcurrentHashMap<>();
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream(), headers);
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 5",
                        "",
                        "hello");
                  }
                  return null;
                }
              });
      try (Reader payload =
          new InputStreamReader(
              connector
                  .connect(
                      new URL(String.format("http://localhost:%d/boo", server.getLocalPort())),
                      url -> ImmutableMap.of("Content-Encoding", ImmutableList.of("gzip")))
                  .getInputStream(),
              ISO_8859_1)) {
        assertThat(CharStreams.toString(payload)).isEqualTo("hello");
      }
    }
    assertThat(headers).containsEntry("x-method", ImmutableList.of("GET"));
    assertThat(headers).containsEntry("x-request-uri", ImmutableList.of("/boo"));
    assertThat(headers).containsEntry("content-encoding", ImmutableList.of("gzip"));
  }

  @Test
  public void serverError_retriesConnect() throws Exception {
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 500 Incredible Catastrophe",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 8",
                        "",
                        "nononono");
                  }
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 5",
                        "",
                        "hello");
                  }
                  return null;
                }
              });
      try (Reader payload =
          new InputStreamReader(
              connector
                  .connect(
                      new URL(String.format("http://localhost:%d", server.getLocalPort())),
                      url -> ImmutableMap.of())
                  .getInputStream(),
              ISO_8859_1)) {
        assertThat(CharStreams.toString(payload)).isEqualTo("hello");
        assertThat(clock.currentTimeMillis()).isGreaterThan(50L);
        assertThat(clock.currentTimeMillis()).isLessThan(150L);
      }
    }
  }

  @Test
  public void connectionRefused_retries() throws Exception {
    final int port;

    // Start and immediately stop server socket to get a free port.
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      port = server.getLocalPort();
    }

    final AtomicReference<ServerSocket> server = new AtomicReference<>();

    try {
      // Schedule server socket to be started only after retry to simulate connection retry.
      sleeper.scheduleRunnable(
          () -> {
            try {
              server.set(new ServerSocket(port, 1, InetAddress.getByName(null)));
            } catch (IOException e) {
              throw new RuntimeException(e);
            }

            @SuppressWarnings("unused")
            Future<?> possiblyIgnoredError =
                executor.submit(
                    () -> {
                      while (!executor.isShutdown()) {
                        try (Socket socket = server.get().accept()) {
                          readHttpRequest(socket.getInputStream());
                          sendLines(
                              socket,
                              "HTTP/1.1 200 OK",
                              "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                              "Connection: close",
                              "Content-Type: text/plain",
                              "Content-Length: 5",
                              "",
                              "hello");
                        }
                      }

                      return null;
                    });
          },
          1);

      try (Reader payload =
          new InputStreamReader(
              connector
                  .connect(
                      new URL(String.format("http://localhost:%d", port)), url -> ImmutableMap.of())
                  .getInputStream(),
              ISO_8859_1)) {
        assertThat(CharStreams.toString(payload)).isEqualTo("hello");
      }
    } finally {
      ServerSocket serverSocket = server.get();

      if (serverSocket != null) {
        serverSocket.close();
      }
    }
  }

  // Deactivated due to https://github.com/bazelbuild/bazel/issues/9380.
  @Ignore
  public void socketTimeout_retries() throws Exception {
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
                try (Socket socket = server.accept()) {
                  // Do nothing to cause SocketTimeoutException on client side.
                }

                // Schedule proper HTTP response once client retries.
                sleeper.scheduleRunnable(
                    () -> {
                      @SuppressWarnings("unused")
                      Future<?> possiblyIgnoredError2 =
                          executor.submit(
                              () -> {
                                while (!executor.isShutdown()) {
                                  try (Socket socket = server.accept()) {
                                    readHttpRequest(socket.getInputStream());
                                    sendLines(
                                        socket,
                                        "HTTP/1.1 200 OK",
                                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                                        "Connection: close",
                                        "Content-Type: text/plain",
                                        "Content-Length: 5",
                                        "",
                                        "hello");
                                  } catch (IOException e) {
                                    throw new RuntimeException(e);
                                  }
                                }
                              });
                    },
                    1);
                return null;
              });

      try (Reader payload =
          new InputStreamReader(
              connector
                  .connect(
                      new URL(String.format("http://localhost:%d", server.getLocalPort())),
                      url -> ImmutableMap.of())
                  .getInputStream(),
              ISO_8859_1)) {
        assertThat(CharStreams.toString(payload)).isEqualTo("hello");
        assertThat(clock.currentTimeMillis()).isEqualTo(1);
      }
    }
  }

  /**
   * It is important part of {@link HttpConnector} contract to not throw raw {@link
   * SocketTimeoutException} because it extends {@link java.io.InterruptedIOException} and {@link
   * HttpConnectorMultiplexer} relies on {@link java.io.InterruptedIOException} to only be thrown
   * when actual interruption happened.
   */
  @Test
  public void socketTimeout_throwsIOExceptionInsteadOfSocketTimeoutException() throws Exception {
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
                try (Socket socket = server.accept()) {
                  // Do nothing to cause SocketTimeoutException on client side.
                }
                return null;
              });

      try (Reader payload =
          new InputStreamReader(
              connector
                  .connect(
                      new URL(String.format("http://localhost:%d", server.getLocalPort())),
                      url -> ImmutableMap.of())
                  .getInputStream(),
              ISO_8859_1)) {
        fail("Should have thrown");
      } catch (IOException expected) {
        if (expected.getCause() != null) {
          // SocketTimeoutException gets wrapped in an IOException and rethrown.
          assertThat(expected).hasCauseThat().isInstanceOf(SocketTimeoutException.class);
          assertThat(expected).hasCauseThat().hasMessageThat().ignoringCase().contains("timed out");
        } else {
          // For windows, it is possible that the thrown exception is a ConnectException (no cause)
          // from {@code HttpURLConnection.connect()} rather than a SocketTimeoutException from
          // {@code HttpURLConnection.getResponseCode()}. In the former case, we expect the
          // SocketTimeoutException to have already occurred but gets suppressed within the final
          // ConnectException (upon max retry).
          ImmutableList<Throwable> suppressed = ImmutableList.copyOf(expected.getSuppressed());
          Optional<Throwable> ste =
              suppressed.stream().filter(t -> t instanceof SocketTimeoutException).findFirst();
          assertThat(ste).isPresent();
          assertThat(ste.get()).isInstanceOf(SocketTimeoutException.class);
          assertThat(ste.get()).hasMessageThat().ignoringCase().contains("timed out");
        }
      }
    }
  }

  @Test
  public void permanentError_doesNotRetryAndThrowsIOException() throws Exception {
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 401 Unauthorized",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 0",
                        "",
                        "");
                  }
                  return null;
                }
              });
      thrown.expect(IOException.class);
      thrown.expectMessage("401 Unauthorized");
      connector.connect(
          new URL(String.format("http://localhost:%d", server.getLocalPort())),
          url -> ImmutableMap.of());
    }
  }

  @Test
  public void permanentErrorNotFound_doesNotRetryAndThrowsFileNotFoundException() throws Exception {
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 404 Not Found",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 0",
                        "",
                        "");
                  }
                  return null;
                }
              });
      thrown.expect(FileNotFoundException.class);
      thrown.expectMessage("404 Not Found");
      connector.connect(
          new URL(String.format("http://localhost:%d", server.getLocalPort())),
          url -> ImmutableMap.of());
    }
  }

  @Test
  public void permanentError_consumesPayloadBeforeReturningn() throws Exception {
    final CyclicBarrier barrier = new CyclicBarrier(2);
    final AtomicBoolean consumed = new AtomicBoolean();
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 501 Oh No",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 1",
                        "",
                        "b");
                    consumed.set(true);
                  } finally {
                    barrier.await();
                  }
                  return null;
                }
              });
      connector.connect(
          new URL(String.format("http://localhost:%d", server.getLocalPort())),
          url -> ImmutableMap.of());
      fail();
    } catch (IOException ignored) {
      // ignored
    } finally {
      barrier.await();
    }
    assertThat(consumed.get()).isTrue();
    assertThat(clock.currentTimeMillis()).isEqualTo(0L);
  }

  @Test
  public void always500_givesUpEventually() throws Exception {
    final AtomicInteger tries = new AtomicInteger();
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  while (true) {
                    try (Socket socket = server.accept()) {
                      readHttpRequest(socket.getInputStream());
                      sendLines(
                          socket,
                          "HTTP/1.1 500 Oh My",
                          "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                          "Connection: close",
                          "Content-Type: text/plain",
                          "Content-Length: 0",
                          "",
                          "");
                      tries.incrementAndGet();
                    }
                  }
                }
              });
      thrown.expect(IOException.class);
      thrown.expectMessage("500 Oh My");
      try {
        connector.connect(
            new URL(String.format("http://localhost:%d", server.getLocalPort())),
            url -> ImmutableMap.of());
      } finally {
        assertThat(tries.get()).isGreaterThan(2);
      }
    }
  }

  @Test
  public void serverSays403_clientRetriesAnyway() throws Exception {
    final AtomicInteger tries = new AtomicInteger();
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  while (true) {
                    try (Socket socket = server.accept()) {
                      readHttpRequest(socket.getInputStream());
                      sendLines(
                          socket,
                          "HTTP/1.1 403 Forbidden",
                          "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                          "Connection: close",
                          "Content-Type: text/plain",
                          "Content-Length: 0",
                          "",
                          "");
                      tries.incrementAndGet();
                    }
                  }
                }
              });
      thrown.expect(IOException.class);
      thrown.expectMessage("403 Forbidden");
      try {
        connector.connect(
            new URL(String.format("http://localhost:%d", server.getLocalPort())),
            url -> ImmutableMap.of());
      } finally {
        assertThat(tries.get()).isGreaterThan(2);
      }
    }
  }

  @Test
  public void pathRedirect_301() throws Exception {
    redirectToDifferentPath_works("301");
  }

  @Test
  public void serverRedirect_301() throws Exception {
    redirectToDifferentServer_works("301");
  }

  /*
   * Also tests behavior for 302 and 307 codes.
   */
  @Test
  public void pathRedirect_303() throws Exception {
    redirectToDifferentPath_works("303");
  }

  @Test
  public void serverRedirects_303() throws Exception {
    redirectToDifferentServer_works("303");
  }

  public void redirectToDifferentPath_works(String code) throws Exception {
    String redirectCode = "HTTP/1.1 " + code + " Redirect";
    final Map<String, List<String>> headers1 = new ConcurrentHashMap<>();
    final Map<String, List<String>> headers2 = new ConcurrentHashMap<>();
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream(), headers1);
                    sendLines(
                        socket,
                        redirectCode,
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Location: /doodle.tar.gz",
                        "Content-Length: 0",
                        "",
                        "");
                  }
                  try (Socket socket = server.accept()) {
                    readHttpRequest(socket.getInputStream(), headers2);
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 0",
                        "",
                        "");
                  }
                  return null;
                }
              });
      URLConnection connection =
          connector.connect(
              new URL(String.format("http://localhost:%d", server.getLocalPort())),
              url -> ImmutableMap.of());
      assertThat(connection.getURL()).isEqualTo(
          new URL(String.format("http://localhost:%d/doodle.tar.gz", server.getLocalPort())));
      try (InputStream input = connection.getInputStream()) {
        assertThat(ByteStreams.toByteArray(input)).isEmpty();
      }
    }
    assertThat(headers1).containsEntry("x-request-uri", ImmutableList.of("/"));
    assertThat(headers2).containsEntry("x-request-uri", ImmutableList.of("/doodle.tar.gz"));
  }

  public void redirectToDifferentServer_works(String code) throws Exception {
    String redirectCode = "HTTP/1.1 " + code + " Redirect";
    final String basic1 = "Basic b25lOmZpcnN0c2VjcmV0";
    final String basic2 = "Basic dHdvOnNlY29uZHNlY3JldA==";
    final Map<String, List<String>> headers1 = new ConcurrentHashMap<>();
    final Map<String, List<String>> headers2 = new ConcurrentHashMap<>();
    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                  try (Socket socket = server1.accept()) {
                    readHttpRequest(socket.getInputStream(), headers1);
                    sendLines(
                        socket,
                        redirectCode,
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        String.format(
                            "Location: http://localhost:%d/doodle.tar.gz", server2.getLocalPort()),
                        "Content-Length: 0",
                        "",
                        "");
                  }
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
                    readHttpRequest(socket.getInputStream(), headers2);
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "Content-Length: 5",
                        "",
                        "hello");
                  }
                  return null;
                }
              });
      // Header function that provides different auth headers for
      // the two servers.
      Function<URL, ImmutableMap<String, List<String>>> authHeaders =
          new Function<>() {
            @Override
            public ImmutableMap<String, List<String>> apply(URL url) {
              if (url.getPort() == server1.getLocalPort()) {
                return ImmutableMap.of("Authentication", ImmutableList.of(basic1));
              } else if (url.getPort() == server2.getLocalPort()) {
                return ImmutableMap.of("Authentication", ImmutableList.of(basic2));
              } else {
                return ImmutableMap.of();
              }
            }
          };
      URLConnection connection =
          connector.connect(
              new URL(String.format("http://localhost:%d", server1.getLocalPort())), authHeaders);
      assertThat(connection.getURL()).isEqualTo(
          new URL(String.format("http://localhost:%d/doodle.tar.gz", server2.getLocalPort())));
      try (InputStream input = connection.getInputStream()) {
        assertThat(ByteStreams.toByteArray(input)).isEqualTo("hello".getBytes(US_ASCII));
      }
      // Verify that the correct form of authentication is used for each server.
      assertThat(headers1).containsEntry("authentication", ImmutableList.of(basic1));
      assertThat(headers2).containsEntry("authentication", ImmutableList.of(basic2));
    }
  }

  private File createTempFile(byte[] fileContents) throws IOException {
    File temp = testFolder.newFile();
    try (FileOutputStream outputStream = new FileOutputStream(temp)) {
      outputStream.write(fileContents);
    }
    return temp;
  }
}
