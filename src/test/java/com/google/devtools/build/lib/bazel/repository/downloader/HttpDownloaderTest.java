// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DefaultHashFunctionNotSetException;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.Timeout;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link HttpDownloader} */
@RunWith(JUnit4.class)
public class HttpDownloaderTest {

  @Rule public final TemporaryFolder workingDir = new TemporaryFolder();

  @Rule public final Timeout timeout = new Timeout(30, SECONDS);

  private final RepositoryCache repositoryCache = mock(RepositoryCache.class);
  private final HttpDownloader httpDownloader = new HttpDownloader();
  private final DownloadManager downloadManager =
      new DownloadManager(repositoryCache, httpDownloader);

  private final ExecutorService executor = Executors.newFixedThreadPool(2);
  private final ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);
  private final JavaIoFileSystem fs;

  public HttpDownloaderTest() throws DefaultHashFunctionNotSetException {
    try {
      DigestHashFunction.setDefault(DigestHashFunction.SHA256);
    } catch (DigestHashFunction.DefaultAlreadySetException e) {
      // Do nothing.
    }
    fs = new JavaIoFileSystem();

    // Scale timeouts down to make tests fast.
    httpDownloader.setTimeoutScaling(0.1f);
  }

  @After
  public void after() {
    executor.shutdown();
  }

  @Test
  public void downloadFrom1UrlOk() throws IOException, InterruptedException {
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
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
              });

      Path resultingFile =
          downloadManager.download(
              Collections.singletonList(
                  new URL(String.format("http://localhost:%d/foo", server.getLocalPort()))),
              Collections.emptyMap(),
              Optional.absent(),
              "testCanonicalId",
              Optional.absent(),
              fs.getPath(workingDir.newFile().getAbsolutePath()),
              eventHandler,
              Collections.emptyMap(),
              "testRepo");

      assertThat(new String(readFile(resultingFile), UTF_8)).isEqualTo("hello");
    }
  }

  @Test
  public void downloadFrom2UrlsFirstOk() throws IOException, InterruptedException {
    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
                while (!executor.isShutdown()) {
                  try (Socket socket = server1.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "",
                        "content1");
                  }
                }
                return null;
              });

      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError2 =
          executor.submit(
              () -> {
                while (!executor.isShutdown()) {
                  try (Socket socket = server2.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "",
                        "content2");
                  }
                }
                return null;
              });

      final List<URL> urls = new ArrayList<>(2);
      urls.add(new URL(String.format("http://localhost:%d/foo", server1.getLocalPort())));
      urls.add(new URL(String.format("http://localhost:%d/foo", server2.getLocalPort())));

      Path resultingFile =
          downloadManager.download(
              urls,
              Collections.emptyMap(),
              Optional.absent(),
              "testCanonicalId",
              Optional.absent(),
              fs.getPath(workingDir.newFile().getAbsolutePath()),
              eventHandler,
              Collections.emptyMap(),
              "testRepo");

      assertThat(new String(readFile(resultingFile), UTF_8)).isEqualTo("content1");
    }
  }

  @Test
  public void downloadFrom2UrlsFirstSocketTimeoutOnBodyReadSecondOk()
      throws IOException, InterruptedException {
    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
                Socket socket = server1.accept();
                readHttpRequest(socket.getInputStream());

                sendLines(
                    socket,
                    "HTTP/1.1 200 OK",
                    "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                    "Connection: close",
                    "Content-Type: text/plain",
                    "",
                    "content1");

                // Never close the socket to cause SocketTimeoutException during body read on client
                // side.
                return null;
              });

      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError2 =
          executor.submit(
              () -> {
                while (!executor.isShutdown()) {
                  try (Socket socket = server2.accept()) {
                    readHttpRequest(socket.getInputStream());
                    sendLines(
                        socket,
                        "HTTP/1.1 200 OK",
                        "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                        "Connection: close",
                        "Content-Type: text/plain",
                        "",
                        "content2");
                  }
                }
                return null;
              });

      final List<URL> urls = new ArrayList<>(2);
      urls.add(new URL(String.format("http://localhost:%d/foo", server1.getLocalPort())));
      urls.add(new URL(String.format("http://localhost:%d/foo", server2.getLocalPort())));

      Path resultingFile =
          downloadManager.download(
              urls,
              Collections.emptyMap(),
              Optional.absent(),
              "testCanonicalId",
              Optional.absent(),
              fs.getPath(workingDir.newFile().getAbsolutePath()),
              eventHandler,
              Collections.emptyMap(),
              "testRepo");

      assertThat(new String(readFile(resultingFile), UTF_8)).isEqualTo("content2");
    }
  }

  @Test
  public void downloadFrom2UrlsBothSocketTimeoutDuringBodyRead()
      throws IOException, InterruptedException {
    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
                Socket socket = server1.accept();
                readHttpRequest(socket.getInputStream());

                sendLines(
                    socket,
                    "HTTP/1.1 200 OK",
                    "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                    "Connection: close",
                    "Content-Type: text/plain",
                    "",
                    "content1");

                // Never close the socket to cause SocketTimeoutException during body read on client
                // side.
                return null;
              });

      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError2 =
          executor.submit(
              () -> {
                Socket socket = server1.accept();
                readHttpRequest(socket.getInputStream());

                sendLines(
                    socket,
                    "HTTP/1.1 200 OK",
                    "Date: Fri, 31 Dec 1999 23:59:59 GMT",
                    "Connection: close",
                    "Content-Type: text/plain",
                    "",
                    "content2");

                // Never close the socket to cause SocketTimeoutException during body read on client
                // side.
                return null;
              });

      final List<URL> urls = new ArrayList<>(2);
      urls.add(new URL(String.format("http://localhost:%d/foo", server1.getLocalPort())));
      urls.add(new URL(String.format("http://localhost:%d/foo", server2.getLocalPort())));

      Path outputFile = fs.getPath(workingDir.newFile().getAbsolutePath());
      try {
        downloadManager.download(
            urls,
            Collections.emptyMap(),
            Optional.absent(),
            "testCanonicalId",
            Optional.absent(),
            outputFile,
            eventHandler,
            Collections.emptyMap(),
            "testRepo");
        fail("Should have thrown");
      } catch (IOException expected) {
        assertThat(expected.getSuppressed()).hasLength(2);

        for (Throwable suppressed : expected.getSuppressed()) {
          assertThat(suppressed).isInstanceOf(IOException.class);
          assertThat(suppressed).hasCauseThat().isInstanceOf(SocketTimeoutException.class);
        }
      }
    }
  }

  private static byte[] readFile(Path path) throws IOException {
    final byte[] data = new byte[(int) path.getFileSize()];

    try (DataInputStream stream = new DataInputStream(path.getInputStream())) {
      stream.readFully(data);
    }

    return data;
  }
}
