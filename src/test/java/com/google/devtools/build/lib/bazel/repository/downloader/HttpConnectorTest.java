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
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.io.ByteSource;
import com.google.common.io.ByteStreams;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.events.EventHandler;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.Proxy;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URL;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link HttpConnector}.
 */
@RunWith(JUnit4.class)
public class HttpConnectorTest {

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Rule
  public TemporaryFolder testFolder = new TemporaryFolder();

  private final ExecutorService executor = Executors.newSingleThreadExecutor();
  private final HttpURLConnection connection = mock(HttpURLConnection.class);
  private final EventHandler eventHandler = mock(EventHandler.class);

  @After
  public void after() throws Exception {
    executor.shutdownNow();
  }

  @Test
  public void testLocalFileDownload() throws Exception {
    byte[] fileContents = "this is a test".getBytes(UTF_8);
    assertThat(
            ByteStreams.toByteArray(
                HttpConnector.connect(
                    createTempFile(fileContents).toURI().toURL(),
                    Proxy.NO_PROXY,
                    eventHandler)))
        .isEqualTo(fileContents);
  }

  @Test
  public void missingLocationInRedirect_throwsIOException() throws Exception {
    thrown.expect(IOException.class);
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    HttpConnector.getLocation(connection);
  }

  @Test
  public void absoluteLocationInRedirect_returnsNewUrl() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi");
    assertThat(HttpConnector.getLocation(connection)).isEqualTo(new URL("http://new.example/hi"));
  }

  @Test
  public void redirectOnlyHasPath_mergesHostFromOriginalUrl() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("/hi");
    assertThat(HttpConnector.getLocation(connection)).isEqualTo(new URL("http://lol.example/hi"));
  }

  @Test
  public void locationOnlyHasPathWithoutSlash_failsToMerge() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Could not merge");
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("omg");
    HttpConnector.getLocation(connection);
  }

  @Test
  public void locationHasFragment_prefersNewFragment() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example#a"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi#b");
    assertThat(HttpConnector.getLocation(connection)).isEqualTo(new URL("http://new.example/hi#b"));
  }

  @Test
  public void locationHasNoFragmentButOriginalDoes_mergesOldFragment() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example#a"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi");
    assertThat(HttpConnector.getLocation(connection)).isEqualTo(new URL("http://new.example/hi#a"));
  }

  @Test
  public void oldUrlHasPasswordRedirectingToSameDomain_mergesPassword() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://lol.example/hi");
    assertThat(HttpConnector.getLocation(connection))
        .isEqualTo(new URL("http://a:b@lol.example/hi"));
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("/hi");
    assertThat(HttpConnector.getLocation(connection))
        .isEqualTo(new URL("http://a:b@lol.example/hi"));
  }

  @Test
  public void oldUrlHasPasswordRedirectingToNewServer_doesntMergePassword() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://new.example/hi");
    assertThat(HttpConnector.getLocation(connection)).isEqualTo(new URL("http://new.example/hi"));
    when(connection.getURL()).thenReturn(new URL("http://a:b@lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("http://lol.example:81/hi");
    assertThat(HttpConnector.getLocation(connection))
        .isEqualTo(new URL("http://lol.example:81/hi"));
  }

  @Test
  public void redirectToFtp_throwsIOException() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Bad Location");
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("ftp://lol.example");
    HttpConnector.getLocation(connection);
  }

  @Test
  public void redirectToHttps_works() throws Exception {
    when(connection.getURL()).thenReturn(new URL("http://lol.example"));
    when(connection.getHeaderField("Location")).thenReturn("https://lol.example");
    assertThat(HttpConnector.getLocation(connection)).isEqualTo(new URL("https://lol.example"));
  }

  @Test
  public void testNormalRequest() throws Exception {
    try (final ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
      Future<Void> thread =
          executor.submit(
              new Callable<Void>() {
                @Override
                public Void call() throws Exception {
                  try (Socket socket = server.accept()) {
                    send(socket,
                        "HTTP/1.1 200 OK\r\n"
                            + "Date: Fri, 31 Dec 1999 23:59:59 GMT\r\n"
                            + "Content-Type: text/plain\r\n"
                            + "Content-Length: 5\r\n"
                            + "\r\n"
                            + "hello");
                  }
                  return null;
                }
              });
      try (Reader payload =
              new InputStreamReader(
                  HttpConnector.connect(
                      new URL(String.format("http://127.0.0.1:%d", server.getLocalPort())),
                      Proxy.NO_PROXY,
                      eventHandler),
                  ISO_8859_1)) {
        assertThat(CharStreams.toString(payload)).isEqualTo("hello");
      }
      thread.get();
    }
  }

  @Test
  public void testRetry() throws Exception {
    try (final ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
      Future<Void> thread =
          executor.submit(
              new Callable<Void>() {
                @Override
                public Void call() throws Exception {
                  try (Socket socket = server.accept()) {
                    send(socket,
                        "HTTP/1.1 500 Incredible Catastrophe\r\n"
                            + "Date: Fri, 31 Dec 1999 23:59:59 GMT\r\n"
                            + "Content-Type: text/plain\r\n"
                            + "Content-Length: 8\r\n"
                            + "\r\n"
                            + "nononono");
                  }
                  try (Socket socket = server.accept()) {
                    send(socket,
                        "HTTP/1.1 200 OK\r\n"
                            + "Date: Fri, 31 Dec 1999 23:59:59 GMT\r\n"
                            + "Content-Type: text/plain\r\n"
                            + "Content-Length: 5\r\n"
                            + "\r\n"
                            + "hello");
                  }
                  return null;
                }
              });
      try (Reader payload =
              new InputStreamReader(
                  HttpConnector.connect(
                      new URL(String.format("http://127.0.0.1:%d", server.getLocalPort())),
                      Proxy.NO_PROXY,
                      eventHandler),
                  ISO_8859_1)) {
        assertThat(CharStreams.toString(payload)).isEqualTo("hello");
      }
      thread.get();
    }
  }

  private static void send(Socket socket, String data) throws IOException {
    ByteStreams.copy(
        ByteSource.wrap(data.getBytes(ISO_8859_1)).openStream(),
        socket.getOutputStream());
  }

  private File createTempFile(byte[] fileContents) throws IOException {
    File temp = testFolder.newFile();
    try (FileOutputStream outputStream = new FileOutputStream(temp)) {
      outputStream.write(fileContents);
    }
    return temp;
  }
}
