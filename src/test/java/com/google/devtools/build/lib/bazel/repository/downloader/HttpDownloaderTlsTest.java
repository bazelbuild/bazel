package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.repository.downloader.DownloaderTestUtils.sendLines;
import static com.google.devtools.build.lib.bazel.repository.downloader.HttpParser.readHttpRequest;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.authandtls.StaticCredentials;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URL;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
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

@RunWith(JUnit4.class)
public class HttpDownloaderTlsTest {

  @Rule public final TemporaryFolder workingDir = new TemporaryFolder();
  @Rule public final Timeout timeout = new Timeout(30, SECONDS);

  private final ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);
  // Scale timeouts down to make test fast.
  private final HttpDownloader httpDownloader = new HttpDownloader(0, Duration.ZERO, 8, .1f);
  private final ExecutorService executor = Executors.newFixedThreadPool(2);
  private final JavaIoFileSystem fs;

  public HttpDownloaderTlsTest() {
    fs = new JavaIoFileSystem(DigestHashFunction.SHA256);
  }

  @After
  public void after() {
    executor.shutdown();
  }

  @Test
  public void downloadFrom2UrlsFirstTlsErrorSecondOk() throws IOException, InterruptedException {
    try (ServerSocket server1 = new ServerSocket(0, 1, InetAddress.getByName(null));
        ServerSocket server2 = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executor.submit(
              () -> {
                // Determine which port was assigned
                try (Socket socket = server1.accept()) {
                   // Write garbage to trigger SSL handshake failure on client
                   socket.getOutputStream().write("Not SSL".getBytes(UTF_8));
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
      // Use https for the first one to trigger SSL handshake
      urls.add(new URL(String.format("https://localhost:%d/foo", server1.getLocalPort())));
      urls.add(new URL(String.format("http://localhost:%d/foo", server2.getLocalPort())));

      Path resultingFile = fs.getPath(workingDir.newFile().getAbsolutePath());
      
      try {
          httpDownloader.download(
              urls,
              Collections.emptyMap(),
              StaticCredentials.EMPTY,
              Optional.empty(),
              "testCanonicalId",
              resultingFile,
              eventHandler,
              Collections.emptyMap(),
              Optional.empty(),
              "testRepo");
      } catch (IOException e) {
          // If the bug exists, this will likely throw an exception wrapping SSLHandshakeException
          // instead of failing over to the second URL.
          // We will print the stack trace for debugging purposes.
          e.printStackTrace();
          throw e; // Rethrow to fail the test
      }

      assertThat(new String(readFile(resultingFile), UTF_8)).isEqualTo("content2");
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
