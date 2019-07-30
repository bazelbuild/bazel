package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.devtools.build.lib.bazel.repository.downloader.DownloaderTestUtils.sendLines;
import static com.google.devtools.build.lib.bazel.repository.downloader.HttpParser.readHttpRequest;
import static org.mockito.Mockito.mock;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URL;
import java.util.Collections;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class HttpDownloaderTest {

  @Rule
  public final TemporaryFolder workingDir = new TemporaryFolder();

  private final RepositoryCache repositoryCache = mock(RepositoryCache.class);
  private final HttpDownloader httpDownloader = new HttpDownloader(repositoryCache);

  private final ExecutorService executor = Executors.newFixedThreadPool(1);
  private final ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);

  @Test
  public void name() {

  }

  @Test
  public void downloadFrom1UrlOk() throws IOException, InterruptedException {
    try (ServerSocket server = new ServerSocket(0, 1, InetAddress.getByName(null))) {
      Future<?> possiblyIgnoredError =
          executor.submit(
              new Callable<Object>() {
                @Override
                public Object call() throws Exception {
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

      Path outputFile = Path.getFileSystemForSerialization()
          .getPath(workingDir.newFile().getAbsolutePath());

      httpDownloader.download(
          Collections.singletonList(
              new URL(String.format("http://localhost:%d/foo", server.getLocalPort()))),
          Collections.emptyMap(),
          Optional.absent(),
          "testCanonicalId",
          Optional.absent(),
          outputFile,
          eventHandler,
          Collections.emptyMap(),
          "testRepo"
      );
    }

  }
}
