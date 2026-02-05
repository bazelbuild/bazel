// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.auth.Credentials;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Phaser;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DownloadManagerTest {

  @Rule public final TemporaryFolder workingDir = new TemporaryFolder();

  private final ExecutorService executor = Executors.newFixedThreadPool(1);

  @After
  public void after() {
    executor.shutdown();
  }

  private static DownloadManager newDownloadManagerForTest(Downloader downloader) {
    RepositoryCache repositoryCache = mock(RepositoryCache.class);
    when(repositoryCache.isEnabled()).thenReturn(false);
    HttpDownloader bzlmodHttpDownloader = mock(HttpDownloader.class);
    return new DownloadManager(repositoryCache, downloader, bzlmodHttpDownloader);
  }

  @Test
  public void retriesOnGenericIOException() throws Exception {
    AtomicInteger attempts = new AtomicInteger();
    Downloader throwingDownloader =
        new Downloader() {
          @Override
          public void download(
              List<URL> urls,
              Map<String, List<String>> headers,
              Credentials credentials,
              Optional<Checksum> checksum,
              String canonicalId,
              Path output,
              ExtendedEventHandler eventHandler,
              Map<String, String> clientEnv,
              Optional<String> type)
              throws IOException {
            attempts.incrementAndGet();
            throw new IOException("boom");
          }
        };

    DownloadManager downloadManager = newDownloadManagerForTest(throwingDownloader);
    downloadManager.setRetries(2);

    JavaIoFileSystem fs = new JavaIoFileSystem(DigestHashFunction.SHA256);
    Path out = fs.getPath(workingDir.newFile().getAbsolutePath());

    Future<Path> f =
        downloadManager.startDownload(
            executor,
            ImmutableList.of(new URL("http://example.invalid/file")),
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty(),
            "canonical",
            Optional.empty(),
            out,
            /* eventHandler= */ mock(ExtendedEventHandler.class),
            ImmutableMap.of(),
            "ctx",
            new Phaser());

    assertThrows(IOException.class, () -> downloadManager.finalizeDownload(f));
    assertThat(attempts.get()).isEqualTo(3); // 1 initial + 2 retries
  }

  @Test
  public void doesNotRetryOnUnrecoverableHttpException() throws Exception {
    AtomicInteger attempts = new AtomicInteger();
    Downloader throwingDownloader =
        new Downloader() {
          @Override
          public void download(
              List<URL> urls,
              Map<String, List<String>> headers,
              Credentials credentials,
              Optional<Checksum> checksum,
              String canonicalId,
              Path output,
              ExtendedEventHandler eventHandler,
              Map<String, String> clientEnv,
              Optional<String> type)
              throws IOException {
            attempts.incrementAndGet();
            throw new UnrecoverableHttpException("nope");
          }
        };

    DownloadManager downloadManager = newDownloadManagerForTest(throwingDownloader);
    downloadManager.setRetries(5);

    JavaIoFileSystem fs = new JavaIoFileSystem(DigestHashFunction.SHA256);
    Path out = fs.getPath(workingDir.newFile().getAbsolutePath());

    Future<Path> f =
        downloadManager.startDownload(
            executor,
            ImmutableList.of(new URL("http://example.invalid/file")),
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty(),
            "canonical",
            Optional.empty(),
            out,
            /* eventHandler= */ mock(ExtendedEventHandler.class),
            ImmutableMap.of(),
            "ctx",
            new Phaser());

    assertThrows(IOException.class, () -> downloadManager.finalizeDownload(f));
    assertThat(attempts.get()).isEqualTo(1);
  }

  @Test
  public void doesNotRetryOnFileNotFoundException() throws Exception {
    AtomicInteger attempts = new AtomicInteger();
    Downloader throwingDownloader =
        new Downloader() {
          @Override
          public void download(
              List<URL> urls,
              Map<String, List<String>> headers,
              Credentials credentials,
              Optional<Checksum> checksum,
              String canonicalId,
              Path output,
              ExtendedEventHandler eventHandler,
              Map<String, String> clientEnv,
              Optional<String> type)
              throws IOException {
            attempts.incrementAndGet();
            throw new FileNotFoundException("missing");
          }
        };

    DownloadManager downloadManager = newDownloadManagerForTest(throwingDownloader);
    downloadManager.setRetries(5);

    JavaIoFileSystem fs = new JavaIoFileSystem(DigestHashFunction.SHA256);
    Path out = fs.getPath(workingDir.newFile().getAbsolutePath());

    Future<Path> f =
        downloadManager.startDownload(
            executor,
            ImmutableList.of(new URL("http://example.invalid/file")),
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty(),
            "canonical",
            Optional.empty(),
            out,
            /* eventHandler= */ mock(ExtendedEventHandler.class),
            ImmutableMap.of(),
            "ctx",
            new Phaser());

    assertThrows(IOException.class, () -> downloadManager.finalizeDownload(f));
    assertThat(attempts.get()).isEqualTo(1);
  }
}

