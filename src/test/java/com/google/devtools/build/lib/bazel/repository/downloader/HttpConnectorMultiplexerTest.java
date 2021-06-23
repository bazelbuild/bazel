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
import static com.google.devtools.build.lib.bazel.repository.downloader.DownloaderTestUtils.makeUrl;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.verifyZeroInteractions;
import static org.mockito.Mockito.when;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.RetryingInputStream.Reconnector;
import com.google.devtools.build.lib.events.EventHandler;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link HttpConnectorMultiplexer}. */
@RunWith(JUnit4.class)
@SuppressWarnings("unchecked")
public class HttpConnectorMultiplexerTest {

  private static final URL TEST_URL = makeUrl("http://test.example");
  private static final byte[] TEST_DATA = "test_data".getBytes(UTF_8);

  private static Optional<Checksum> makeChecksum(String string) {
    try {
      return Optional.of(Checksum.fromString(KeyType.SHA256, string));
    } catch (Checksum.InvalidChecksumException e) {
      throw new IllegalStateException(e);
    }
  }

  private static final Optional<Checksum> DUMMY_CHECKSUM =
      makeChecksum("abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd");

  @Rule
  public final Timeout globalTimeout = new Timeout(10000);

  private final HttpStream stream = new HttpStream(new ByteArrayInputStream(TEST_DATA), TEST_URL);
  private final HttpConnector connector = mock(HttpConnector.class);
  private final URLConnection connection = mock(URLConnection.class);
  private final EventHandler eventHandler = mock(EventHandler.class);
  private final HttpStream.Factory streamFactory = mock(HttpStream.Factory.class);
  private final HttpConnectorMultiplexer multiplexer =
      new HttpConnectorMultiplexer(eventHandler, connector, streamFactory);

  @Before
  public void before() throws Exception {
    when(connector.connect(eq(TEST_URL), any(Function.class))).thenReturn(connection);
    when(streamFactory.create(
            same(connection),
            any(URL.class),
            any(Optional.class),
            any(Reconnector.class),
            any(Optional.class)))
        .thenReturn(stream);
  }

  @Test
  public void ftpUrl_throwsIae() throws Exception {
    assertThrows(
        IllegalArgumentException.class,
        () -> multiplexer.connect(new URL("ftp://lol.example"), Optional.absent()));
  }

  @Test
  public void threadIsInterrupted_throwsIeProntoAndDoesNothingElse() throws Exception {
    final AtomicBoolean wasInterrupted = new AtomicBoolean(true);
    Thread task =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                Thread.currentThread().interrupt();
                try {
                  multiplexer.connect(new URL("http://lol.example"), Optional.absent());
                } catch (InterruptedIOException ignored) {
                  return;
                } catch (Exception ignored) {
                  // ignored
                }
                wasInterrupted.set(false);
              }
            });
    task.start();
    task.join();
    assertThat(wasInterrupted.get()).isTrue();
    verifyZeroInteractions(connector);
  }

  @Test
  public void success() throws Exception {
    assertThat(toByteArray(multiplexer.connect(TEST_URL, DUMMY_CHECKSUM))).isEqualTo(TEST_DATA);
    verify(connector).connect(eq(TEST_URL), any(Function.class));
    verify(streamFactory)
        .create(
            any(URLConnection.class),
            any(URL.class),
            eq(DUMMY_CHECKSUM),
            any(Reconnector.class),
            any(Optional.class));
    verifyNoMoreInteractions(connector, streamFactory);
  }

  @Test
  public void failure() throws Exception {
    when(connector.connect(any(URL.class), any(Function.class))).thenThrow(new IOException("oops"));
    IOException e =
        assertThrows(IOException.class, () -> multiplexer.connect(TEST_URL, Optional.absent()));
    assertThat(e).hasMessageThat().contains("oops");
    verify(connector).connect(any(URL.class), any(Function.class));
    verifyNoMoreInteractions(connector, streamFactory);
  }

  @Test
  public void testHeaderComputationFunction() throws Exception {
    Map<String, String> baseHeaders =
        ImmutableMap.of("Accept-Encoding", "gzip", "User-Agent", "Bazel/testing");
    Map<URI, Map<String, String>> additionalHeaders =
        ImmutableMap.of(
            new URI("http://hosting.example.com/user/foo/file.txt"),
            ImmutableMap.of("Authentication", "Zm9vOmZvb3NlY3JldA=="));

    Function<URL, ImmutableMap<String, String>> headerFunction =
        HttpConnectorMultiplexer.getHeaderFunction(baseHeaders, additionalHeaders);

    // Unreleated URL
    assertThat(headerFunction.apply(new URL("http://example.org/some/path/file.txt")))
        .containsExactly("Accept-Encoding", "gzip", "User-Agent", "Bazel/testing");

    // With auth headers
    assertThat(headerFunction.apply(new URL("http://hosting.example.com/user/foo/file.txt")))
        .containsExactly(
            "Accept-Encoding",
            "gzip",
            "User-Agent",
            "Bazel/testing",
            "Authentication",
            "Zm9vOmZvb3NlY3JldA==");

    // Other hosts
    assertThat(headerFunction.apply(new URL("http://hosting2.example.com/user/foo/file.txt")))
        .containsExactly("Accept-Encoding", "gzip", "User-Agent", "Bazel/testing");
    assertThat(headerFunction.apply(new URL("http://sub.hosting.example.com/user/foo/file.txt")))
        .containsExactly("Accept-Encoding", "gzip", "User-Agent", "Bazel/testing");
    assertThat(headerFunction.apply(new URL("http://example.com/user/foo/file.txt")))
        .containsExactly("Accept-Encoding", "gzip", "User-Agent", "Bazel/testing");
    assertThat(
            headerFunction.apply(
                new URL("http://hosting.example.com.evil.example/user/foo/file.txt")))
        .containsExactly("Accept-Encoding", "gzip", "User-Agent", "Bazel/testing");

    // Verify that URL-specific headers overwrite
    Map<String, String> annonAuth =
        ImmutableMap.of("Authentication", "YW5vbnltb3VzOmZvb0BleGFtcGxlLm9yZw==");
    Function<URL, ImmutableMap<String, String>> combinedHeaders =
        HttpConnectorMultiplexer.getHeaderFunction(annonAuth, additionalHeaders);
    assertThat(combinedHeaders.apply(new URL("http://hosting.example.com/user/foo/file.txt")))
        .containsExactly("Authentication", "Zm9vOmZvb3NlY3JldA==");
    assertThat(combinedHeaders.apply(new URL("http://unreleated.example.org/user/foo/file.txt")))
        .containsExactly("Authentication", "YW5vbnltb3VzOmZvb0BleGFtcGxlLm9yZw==");
  }
}
