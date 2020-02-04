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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Arrays.asList;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.verifyZeroInteractions;
import static org.mockito.Mockito.when;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.RetryingInputStream.Reconnector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.Sleeper;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.Timeout;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Unit tests for {@link HttpConnectorMultiplexer}. */
@RunWith(JUnit4.class)
@SuppressWarnings("unchecked")
public class HttpConnectorMultiplexerTest {

  private static final URL URL1 = makeUrl("http://first.example");
  private static final URL URL2 = makeUrl("http://second.example");
  private static final URL URL3 = makeUrl("http://third.example");
  private static final byte[] data1 = "first".getBytes(UTF_8);
  private static final byte[] data2 = "second".getBytes(UTF_8);
  private static final byte[] data3 = "third".getBytes(UTF_8);

  private static final Optional<Checksum> DUMMY_CHECKSUM =
      Optional.of(
          Checksum.fromString(
              KeyType.SHA256, "abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd"));

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Rule
  public final Timeout globalTimeout = new Timeout(10000);

  private final HttpStream stream1 = fakeStream(URL1, data1);
  private final HttpStream stream2 = fakeStream(URL2, data2);
  private final HttpStream stream3 = fakeStream(URL3, data3);
  private final ManualClock clock = new ManualClock();
  private final Sleeper sleeper = mock(Sleeper.class);
  private final HttpConnector connector = mock(HttpConnector.class);
  private final URLConnection connection1 = mock(URLConnection.class);
  private final URLConnection connection2 = mock(URLConnection.class);
  private final URLConnection connection3 = mock(URLConnection.class);
  private final EventHandler eventHandler = mock(EventHandler.class);
  private final HttpStream.Factory streamFactory = mock(HttpStream.Factory.class);
  private final HttpConnectorMultiplexer multiplexer =
      new HttpConnectorMultiplexer(eventHandler, connector, streamFactory, clock, sleeper);

  @Before
  public void before() throws Exception {
    when(connector.connect(eq(URL1), any(Function.class))).thenReturn(connection1);
    when(connector.connect(eq(URL2), any(Function.class))).thenReturn(connection2);
    when(connector.connect(eq(URL3), any(Function.class))).thenReturn(connection3);
    when(streamFactory.create(
            same(connection1), any(URL.class), any(Optional.class), any(Reconnector.class)))
        .thenReturn(stream1);
    when(streamFactory.create(
            same(connection2), any(URL.class), any(Optional.class), any(Reconnector.class)))
        .thenReturn(stream2);
    when(streamFactory.create(
            same(connection3), any(URL.class), any(Optional.class), any(Reconnector.class)))
        .thenReturn(stream3);
  }

  @Test
  public void emptyList_throwsIae() throws Exception {
    thrown.expect(IllegalArgumentException.class);
    multiplexer.connect(ImmutableList.<URL>of(), null);
  }

  @Test
  public void ftpUrl_throwsIae() throws Exception {
    thrown.expect(IllegalArgumentException.class);
    multiplexer.connect(asList(new URL("ftp://lol.example")), null);
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
                  multiplexer.connect(asList(new URL("http://lol.example")), null);
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
  public void singleUrl_justCallsConnector() throws Exception {
    assertThat(toByteArray(multiplexer.connect(asList(URL1), DUMMY_CHECKSUM))).isEqualTo(data1);
    verify(connector).connect(eq(URL1), any(Function.class));
    verify(streamFactory)
        .create(
            any(URLConnection.class), any(URL.class), eq(DUMMY_CHECKSUM), any(Reconnector.class));
    verifyNoMoreInteractions(sleeper, connector, streamFactory);
  }

  @Test
  public void multipleUrlsFail_throwsIOException() throws Exception {
    when(connector.connect(any(URL.class), any(Function.class))).thenThrow(new IOException());
    IOException e =
        assertThrows(IOException.class, () -> multiplexer.connect(asList(URL1, URL2, URL3), null));
    assertThat(e).hasMessageThat().contains("All mirrors are down");
    verify(connector, times(3)).connect(any(URL.class), any(Function.class));
    verify(sleeper, times(2)).sleepMillis(anyLong());
    verifyNoMoreInteractions(sleeper, connector, streamFactory);
  }

  @Test
  public void firstUrlFails_returnsSecond() throws Exception {
    doAnswer(
        new Answer<Void>() {
          @Override
          public Void answer(InvocationOnMock invocation) throws Throwable {
            clock.advanceMillis(1000);
            return null;
          }
        }).when(sleeper).sleepMillis(anyLong());
    when(connector.connect(eq(URL1), any(Function.class))).thenThrow(new IOException());
    assertThat(toByteArray(multiplexer.connect(asList(URL1, URL2), DUMMY_CHECKSUM)))
        .isEqualTo(data2);
    assertThat(clock.currentTimeMillis()).isEqualTo(1000L);
    verify(connector).connect(eq(URL1), any(Function.class));
    verify(connector).connect(eq(URL2), any(Function.class));
    verify(streamFactory)
        .create(
            any(URLConnection.class), any(URL.class), eq(DUMMY_CHECKSUM), any(Reconnector.class));
    verify(sleeper).sleepMillis(anyLong());
    verifyNoMoreInteractions(sleeper, connector, streamFactory);
  }

  @Test
  public void twoSuccessfulUrlsAndFirstWins_returnsFirstAndInterruptsSecond() throws Exception {
    final CyclicBarrier barrier = new CyclicBarrier(2);
    final AtomicBoolean wasInterrupted = new AtomicBoolean(true);
    when(connector.connect(eq(URL1), any(Function.class)))
        .thenAnswer(
            new Answer<URLConnection>() {
              @Override
              public URLConnection answer(InvocationOnMock invocation) throws Throwable {
                barrier.await();
                return connection1;
              }
            });
    doAnswer(
        new Answer<Void>() {
          @Override
          public Void answer(InvocationOnMock invocation) throws Throwable {
            barrier.await();
            TimeUnit.MILLISECONDS.sleep(10000);
            wasInterrupted.set(false);
            return null;
          }
        }).when(sleeper).sleepMillis(anyLong());
    assertThat(toByteArray(multiplexer.connect(asList(URL1, URL2), DUMMY_CHECKSUM)))
        .isEqualTo(data1);
    assertThat(wasInterrupted.get()).isTrue();
  }

  @Test
  public void parentThreadGetsInterrupted_interruptsChildrenThenThrowsIe() throws Exception {
    final CyclicBarrier barrier = new CyclicBarrier(3);
    final AtomicBoolean wasInterrupted1 = new AtomicBoolean(true);
    final AtomicBoolean wasInterrupted2 = new AtomicBoolean(true);
    final AtomicBoolean wasInterrupted3 = new AtomicBoolean(true);
    when(connector.connect(eq(URL1), any(Function.class)))
        .thenAnswer(
            new Answer<URLConnection>() {
              @Override
              public URLConnection answer(InvocationOnMock invocation) throws Throwable {
                barrier.await();
                TimeUnit.MILLISECONDS.sleep(10000);
                wasInterrupted1.set(false);
                throw new RuntimeException();
              }
            });
    when(connector.connect(eq(URL2), any(Function.class)))
        .thenAnswer(
            new Answer<URLConnection>() {
              @Override
              public URLConnection answer(InvocationOnMock invocation) throws Throwable {
                barrier.await();
                TimeUnit.MILLISECONDS.sleep(10000);
                wasInterrupted2.set(false);
                throw new RuntimeException();
              }
            });
    Thread task =
        new Thread(
            new Runnable() {
              @Override
              public void run() {
                try {
                  multiplexer.connect(asList(URL1, URL2), null);
                } catch (InterruptedIOException ignored) {
                  return;
                } catch (Exception ignored) {
                  // ignored
                }
                wasInterrupted3.set(false);
              }
            });
    task.start();
    barrier.await();
    task.interrupt();
    task.join();
    assertThat(wasInterrupted1.get()).isTrue();
    assertThat(wasInterrupted2.get()).isTrue();
    assertThat(wasInterrupted3.get()).isTrue();
  }

  private static HttpStream fakeStream(URL url, byte[] data) {
    return new HttpStream(new ByteArrayInputStream(data), url);
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
