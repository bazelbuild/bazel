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
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.hash.Hashing;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.bazel.repository.downloader.RetryingInputStream.Reconnector;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipException;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.Timeout;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Integration tests for {@link HttpStream.Factory} and friends. */
@RunWith(JUnit4.class)
public class HttpStreamTest {

  private static final Random randoCalrissian = new Random();
  private static final byte[] data = "hello".getBytes(UTF_8);
  private static final String GOOD_CHECKSUM =
      "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
  private static final String BAD_CHECKSUM =
      "0000000000000000000000000000000000000000000000000000000000000000";
  private static final URL AURL = makeUrl("http://doodle.example");

  @Rule
  public final ExpectedException thrown = ExpectedException.none();

  @Rule
  public final Timeout globalTimeout = new Timeout(10000);

  private final HttpURLConnection connection = mock(HttpURLConnection.class);
  private final Reconnector reconnector = mock(Reconnector.class);
  private final ProgressInputStream.Factory progress = mock(ProgressInputStream.Factory.class);
  private final HttpStream.Factory streamFactory = new HttpStream.Factory(progress);

  @Before
  public void before() throws Exception {
    when(connection.getInputStream()).thenReturn(new ByteArrayInputStream(data));
    when(progress.create(any(InputStream.class), any(), any(URL.class))).thenAnswer(
        new Answer<InputStream>() {
          @Override
          public InputStream answer(InvocationOnMock invocation) throws Throwable {
            return (InputStream) invocation.getArguments()[0];
          }
        });
  }

  @Test
  public void noChecksum_readsOk() throws Exception {
    try (HttpStream stream = streamFactory.create(connection, AURL, "", reconnector)) {
      assertThat(toByteArray(stream)).isEqualTo(data);
    }
  }

  @Test
  public void smallDataWithValidChecksum_readsOk() throws Exception {
    try (HttpStream stream = streamFactory.create(connection, AURL, GOOD_CHECKSUM, reconnector)) {
      assertThat(toByteArray(stream)).isEqualTo(data);
    }
  }

  @Test
  public void smallDataWithInvalidChecksum_throwsIOExceptionInCreatePhase() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("Checksum");
    streamFactory.create(connection, AURL, BAD_CHECKSUM, reconnector);
  }

  @Test
  public void bigDataWithValidChecksum_readsOk() throws Exception {
    // at google, we know big data
    byte[] bigData = new byte[HttpStream.PRECHECK_BYTES + 70001];
    randoCalrissian.nextBytes(bigData);
    when(connection.getInputStream()).thenReturn(new ByteArrayInputStream(bigData));
    try (HttpStream stream =
            streamFactory.create(
                connection, AURL, Hashing.sha256().hashBytes(bigData).toString(), reconnector)) {
      assertThat(toByteArray(stream)).isEqualTo(bigData);
    }
  }

  @Test
  public void bigDataWithInvalidChecksum_throwsIOExceptionAfterCreateOnEof() throws Exception {
    // the probability of this test flaking is 8.6361686e-78
    byte[] bigData = new byte[HttpStream.PRECHECK_BYTES + 70001];
    randoCalrissian.nextBytes(bigData);
    when(connection.getInputStream()).thenReturn(new ByteArrayInputStream(bigData));
    try (HttpStream stream = streamFactory.create(connection, AURL, BAD_CHECKSUM, reconnector)) {
      thrown.expect(IOException.class);
      thrown.expectMessage("Checksum");
      ByteStreams.exhaust(stream);
      fail("Should have thrown error before close()");
    }
  }

  @Test
  public void httpServerSaidGzippedButNotGzipped_throwsZipExceptionInCreate() throws Exception {
    when(connection.getURL()).thenReturn(AURL);
    when(connection.getContentEncoding()).thenReturn("gzip");
    thrown.expect(ZipException.class);
    streamFactory.create(connection, AURL, "", reconnector);
  }

  @Test
  public void javascriptGzippedInTransit_automaticallyGunzips() throws Exception {
    when(connection.getURL()).thenReturn(AURL);
    when(connection.getContentEncoding()).thenReturn("x-gzip");
    when(connection.getInputStream()).thenReturn(new ByteArrayInputStream(gzipData(data)));
    try (HttpStream stream = streamFactory.create(connection, AURL, "", reconnector)) {
      assertThat(toByteArray(stream)).isEqualTo(data);
    }
  }

  @Test
  public void serverSaysTarballPathIsGzipped_doesntAutomaticallyGunzip() throws Exception {
    byte[] gzData = gzipData(data);
    when(connection.getURL()).thenReturn(new URL("http://doodle.example/foo.tar.gz"));
    when(connection.getContentEncoding()).thenReturn("gzip");
    when(connection.getInputStream()).thenReturn(new ByteArrayInputStream(gzData));
    try (HttpStream stream = streamFactory.create(connection, AURL, "", reconnector)) {
      assertThat(toByteArray(stream)).isEqualTo(gzData);
    }
  }

  @Test
  public void threadInterrupted_haltsReadingAndThrowsInterrupt() throws Exception {
    final AtomicBoolean wasInterrupted = new AtomicBoolean();
    Thread thread = new Thread(
        new Runnable() {
          @Override
          public void run() {
            try (HttpStream stream = streamFactory.create(connection, AURL, "", reconnector)) {
              stream.read();
              Thread.currentThread().interrupt();
              stream.read();
              fail();
            } catch (InterruptedIOException expected) {
              wasInterrupted.set(true);
            } catch (IOException ignored) {
              // ignored
            }
          }
        });
    thread.start();
    thread.join();
    assertThat(wasInterrupted.get()).isTrue();
  }

  private static byte[] gzipData(byte[] bytes) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (InputStream input = new ByteArrayInputStream(bytes);
        OutputStream output = new GZIPOutputStream(baos)) {
      ByteStreams.copy(input, output);
    }
    return baos.toByteArray();
  }
}
