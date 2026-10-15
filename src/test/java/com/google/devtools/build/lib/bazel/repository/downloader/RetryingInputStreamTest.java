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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.downloader.RetryingInputStream.Reconnector;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URLConnection;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link RetryingInputStream}. */
@RunWith(JUnit4.class)
public class RetryingInputStreamTest {

  private final InputStream delegate = mock(InputStream.class);
  private final InputStream newDelegate = mock(InputStream.class);
  private final Reconnector reconnector = mock(Reconnector.class);
  private final URLConnection connection = mock(URLConnection.class);
  private final URLConnection initialConnection = mock(URLConnection.class);
  private final RetryingInputStream stream =
      new RetryingInputStream(delegate, initialConnection, reconnector);

  @After
  public void after() throws Exception {
    verifyNoMoreInteractions(delegate, newDelegate, reconnector);
  }

  @Test
  public void close_callsDelegate() throws Exception {
    stream.close();
    verify(delegate).close();
  }

  @Test
  public void available_callsDelegate() throws Exception {
    stream.available();
    verify(delegate).available();
  }

  @Test
  public void read_callsdelegate() throws Exception {
    stream.read();
    verify(delegate).read();
  }

  @Test
  public void bufferRead_callsdelegate() throws Exception {
    byte[] buffer = new byte[1024];
    stream.read(buffer);
    verify(delegate).read(same(buffer), eq(0), eq(1024));
  }

  @Test
  public void readInterrupted_alwaysPassesThrough() throws Exception {
    when(delegate.read()).thenThrow(new InterruptedIOException());
    assertThrows(InterruptedIOException.class, () -> stream.read());
    verify(delegate).read();
  }

  @Test
  @SuppressWarnings("unchecked")
  public void readTimesOut_retries() throws Exception {
    when(delegate.read()).thenReturn(1).thenThrow(new SocketTimeoutException());
    when(reconnector.connect(any(Throwable.class), any(ImmutableMap.class))).thenReturn(connection);
    when(connection.getInputStream()).thenReturn(newDelegate);
    when(newDelegate.read()).thenReturn(2);
    when(connection.getHeaderField("Content-Range")).thenReturn("bytes 1-42/42");
    assertThat(stream.read()).isEqualTo(1);
    assertThat(stream.read()).isEqualTo(2);
    verify(reconnector)
        .connect(any(Throwable.class), eq(ImmutableMap.of("Range", ImmutableList.of("bytes=1-"))));
    verify(delegate, times(2)).read();
    verify(delegate).close();
    verify(newDelegate).read();
  }

  @Test
  @SuppressWarnings("unchecked")
  public void failureWhenNoBytesAreRead_doesntUseRange() throws Exception {
    when(delegate.read()).thenThrow(new SocketTimeoutException());
    when(newDelegate.read()).thenReturn(1);
    when(reconnector.connect(any(Throwable.class), any(ImmutableMap.class))).thenReturn(connection);
    when(connection.getInputStream()).thenReturn(newDelegate);
    assertThat(stream.read()).isEqualTo(1);
    verify(reconnector).connect(any(Throwable.class), eq(ImmutableMap.of()));
    verify(delegate).read();
    verify(delegate).close();
    verify(newDelegate).read();
  }

  @Test
  @SuppressWarnings("unchecked")
  public void reconnectFails_alwaysPassesThrough() throws Exception {
    when(delegate.read()).thenThrow(new IOException());
    when(reconnector.connect(any(Throwable.class), any(ImmutableMap.class)))
        .thenThrow(new IOException());
    assertThrows(IOException.class, () -> stream.read());
    verify(delegate).read();
      verify(delegate).close();
    verify(reconnector).connect(any(Throwable.class), any(ImmutableMap.class));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void maxRetries_givesUp() throws Exception {
    when(delegate.read())
        .thenReturn(1)
        .thenThrow(new IOException())
        .thenThrow(new IOException())
        .thenThrow(new IOException())
        .thenThrow(new SocketTimeoutException());
    when(reconnector.connect(any(Throwable.class), any(ImmutableMap.class))).thenReturn(connection);
    when(connection.getInputStream()).thenReturn(delegate);
    when(connection.getHeaderField("Content-Range")).thenReturn("bytes 1-42/42");
    stream.read();
    SocketTimeoutException e = assertThrows(SocketTimeoutException.class, () -> stream.read());
    assertThat(e.getSuppressed()).hasLength(3);
    verify(reconnector, times(3))
        .connect(any(Throwable.class), eq(ImmutableMap.of("Range", ImmutableList.of("bytes=1-"))));
      verify(delegate, times(5)).read();
    verify(delegate, times(3)).close();
  }

  @Test
  public void reconnect_http206_resumesAtOffset() throws Exception {
    HttpURLConnection httpConnection = mock(HttpURLConnection.class);
    when(delegate.read()).thenReturn(1).thenThrow(new SocketTimeoutException());
    when(reconnector.connect(any(), any())).thenReturn(httpConnection);
    when(httpConnection.getResponseCode()).thenReturn(HttpURLConnection.HTTP_PARTIAL);
    when(httpConnection.getHeaderField("Content-Range")).thenReturn("bytes 1-2/3");
    when(httpConnection.getInputStream()).thenReturn(new ByteArrayInputStream(new byte[] {2, 3}));

    assertThat(stream.read()).isEqualTo(1);
    assertThat(stream.read()).isEqualTo(2);
    assertThat(stream.read()).isEqualTo(3);
    assertThat(stream.read()).isEqualTo(-1);

    verify(reconnector)
        .connect(any(Throwable.class), eq(ImmutableMap.of("Range", ImmutableList.of("bytes=1-"))));
    verify(delegate, times(2)).read();
    verify(delegate).close();
  }

  @Test
  public void reconnect_http200_skipsAlreadyReadBytes() throws Exception {
    HttpURLConnection httpConnection = mock(HttpURLConnection.class);
    when(delegate.read(any(), anyInt(), anyInt())).thenReturn(5);
    when(delegate.read()).thenThrow(new SocketTimeoutException());
    when(reconnector.connect(any(), any())).thenReturn(httpConnection);
    when(httpConnection.getResponseCode()).thenReturn(HttpURLConnection.HTTP_OK);
    when(httpConnection.getInputStream())
        .thenReturn(new ByteArrayInputStream(new byte[] {0, 1, 2, 3, 4, 5, 6}));

    assertThat(stream.read(new byte[5], 0, 5)).isEqualTo(5);
    assertThat(stream.read()).isEqualTo(5);
    assertThat(stream.read()).isEqualTo(6);
    assertThat(stream.read()).isEqualTo(-1);

    verify(reconnector)
        .connect(any(Throwable.class), eq(ImmutableMap.of("Range", ImmutableList.of("bytes=5-"))));
    verify(delegate).read(any(), anyInt(), anyInt());
    verify(delegate).read();
    verify(delegate).close();
  }

  @Test
  public void reconnect_noContentRange_skipsAlreadyReadBytes() throws Exception {
    when(delegate.read()).thenReturn(1).thenThrow(new SocketTimeoutException());
    when(reconnector.connect(any(), any())).thenReturn(connection);
    when(connection.getInputStream()).thenReturn(new ByteArrayInputStream(new byte[] {1, 2}));

    assertThat(stream.read()).isEqualTo(1);
    assertThat(stream.read()).isEqualTo(2);

    verify(reconnector)
        .connect(any(Throwable.class), eq(ImmutableMap.of("Range", ImmutableList.of("bytes=1-"))));
    verify(delegate, times(2)).read();
    verify(delegate).close();
  }

  @Test
  public void reconnect_http200ShorterThanOffset_throws() throws Exception {
    HttpURLConnection httpConnection = mock(HttpURLConnection.class);
    when(delegate.read(any(), anyInt(), anyInt())).thenReturn(5);
    when(delegate.read()).thenThrow(new IOException());
    when(reconnector.connect(any(), any())).thenReturn(httpConnection);
    when(httpConnection.getResponseCode()).thenReturn(HttpURLConnection.HTTP_OK);
    when(httpConnection.getInputStream()).thenReturn(new ByteArrayInputStream(new byte[] {0, 1}));

    assertThat(stream.read(new byte[5], 0, 5)).isEqualTo(5);
    assertThrows(EOFException.class, () -> stream.read());

    verify(reconnector).connect(any(), any());
    verify(delegate).read(any(), anyInt(), anyInt());
    verify(delegate).read();
    verify(delegate).close();
  }

  @Test
  public void reconnect_http200WithDifferentContentEncoding_throws() throws Exception {
    when(initialConnection.getContentEncoding()).thenReturn("gzip");
    RetryingInputStream stream = new RetryingInputStream(delegate, initialConnection, reconnector);
    HttpURLConnection httpConnection = mock(HttpURLConnection.class);
    when(delegate.read()).thenReturn(1).thenThrow(new IOException());
    when(reconnector.connect(any(), any())).thenReturn(httpConnection);
    when(httpConnection.getResponseCode()).thenReturn(HttpURLConnection.HTTP_OK);

    assertThat(stream.read()).isEqualTo(1);
    IOException e = assertThrows(IOException.class, () -> stream.read());
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Tried to reconnect at offset 1 but server sent a different representation");

    verify(reconnector).connect(any(), any());
    verify(delegate, times(2)).read();
    verify(delegate).close();
  }

  @Test
  public void reconnect_http200WithDifferentContentLength_throws() throws Exception {
    when(initialConnection.getHeaderField("Content-Length")).thenReturn("42");
    RetryingInputStream stream = new RetryingInputStream(delegate, initialConnection, reconnector);
    HttpURLConnection httpConnection = mock(HttpURLConnection.class);
    when(delegate.read()).thenReturn(1).thenThrow(new IOException());
    when(reconnector.connect(any(), any())).thenReturn(httpConnection);
    when(httpConnection.getResponseCode()).thenReturn(HttpURLConnection.HTTP_OK);
    when(httpConnection.getHeaderField("Content-Length")).thenReturn("43");

    assertThat(stream.read()).isEqualTo(1);
    IOException e = assertThrows(IOException.class, () -> stream.read());
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Tried to reconnect at offset 1 but server sent a different representation");

    verify(reconnector).connect(any(), any());
    verify(delegate, times(2)).read();
    verify(delegate).close();
  }

  @Test
  public void reconnect_invalidContentRange_throws() throws Exception {
    when(delegate.read()).thenReturn(1).thenThrow(new SocketTimeoutException());
    when(reconnector.connect(any(), any())).thenReturn(connection);
    when(connection.getHeaderField("Content-Range")).thenReturn("bytes 0-42/42");

    assertThat(stream.read()).isEqualTo(1);
    IOException e = assertThrows(IOException.class, () -> stream.read());
    assertThat(e)
        .hasMessageThat()
        .contains("Tried to reconnect at offset 1 but server didn't support it");

    verify(reconnector)
        .connect(any(Throwable.class), eq(ImmutableMap.of("Range", ImmutableList.of("bytes=1-"))));
    verify(delegate, times(2)).read();
    verify(delegate).close();
  }
}
