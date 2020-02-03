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
import static com.google.devtools.build.lib.bazel.repository.downloader.DownloaderTestUtils.makeUrl;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.ManualClock;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Locale;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ProgressInputStream}. */
@RunWith(JUnit4.class)
public class ProgressInputStreamTest {

  private final ManualClock clock = new ManualClock();
  private final EventHandler eventHandler = mock(EventHandler.class);
  private final ExtendedEventHandler extendedEventHandler =
      new Reporter(new EventBus(), eventHandler);
  private final InputStream delegate = mock(InputStream.class);
  private final URL url = makeUrl("http://lol.example");
  private ProgressInputStream stream =
      new ProgressInputStream(Locale.US, clock, extendedEventHandler, 1, delegate, url, url);

  @After
  public void after() throws Exception {
    verifyNoMoreInteractions(eventHandler, delegate);
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
  public void readThrowsException_passesThrough() throws Exception {
    when(delegate.read()).thenThrow(new IOException());
    assertThrows(IOException.class, () -> stream.read());
    verify(delegate).read();
  }

  @Test
  public void readsAfterInterval_emitsProgressOnce() throws Exception {
    when(delegate.read()).thenReturn(42);
    assertThat(stream.read()).isEqualTo(42);
    clock.advanceMillis(1);
    assertThat(stream.read()).isEqualTo(42);
    assertThat(stream.read()).isEqualTo(42);
    verify(delegate, times(3)).read();
    verify(eventHandler).handle(Event.progress("Downloading http://lol.example: 2 bytes"));
  }

  @Test
  public void multipleIntervalsElapsed_showsMultipleProgress() throws Exception {
    stream.read();
    stream.read();
    clock.advanceMillis(1);
    stream.read();
    stream.read();
    clock.advanceMillis(1);
    stream.read();
    stream.read();
    verify(delegate, times(6)).read();
    verify(eventHandler).handle(Event.progress("Downloading http://lol.example: 3 bytes"));
    verify(eventHandler).handle(Event.progress("Downloading http://lol.example: 5 bytes"));
  }

  @Test
  public void bufferReadsAfterInterval_emitsProgressOnce() throws Exception {
    byte[] buffer = new byte[1024];
    when(delegate.read(any(byte[].class), anyInt(), anyInt())).thenReturn(1024);
    assertThat(stream.read(buffer)).isEqualTo(1024);
    clock.advanceMillis(1);
    assertThat(stream.read(buffer)).isEqualTo(1024);
    assertThat(stream.read(buffer)).isEqualTo(1024);
    verify(delegate, times(3)).read(same(buffer), eq(0), eq(1024));
    verify(eventHandler).handle(Event.progress("Downloading http://lol.example: 2,048 bytes"));
  }

  @Test
  public void bufferReadsAfterIntervalInGermany_usesPeriodAsSeparator() throws Exception {
    stream =
        new ProgressInputStream(Locale.GERMANY, clock, extendedEventHandler, 1, delegate, url, url);
    byte[] buffer = new byte[1024];
    when(delegate.read(any(byte[].class), anyInt(), anyInt())).thenReturn(1024);
    clock.advanceMillis(1);
    stream.read(buffer);
    verify(delegate).read(same(buffer), eq(0), eq(1024));
    verify(eventHandler).handle(Event.progress("Downloading http://lol.example: 1.024 bytes"));
  }

  @Test
  public void redirectedToDifferentServer_showsOriginalUrlWithVia() throws Exception {
    stream =
        new ProgressInputStream(
            Locale.US,
            clock,
            extendedEventHandler,
            1,
            delegate,
            new URL("http://cdn.example/foo"),
            url);
    when(delegate.read()).thenReturn(42);
    assertThat(stream.read()).isEqualTo(42);
    clock.advanceMillis(1);
    assertThat(stream.read()).isEqualTo(42);
    assertThat(stream.read()).isEqualTo(42);
    verify(delegate, times(3)).read();
    verify(eventHandler).handle(
        Event.progress("Downloading http://lol.example via cdn.example: 2 bytes"));
  }
}
