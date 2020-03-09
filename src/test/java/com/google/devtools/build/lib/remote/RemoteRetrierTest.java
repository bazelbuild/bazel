// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.mockito.Mockito.when;

import com.google.common.collect.Range;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.Retrier.Backoff;
import com.google.devtools.build.lib.remote.Retrier.Sleeper;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.common.options.Options;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Tests for {@link RemoteRetrier}.
 */
@RunWith(JUnit4.class)
public class RemoteRetrierTest {

  interface Foo {
    String foo();
  }

  private RemoteRetrierTest.Foo fooMock;
  private ListeningScheduledExecutorService retryService;

  @Before
  public void setUp() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
    fooMock = Mockito.mock(RemoteRetrierTest.Foo.class);
  }

  @After
  public void tearDown() throws InterruptedException {
    retryService.shutdownNow();
    retryService.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
  }

  @Test
  public void testExponentialBackoff() throws Exception {
    Retrier.Backoff backoff =
        new ExponentialBackoff(Duration.ofSeconds(1), Duration.ofSeconds(10), 2, 0, 6);
    assertThat(backoff.nextDelayMillis()).isEqualTo(1000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(2000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(4000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(8000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(10000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(10000);
    assertThat(backoff.nextDelayMillis()).isLessThan(0L);
  }

  @Test
  public void testExponentialBackoffJittered() throws Exception {
    Retrier.Backoff backoff =
        new ExponentialBackoff(Duration.ofSeconds(1), Duration.ofSeconds(10), 2, 0.1, 6);
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(900L, 1100L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(1800L, 2200L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(3600L, 4400L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(7200L, 8800L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(9000L, 11000L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(9000L, 11000L));
    assertThat(backoff.nextDelayMillis()).isLessThan(0L);
  }

  @Test
  public void testNoRetries() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteMaxRetryAttempts = 0;

    RemoteRetrier retrier =
        Mockito.spy(new RemoteRetrier(options, (e) -> true, retryService, Retrier.ALLOW_ALL_CALLS));
    when(fooMock.foo())
        .thenReturn("bla")
        .thenThrow(Status.Code.UNKNOWN.toStatus().asRuntimeException());
    assertThat(retrier.execute(() -> fooMock.foo())).isEqualTo("bla");
    assertThrows(StatusRuntimeException.class, () -> retrier.execute(fooMock::foo));
    Mockito.verify(fooMock, Mockito.times(2)).foo();
  }

  @Test
  public void testNonRetriableError() throws Exception {
    Supplier<Backoff> s =
        () -> new ExponentialBackoff(Duration.ofSeconds(1), Duration.ofSeconds(10), 2.0, 0.0, 2);
    RemoteRetrier retrier =
        Mockito.spy(
            new RemoteRetrier(
                s,
                (e) -> false,
                retryService,
                Retrier.ALLOW_ALL_CALLS,
                Mockito.mock(Sleeper.class)));
    when(fooMock.foo()).thenThrow(Status.Code.UNKNOWN.toStatus().asRuntimeException());
    assertThrows(StatusRuntimeException.class, () -> retrier.execute(fooMock::foo));
    Mockito.verify(fooMock, Mockito.times(1)).foo();
  }

  @Test
  public void testRepeatedRetriesReset() throws Exception {
    Supplier<Backoff> s =
        () -> new ExponentialBackoff(Duration.ofSeconds(1), Duration.ofSeconds(10), 2.0, 0.0, 2);
    Sleeper sleeper = Mockito.mock(Sleeper.class);
    RemoteRetrier retrier =
        Mockito.spy(
            new RemoteRetrier(s, (e) -> true, retryService, Retrier.ALLOW_ALL_CALLS, sleeper));

    when(fooMock.foo()).thenThrow(Status.Code.UNKNOWN.toStatus().asRuntimeException());
    assertThrows(StatusRuntimeException.class, () -> retrier.execute(fooMock::foo));
    assertThrows(StatusRuntimeException.class, () -> retrier.execute(fooMock::foo));
    Mockito.verify(sleeper, Mockito.times(2)).sleep(1000);
    Mockito.verify(sleeper, Mockito.times(2)).sleep(2000);
    Mockito.verify(fooMock, Mockito.times(6)).foo();
  }

  @Test
  public void testInterruptedExceptionIsPassedThrough() throws Exception {
    InterruptedException thrown = new InterruptedException();

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteMaxRetryAttempts = 0;
    RemoteRetrier retrier =
        new RemoteRetrier(options, (e) -> true, retryService, Retrier.ALLOW_ALL_CALLS);
    InterruptedException expected =
        assertThrows(
            InterruptedException.class,
            () ->
                retrier.execute(
                    () -> {
                      throw thrown;
                    }));
    assertThat(expected).isSameInstanceAs(thrown);
  }
}
