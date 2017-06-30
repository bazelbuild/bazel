// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;
import static org.mockito.Mockito.when;

import com.google.common.collect.Range;
import io.grpc.Status;
import java.time.Duration;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link Retrier}. */
@RunWith(JUnit4.class)
public class RetrierTest {

  interface Foo {
    public String foo();
  }

  private Foo fooMock;

  @Before
  public void setUp() {
    fooMock = Mockito.mock(Foo.class);
  }

  @Test
  public void testExponentialBackoff() throws Exception {
    Retrier.Backoff backoff =
        Retrier.Backoff.exponential(
                Duration.ofSeconds(1), Duration.ofSeconds(10), 2, 0, 6)
            .get();
    assertThat(backoff.nextDelayMillis()).isEqualTo(1000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(2000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(4000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(8000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(10000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(10000);
    assertThat(backoff.nextDelayMillis()).isEqualTo(Retrier.Backoff.STOP);
  }

  @Test
  public void testExponentialBackoffJittered() throws Exception {
    Retrier.Backoff backoff =
        Retrier.Backoff.exponential(
                Duration.ofSeconds(1), Duration.ofSeconds(10), 2, 0.1, 6)
            .get();
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(900L, 1100L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(1800L, 2200L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(3600L, 4400L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(7200L, 8800L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(9000L, 11000L));
    assertThat(backoff.nextDelayMillis()).isIn(Range.closedOpen(9000L, 11000L));
    assertThat(backoff.nextDelayMillis()).isEqualTo(Retrier.Backoff.STOP);
  }

  void assertThrows(Retrier retrier, int attempts) throws InterruptedException {
    try {
      retrier.execute(() -> fooMock.foo());
      fail();
    } catch (RetryException e) {
      assertThat(e.getAttempts()).isEqualTo(attempts);
    }
  }

  @Test
  public void testNoRetries() throws Exception {
    Retrier retrier = Mockito.spy(Retrier.NO_RETRIES);
    Mockito.doNothing().when(retrier).sleep(Mockito.anyLong());
    when(fooMock.foo())
        .thenReturn("bla")
        .thenThrow(Status.Code.UNKNOWN.toStatus().asRuntimeException());
    assertThat(retrier.execute(() -> fooMock.foo())).isEqualTo("bla");
    assertThrows(retrier, 1);
    Mockito.verify(fooMock, Mockito.times(2)).foo();
  }

  @Test
  public void testRepeatedRetriesReset() throws Exception {
    Retrier retrier =
        Mockito.spy(
            new Retrier(
                Retrier.Backoff.exponential(
                    Duration.ofSeconds(1), Duration.ofSeconds(10), 2, 0, 2),
                Retrier.RETRY_ALL));
    Mockito.doNothing().when(retrier).sleep(Mockito.anyLong());
    when(fooMock.foo()).thenThrow(Status.Code.UNKNOWN.toStatus().asRuntimeException());
    assertThrows(retrier, 3);
    assertThrows(retrier, 3);
    Mockito.verify(retrier, Mockito.times(2)).sleep(1000);
    Mockito.verify(retrier, Mockito.times(2)).sleep(2000);
    Mockito.verify(fooMock, Mockito.times(6)).foo();
  }
}

