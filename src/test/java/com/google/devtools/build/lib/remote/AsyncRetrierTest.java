// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.Retrier.Backoff;
import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker;
import com.google.devtools.build.lib.remote.Retrier.CircuitBreaker.State;
import com.google.devtools.build.lib.remote.Retrier.RetryException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.function.Predicate;
import java.util.function.Supplier;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/**
 * Tests for {@link AsyncRetrier}.
 */
@RunWith(JUnit4.class)
public class AsyncRetrierTest {

  @Mock
  private CircuitBreaker alwaysOpen;

  private static final Predicate<Exception> RETRY_ALL = (e) -> true;

  private static ListeningScheduledExecutorService retryService;

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @Before
  public void setup() {
    MockitoAnnotations.initMocks(this);
    when(alwaysOpen.state()).thenReturn(State.ACCEPT_CALLS);
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
  }

  @Test
  public void asyncRetryShouldWork() throws Exception {
    // Test that a call is retried according to the backoff.
    // All calls fail.

    Supplier<Backoff> s = () -> new ZeroBackoff(/*maxRetries=*/ 2);
    AsyncRetrier r = new AsyncRetrier(s, RETRY_ALL, retryService, alwaysOpen);
    try {
      r.executeAsync(
              () -> {
                throw new Exception("call failed");
              })
          .get();
      fail("exception expected.");
    } catch (ExecutionException e) {
      assertThat(e.getCause()).isInstanceOf(RetryException.class);
      assertThat(((RetryException) e.getCause()).getAttempts()).isEqualTo(3);
    }
  }
}
