// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.ConcurrencyMeter.Ticket;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Future.State;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ConcurrencyMeter}. */
@RunWith(JUnit4.class)
public final class ConcurrencyMeterTest {

  private static void assertFutureIsSuccessful(Future<?> future) {
    assertThat(future.state()).isEqualTo(State.SUCCESS);
  }

  @Test
  public void testGrant() throws Exception {
    ConcurrencyMeter scheduler = new ConcurrencyMeter("meter", 3, BlazeClock.instance());

    ListenableFuture<Ticket> req1 = scheduler.request(2, 0);
    assertFutureIsSuccessful(req1);
    assertThat(scheduler.queueSize()).isEqualTo(0);
    req1.get().done();

    ListenableFuture<Ticket> req2 = scheduler.request(2, 0);
    assertFutureIsSuccessful(req2);

    ListenableFuture<Ticket> req3 = scheduler.request(1, 0);
    assertFutureIsSuccessful(req3);
    assertThat(scheduler.queueSize()).isEqualTo(0);
  }

  @Test
  public void testBlock() throws Exception {
    ConcurrencyMeter scheduler = new ConcurrencyMeter("meter", 3, BlazeClock.instance());

    ListenableFuture<Ticket> req1 = scheduler.request(2, 0);
    assertFutureIsSuccessful(req1);

    ListenableFuture<Ticket> req2 = scheduler.request(2, 0);
    assertThat(req2.isDone()).isFalse();
    assertThat(scheduler.queueSize()).isEqualTo(1);

    req1.get().done();
    assertFutureIsSuccessful(req2);
    assertThat(scheduler.queueSize()).isEqualTo(0);
  }

  @Test
  public void testGrantZero() {
    ConcurrencyMeter scheduler = new ConcurrencyMeter("meter", 3, BlazeClock.instance());
    ListenableFuture<Ticket> req = scheduler.request(0, 0);
    assertFutureIsSuccessful(req);
  }

  @Test
  public void testGrantFromZero() {
    ConcurrencyMeter scheduler = new ConcurrencyMeter("meter", 3, BlazeClock.instance());

    ListenableFuture<Ticket> req1 = scheduler.request(10, 0);
    assertFutureIsSuccessful(req1);

    ListenableFuture<Ticket> req2 = scheduler.request(0, 0);
    assertThat(req2.isDone()).isFalse();
  }

  @Test
  public void testPriority() throws Exception {
    ConcurrencyMeter scheduler = new ConcurrencyMeter("meter", 3, BlazeClock.instance());

    ListenableFuture<Ticket> req1 = scheduler.request(2, 0);
    assertFutureIsSuccessful(req1);

    ListenableFuture<Ticket> req2 = scheduler.request(2, 0);
    assertThat(req2.isDone()).isFalse();

    ListenableFuture<Ticket> req3 = scheduler.request(2, 1);
    assertThat(req3.isDone()).isFalse();

    req1.get().done();
    assertThat(req2.isDone()).isFalse();
    assertFutureIsSuccessful(req3);

    req3.get().done();
    assertFutureIsSuccessful(req2);
  }

  @Test
  public void testThreadSafety() throws Exception {
    int requestsPerThread = 10;
    int threads = 10;
    Random r = new Random();

    ConcurrencyMeter scheduler = new ConcurrencyMeter("meter", 100, BlazeClock.instance());
    ExecutorService exec = Executors.newFixedThreadPool(threads);
    ExecutorService unboundedPool = Executors.newCachedThreadPool();
    List<Future<?>> results = new ArrayList<>();
    CountDownLatch allJobsDone = new CountDownLatch(threads * requestsPerThread);

    // For every thread, we'll ask for requestsPerThread resource bundles. For
    // each of those, we'll set up a listener to release the resources after
    // a small, but random amount of time.
    for (int i = 0; i < threads; i++) {
      results.add(
          exec.submit(
              () -> {
                for (int j = 0; j < requestsPerThread; j++) {
                  int size = r.nextInt(20) + 3;
                  ListenableFuture<Ticket> req = scheduler.request(size, 0);
                  req.addListener(
                      () -> {
                        long sleepiness = r.nextInt(30);
                        try {
                          Thread.sleep(sleepiness);
                          req.get().done();
                          allJobsDone.countDown();
                        } catch (Exception e) {
                          if (e instanceof InterruptedException) {
                            Thread.currentThread().interrupt();
                          }
                          throw new IllegalStateException(e);
                        }
                      },
                      unboundedPool);
                }
              }));
    }

    exec.shutdown();
    exec.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);

    assertThat(results).hasSize(threads);
    for (Future<?> result : results) {
      assertFutureIsSuccessful(result); // Make sure nothing went wrong.
    }
    allJobsDone.await();

    // Make sure nothing is left to be scheduled
    assertFutureIsSuccessful(scheduler.request(0, 0));
    assertThat(scheduler.queueSize()).isEqualTo(0);
  }

  @Test
  public void cancelledRequest_releasedImmediately() throws Exception {
    ConcurrencyMeter meter = new ConcurrencyMeter("meter", 1, BlazeClock.instance());
    Ticket ticket = meter.request(1, 1).get();
    ListenableFuture<Ticket> blockedRequest = meter.request(1, 1);

    blockedRequest.cancel(/* mayInterruptIfRunning= */ false);
    assertThat(blockedRequest.isCancelled()).isTrue();

    ticket.done();
    assertFutureIsSuccessful(meter.request(1, 1));
  }

  @Test
  public void manyBlockedAllCancelled_noStackOverflow() throws Exception {
    ConcurrencyMeter meter = new ConcurrencyMeter("meter", 1, BlazeClock.instance());
    Ticket liveTicket = meter.request(1, 1).get();

    List<ListenableFuture<Ticket>> blockedRequests = new ArrayList<>();
    for (int i = 0; i < 100_000; i++) {
      blockedRequests.add(meter.request(1, 1));
    }
    for (ListenableFuture<Ticket> blockedRequest : blockedRequests) {
      blockedRequest.cancel(/* mayInterruptIfRunning= */ true);
      assertThat(blockedRequest.isCancelled()).isTrue();
    }

    liveTicket.done();
  }

  @Test
  public void stats() throws Exception {
    ManualClock clock = new ManualClock();
    ConcurrencyMeter meter = new ConcurrencyMeter("meter", 10, clock);

    Ticket ticket1 = meter.request(1, 1).get();
    Ticket ticket2 = meter.request(1, 1).get();
    clock.advance(Duration.ofMillis(1));

    Instant timeOfMax = clock.now();
    meter.request(1, 1).get(); // Unreleased ticket.
    ticket1.done();
    clock.advance(Duration.ofMillis(1));

    ticket2.done();

    assertThat(meter.getStats())
        .isEqualTo(new ConcurrencyMeter.Stats("meter", 10, 1, 3, timeOfMax.toEpochMilli()));
  }

  @Test
  public void stats_maxObservedMultipleTimes_maxLeasedTimeMsMatchesLastTime() throws Exception {
    ManualClock clock = new ManualClock();
    ConcurrencyMeter meter = new ConcurrencyMeter("meter", 1, clock);

    Ticket ticket1 = meter.request(1, 1).get();
    ticket1.done();
    clock.advance(Duration.ofMillis(1));

    Ticket ticket2 = meter.request(1, 1).get();
    ticket2.done();
    clock.advance(Duration.ofMillis(1));

    Instant timeOfLastMax = clock.now();
    Ticket ticket3 = meter.request(1, 1).get();
    ticket3.done();
    clock.advance(Duration.ofMillis(1));

    var stats = meter.getStats();
    assertThat(stats.maxLeasedTimeMs()).isEqualTo(timeOfLastMax.toEpochMilli());
  }

  @Test
  public void stats_noPermitsLeased_noTimestamp() {
    Clock throwingClock =
        new Clock() {
          @Override
          public long currentTimeMillis() {
            throw new UnsupportedOperationException("Should not need to get the current time");
          }

          @Override
          public long nanoTime() {
            throw new UnsupportedOperationException("Should not need to get the current time");
          }
        };
    ConcurrencyMeter meter = new ConcurrencyMeter("meter", 1, throwingClock);

    var stats = meter.getStats();
    assertThat(stats.maxLeased()).isEqualTo(0);
    assertThat(stats.maxLeasedTimeMs()).isEqualTo(0);
  }
}
