// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code ManualSleeper}. */
@RunWith(JUnit4.class)
public class ManualSleeperTest {

  private final ManualClock clock = new ManualClock();
  private final ManualSleeper sleeper = new ManualSleeper(clock);

  @Test
  public void sleepMillis_0_ok() throws InterruptedException {
    sleeper.sleepMillis(0);
    assertThat(clock.currentTimeMillis()).isEqualTo(0);
  }

  @Test
  public void sleepMillis_100_ok() throws InterruptedException {
    sleeper.sleepMillis(100);
    assertThat(clock.currentTimeMillis()).isEqualTo(100);
  }

  @Test
  public void sleepMillis_minus1_throws() throws InterruptedException {
    try {
      sleeper.sleepMillis(-1);
      fail("Should have thrown");
    } catch (IllegalArgumentException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("sleeper can't time travel");
    }
  }

  @Test
  public void scheduleRunnable_0_doesNotRunItImmediately() {
    AtomicInteger counter = new AtomicInteger();
    sleeper.scheduleRunnable(counter::incrementAndGet, 0);

    assertThat(counter.get()).isEqualTo(0);
  }

  @Test
  public void scheduleRunnable_100_doesNotRunItImmediately() {
    AtomicInteger counter = new AtomicInteger();
    sleeper.scheduleRunnable(counter::incrementAndGet, 100);

    assertThat(counter.get()).isEqualTo(0);
  }

  @Test
  public void scheduleRunnable_minus1_throws() {
    AtomicInteger counter = new AtomicInteger();

    try {
      sleeper.scheduleRunnable(counter::incrementAndGet, -1);
      fail("Should have thrown");
    } catch (IllegalArgumentException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("sleeper can't time travel");
      assertThat(counter.get()).isEqualTo(0);
    }
  }

  @Test
  public void scheduleRunnable_0_runsAfterSleep0() throws InterruptedException {
    AtomicInteger counter = new AtomicInteger();
    sleeper.scheduleRunnable(counter::incrementAndGet, 0);

    sleeper.sleepMillis(0);

    assertThat(counter.get()).isEqualTo(1);
  }

  @Test
  public void scheduleRunnable_0_runsAfterSleep0_doesNotRunSecondTime()
      throws InterruptedException {
    AtomicInteger counter = new AtomicInteger();
    sleeper.scheduleRunnable(counter::incrementAndGet, 0);

    sleeper.sleepMillis(0);
    sleeper.sleepMillis(0);

    assertThat(counter.get()).isEqualTo(1);
  }

  @Test
  public void scheduleRunnable_100_runsAfterSleepExactly100() throws InterruptedException {
    AtomicInteger counter = new AtomicInteger();
    sleeper.scheduleRunnable(counter::incrementAndGet, 100);

    sleeper.sleepMillis(50);
    assertThat(counter.get()).isEqualTo(0);

    sleeper.sleepMillis(50);
    assertThat(counter.get()).isEqualTo(1);
  }

  @Test
  public void scheduleRunnable_100_runsAfterSleepOver100() throws InterruptedException {
    AtomicInteger counter = new AtomicInteger();
    sleeper.scheduleRunnable(counter::incrementAndGet, 100);

    sleeper.sleepMillis(50);
    assertThat(counter.get()).isEqualTo(0);

    sleeper.sleepMillis(150);
    assertThat(counter.get()).isEqualTo(1);
  }

  @Test
  public void scheduleRunnable_100_doesNotRunAgain() throws InterruptedException {
    AtomicInteger counter = new AtomicInteger();
    sleeper.scheduleRunnable(counter::incrementAndGet, 100);

    sleeper.sleepMillis(150);
    assertThat(counter.get()).isEqualTo(1);

    sleeper.sleepMillis(100);
    assertThat(counter.get()).isEqualTo(1);
  }
}
