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

package com.google.devtools.build.lib.server;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.server.CommandManager.RunningCommand;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.lang.Thread.State;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommandManager}. */
@RunWith(JUnit4.class)
public class CommandManagerTest {

  @Test
  public void testBasicOperationsOnSingleThread() {
    CommandManager underTest =
        new CommandManager(/*doIdleServerTasks=*/ false, "slow interrupt message suffix");
    assertThat(underTest.isEmpty()).isTrue();
    try (RunningCommand firstCommand = underTest.createCommand()) {
      assertThat(underTest.isEmpty()).isFalse();
      assertThat(isValidUuid(firstCommand.getId())).isTrue();
      try (RunningCommand secondCommand = underTest.createCommand()) {
        assertThat(underTest.isEmpty()).isFalse();
        assertThat(isValidUuid(secondCommand.getId())).isTrue();
        assertThat(firstCommand.getId()).isNotEqualTo(secondCommand.getId());
      }
      assertThat(underTest.isEmpty()).isFalse();
    }
    assertThat(underTest.isEmpty()).isTrue();
  }

  @Test
  public void testNotifiesOnBusyAndIdle() throws Exception {
    AtomicInteger notificationCounter = new AtomicInteger(0);
    CommandManager underTest =
        new CommandManager(/*doIdleServerTasks=*/ false, "slow interrupt message suffix");
    AtomicBoolean waiting = new AtomicBoolean(false);
    CyclicBarrier cyclicBarrier = new CyclicBarrier(2);

    TestThread thread =
        new TestThread(
            () -> {
              try {
                while (true) {
                  waiting.set(true);
                  underTest.waitForChange();
                  waiting.set(false);
                  notificationCounter.incrementAndGet();
                  cyclicBarrier.await();
                }
              } catch (InterruptedException e) {
                // Used to terminate the thread.
              }
            });
    thread.start();

    // We want to ensure at each step that we are actively awaiting notification.
    waitForThreadWaiting(waiting, thread);
    try (RunningCommand firstCommand = underTest.createCommand()) {
      cyclicBarrier.await();
      assertThat(notificationCounter.get()).isEqualTo(1);
      waitForThreadWaiting(waiting, thread);
      try (RunningCommand secondCommand = underTest.createCommand()) {
        cyclicBarrier.await();
        assertThat(notificationCounter.get()).isEqualTo(2);
        waitForThreadWaiting(waiting, thread);
      }
      cyclicBarrier.await();
      assertThat(notificationCounter.get()).isEqualTo(3);
      waitForThreadWaiting(waiting, thread);
    }
    cyclicBarrier.await();
    assertThat(notificationCounter.get()).isEqualTo(4);

    thread.interrupt();
    thread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void testIdleTasksEnabled() throws Exception {
    CommandManager underTest =
        new CommandManager(/* doIdleServerTasks= */ true, "slow interrupt message suffix");

    CountDownLatch taskRunning = new CountDownLatch(1);

    IdleTask idleTask =
        new IdleTask() {
          @Override
          public String displayName() {
            return "my idle task";
          }

          @Override
          public void run() {
            taskRunning.countDown();
          }
        };

    // The 1st command collects no results and registers a task.
    try (RunningCommand c1 = underTest.createCommand()) {
      assertThat(underTest.getIdleTaskResults()).isNull();
      c1.setIdleTasks(ImmutableList.of(idleTask));
    }

    taskRunning.await();

    // The 2nd command does not attempt to collect results and registers no tasks.
    try (RunningCommand c2 = underTest.createCommand()) {}

    // The 3rd command collects results from the 1st command and registers no tasks.
    try (RunningCommand c3 = underTest.createCommand()) {
      assertThat(
              underTest.getIdleTaskResults().stream()
                  .map(r -> new IdleTask.Result(r.name(), r.status(), Duration.ZERO)))
          .containsExactly(
              new IdleTask.Result("my idle task", IdleTask.Status.SUCCESS, Duration.ZERO));
    }

    // The 4th command collects no results.
    try (RunningCommand c4 = underTest.createCommand()) {
      assertThat(underTest.getIdleTaskResults()).isNull();
    }
  }

  @Test
  public void testIdleTasksDisabled() throws Exception {
    CommandManager underTest =
        new CommandManager(/* doIdleServerTasks= */ false, "slow interrupt message suffix");

    IdleTask idleTask =
        new IdleTask() {
          @Override
          public String displayName() {
            return "my idle task";
          }

          @Override
          public void run() {}
        };

    try (RunningCommand c1 = underTest.createCommand()) {
      c1.setIdleTasks(ImmutableList.of(idleTask));
    }

    try (RunningCommand c2 = underTest.createCommand()) {
      assertThat(underTest.getIdleTaskResults()).isNull();
    }
  }

  private static void waitForThreadWaiting(AtomicBoolean readyToWaitForChange, Thread thread)
      throws InterruptedException {
    while (!(readyToWaitForChange.get() && thread.getState() == State.WAITING)) {
      Thread.sleep(50);
    }
  }

  private static boolean isValidUuid(String uuidString) {
    try {
      UUID unused = UUID.fromString(uuidString);
      return true;
    } catch (IllegalArgumentException e) {
      return false;
    }
  }
}
