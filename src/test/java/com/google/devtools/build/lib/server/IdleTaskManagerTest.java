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
package com.google.devtools.build.lib.server;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Uninterruptibles;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link IdleTaskManager}. */
@RunWith(JUnit4.class)
public class IdleTaskManagerTest {
  @Test
  public void registeredTask_runSuccessfulTask() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicBoolean taskDone = new AtomicBoolean();
    IdleTask task =
        () -> {
          taskRunning.countDown();
          Uninterruptibles.sleepUninterruptibly(Duration.ofMillis(200));
          taskDone.set(true);
        };
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task));

    manager.idle();
    taskRunning.await(); // wait for task to start
    manager.busy();

    assertThat(taskDone.get()).isTrue();
  }

  @Test
  public void registeredTask_runFailedTask() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicBoolean taskDone = new AtomicBoolean();
    IdleTask task =
        () -> {
          taskRunning.countDown();
          Uninterruptibles.sleepUninterruptibly(Duration.ofMillis(200));
          try {
            throw new IdleTaskException(new RuntimeException("failed"));
          } finally {
            taskDone.set(true);
          }
        };
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task));

    manager.idle();
    taskRunning.await(); // wait for task to start
    manager.busy();

    assertThat(taskDone.get()).isTrue();
  }

  @Test
  public void registeredTask_interruptTask() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicBoolean taskInterrupted = new AtomicBoolean();
    IdleTask task =
        () -> {
          taskRunning.countDown();
          try {
            Thread.sleep(Duration.ofDays(1));
          } catch (InterruptedException e) {
            taskInterrupted.set(true);
            throw e;
          }
        };
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task));

    manager.idle();
    taskRunning.await(); // wait for task to start
    manager.busy();

    assertThat(taskInterrupted.get()).isTrue();
  }

  @Test
  public void registeredTask_multipleTasksRunSerially() throws Exception {
    AtomicBoolean taskRunning = new AtomicBoolean(false);
    AtomicBoolean concurrentTasksDetected = new AtomicBoolean(false);
    CountDownLatch finishedTasks = new CountDownLatch(3);

    ImmutableList<IdleTask> tasks =
        ImmutableList.of(
            () -> runTask(taskRunning, concurrentTasksDetected, finishedTasks),
            () -> runTask(taskRunning, concurrentTasksDetected, finishedTasks),
            () -> runTask(taskRunning, concurrentTasksDetected, finishedTasks));

    IdleTaskManager manager = new IdleTaskManager(tasks);

    manager.idle();
    finishedTasks.await();
    manager.busy();

    assertThat(concurrentTasksDetected.get()).isFalse();
  }

  private static final void runTask(
      AtomicBoolean taskRunning,
      AtomicBoolean concurrentTasksDetected,
      CountDownLatch finishedTasks)
      throws InterruptedException {
    if (!taskRunning.compareAndSet(false, true)) {
      concurrentTasksDetected.set(true);
    }
    Thread.sleep(Duration.ofMillis(200)); // make it more likely that a bug will be caught
    finishedTasks.countDown();
    if (!taskRunning.compareAndSet(true, false)) {
      concurrentTasksDetected.set(true);
    }
  }
}
