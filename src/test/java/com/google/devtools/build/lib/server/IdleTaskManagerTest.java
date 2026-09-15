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
  public void registeredTask_taskSuccessful() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicBoolean taskDone = new AtomicBoolean();
    IdleTask task =
        makeTask(
            "task",
            () -> {
              taskRunning.countDown();
              Uninterruptibles.sleepUninterruptibly(Duration.ofMillis(200));
              taskDone.set(true);
            });
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task));

    manager.idle();
    taskRunning.await(); // wait for task to start
    ImmutableList<IdleTask.Result> stats = manager.busy();

    assertThat(taskDone.get()).isTrue();

    assertThat(stats.stream().map(s -> new IdleTask.Result(s.name(), s.status(), Duration.ZERO)))
        .containsExactly(new IdleTask.Result("task", IdleTask.Status.SUCCESS, Duration.ZERO));
  }

  @Test
  public void registeredTask_taskFailed() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicBoolean taskDone = new AtomicBoolean();
    IdleTask task =
        makeTask(
            "task",
            () -> {
              taskRunning.countDown();
              Uninterruptibles.sleepUninterruptibly(Duration.ofMillis(200));
              try {
                throw new IdleTaskException(new RuntimeException("failed"));
              } finally {
                taskDone.set(true);
              }
            });
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task));

    manager.idle();
    taskRunning.await(); // wait for task to start
    ImmutableList<IdleTask.Result> stats = manager.busy();

    assertThat(taskDone.get()).isTrue();

    assertThat(stats.stream().map(s -> new IdleTask.Result(s.name(), s.status(), Duration.ZERO)))
        .containsExactly(new IdleTask.Result("task", IdleTask.Status.FAILURE, Duration.ZERO));
  }

  @Test
  public void registeredTask_taskInterrupted() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicBoolean taskInterrupted = new AtomicBoolean();
    IdleTask task =
        makeTask(
            "task",
            () -> {
              taskRunning.countDown();
              try {
                Thread.sleep(Duration.ofDays(1));
              } catch (InterruptedException e) {
                taskInterrupted.set(true);
                throw e;
              }
            });
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task));

    manager.idle();
    taskRunning.await(); // wait for task to start
    ImmutableList<IdleTask.Result> stats = manager.busy();

    assertThat(taskInterrupted.get()).isTrue();

    assertThat(stats.stream().map(s -> new IdleTask.Result(s.name(), s.status(), Duration.ZERO)))
        .containsExactly(new IdleTask.Result("task", IdleTask.Status.INTERRUPTED, Duration.ZERO));
  }

  @Test
  public void registeredTask_taskNotStarted() throws Exception {
    AtomicBoolean taskStarted = new AtomicBoolean();
    IdleTask task = makeTask("task", Duration.ofDays(1), () -> taskStarted.set(true));
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task));

    manager.idle();
    Thread.sleep(Duration.ofMillis(200)); // make it more likely that a bug will be caught
    ImmutableList<IdleTask.Result> stats = manager.busy();

    assertThat(taskStarted.get()).isFalse();

    assertThat(stats)
        .containsExactly(new IdleTask.Result("task", IdleTask.Status.NOT_STARTED, Duration.ZERO));
  }

  @Test
  public void registeredTask_multipleTasks() throws Exception {
    AtomicBoolean taskRunning = new AtomicBoolean(false);
    AtomicBoolean concurrentTasksDetected = new AtomicBoolean(false);
    CountDownLatch finishedTasks = new CountDownLatch(3);

    ImmutableList<IdleTask> tasks =
        ImmutableList.of(
            makeTask("a", () -> runTask(taskRunning, concurrentTasksDetected, finishedTasks)),
            makeTask("b", () -> runTask(taskRunning, concurrentTasksDetected, finishedTasks)),
            makeTask("c", () -> runTask(taskRunning, concurrentTasksDetected, finishedTasks)));

    IdleTaskManager manager = new IdleTaskManager(tasks);

    manager.idle();
    finishedTasks.await();
    ImmutableList<IdleTask.Result> stats = manager.busy();

    assertThat(concurrentTasksDetected.get()).isFalse();

    assertThat(stats.stream().map(s -> new IdleTask.Result(s.name(), s.status(), Duration.ZERO)))
        .containsExactly(
            new IdleTask.Result("a", IdleTask.Status.SUCCESS, Duration.ZERO),
            new IdleTask.Result("b", IdleTask.Status.SUCCESS, Duration.ZERO),
            new IdleTask.Result("c", IdleTask.Status.SUCCESS, Duration.ZERO))
        .inOrder();
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

  private static IdleTask makeTask(String name, IdleTaskRunnable runnable) {
    return makeTask(name, Duration.ZERO, runnable);
  }

  private static IdleTask makeTask(String name, Duration delay, IdleTaskRunnable runnable) {
    return new IdleTask() {
      @Override
      public String displayName() {
        return name;
      }

      @Override
      public Duration delay() {
        return delay;
      }

      @Override
      public void run() throws IdleTaskException, InterruptedException {
        runnable.run();
      }
    };
  }

  private interface IdleTaskRunnable {
    void run() throws IdleTaskException, InterruptedException;
  }
}
