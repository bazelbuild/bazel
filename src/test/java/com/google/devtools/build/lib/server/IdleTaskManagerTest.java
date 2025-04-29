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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link IdleTaskManager}. */
@RunWith(JUnit4.class)
public class IdleTaskManagerTest {
  @Test
  public void noRegisteredTasks_gcAndShrinkInterners() throws Exception {
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(), false);

    long idleCalled = System.nanoTime();
    manager.idle();
    Thread.sleep(250); // give tasks a chance to run
    manager.busy();
    long busyReturned = System.nanoTime();

    assertThat(idleCalled).isLessThan(manager.runGcAndMaybeShrinkInternersCalled);
    assertThat(manager.runGcAndMaybeShrinkInternersCalled).isLessThan(busyReturned);
  }

  @Test
  public void registeredTask_runSuccessfulTaskThenGcAndShrinkInterners() throws Exception {
    AtomicLong taskCalled = new AtomicLong();
    IdleTask task = () -> taskCalled.set(System.nanoTime());
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task), false);

    long idleCalled = System.nanoTime();
    manager.idle();
    Thread.sleep(250); // give tasks a chance to run
    manager.busy();
    long busyReturned = System.nanoTime();

    assertThat(idleCalled).isLessThan(taskCalled.get());
    assertThat(taskCalled.get()).isLessThan(manager.runGcAndMaybeShrinkInternersCalled);
    assertThat(manager.runGcAndMaybeShrinkInternersCalled).isLessThan(busyReturned);
  }

  @Test
  public void registeredTask_runFailedTaskThenGcAndShrinkInterners() throws Exception {
    AtomicLong taskCalled = new AtomicLong();
    IdleTask task =
        () -> {
          taskCalled.set(System.nanoTime());
          throw new IdleTaskException(new RuntimeException("failed"));
        };
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task), false);

    long idleCalled = System.nanoTime();
    manager.idle();
    Thread.sleep(250); // give tasks a chance to run
    manager.busy();
    long busyReturned = System.nanoTime();

    assertThat(idleCalled).isLessThan(taskCalled.get());
    assertThat(taskCalled.get()).isLessThan(manager.runGcAndMaybeShrinkInternersCalled);
    assertThat(manager.runGcAndMaybeShrinkInternersCalled).isLessThan(busyReturned);
  }

  @Test
  public void registeredTask_interruptTaskThenSkipGcAndShrinkInterners() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicBoolean taskInterrupted = new AtomicBoolean();
    IdleTask task =
        () -> {
          taskRunning.countDown();
          try {
            while (true) {
              Thread.sleep(1000);
            }
          } catch (InterruptedException e) {
            taskInterrupted.set(true);
            throw e;
          }
        };
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task), false);

    manager.idle();
    taskRunning.await();
    manager.busy();

    assertThat(taskInterrupted.get()).isTrue();
    assertThat(manager.runGcAndMaybeShrinkInternersCalled).isEqualTo(0);
  }
}
