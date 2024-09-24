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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link IdleTaskManager}. */
@RunWith(JUnit4.class)
public class IdleTaskManagerTest {
  @Test
  public void noRegisteredTasks_runGc() throws Exception {
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(), false);

    long idleCalled = System.nanoTime();
    manager.idle();
    Thread.sleep(100); // give task a chance to run
    manager.busy();
    long busyReturned = System.nanoTime();

    assertThat(idleCalled).isLessThan(manager.runGcCalled);
    assertThat(manager.runGcCalled).isLessThan(busyReturned);
  }

  @Test
  public void registeredTask_runTaskThenGc() throws Exception {
    AtomicLong taskCalled = new AtomicLong();
    IdleTask task = () -> taskCalled.set(System.nanoTime());
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task), false);

    long idleCalled = System.nanoTime();
    manager.idle();
    Thread.sleep(100); // give task a chance to run
    manager.busy();
    long busyReturned = System.nanoTime();

    assertThat(idleCalled).isLessThan(taskCalled.get());
    assertThat(taskCalled.get()).isLessThan(manager.runGcCalled);
    assertThat(manager.runGcCalled).isLessThan(busyReturned);
  }

  @Test
  public void registeredTask_interruptTaskThenSkipGc() throws Exception {
    CountDownLatch taskRunning = new CountDownLatch(1);
    AtomicLong taskInterrupted = new AtomicLong();
    IdleTask task =
        () -> {
          taskRunning.countDown();
          while (true) {
            try {
              Thread.sleep(100);
            } catch (InterruptedException e) {
              taskInterrupted.set(System.nanoTime());
              return;
            }
          }
        };
    IdleTaskManager manager = new IdleTaskManager(ImmutableList.of(task), false);

    manager.idle();
    Uninterruptibles.awaitUninterruptibly(taskRunning);
    manager.busy();
    long busyReturned = System.nanoTime();

    assertThat(taskInterrupted.get()).isLessThan(busyReturned);
    assertThat(manager.runGcCalled).isEqualTo(0);
  }
}
