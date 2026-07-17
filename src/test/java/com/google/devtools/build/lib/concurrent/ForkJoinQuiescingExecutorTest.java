// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.concurrent.ForkJoinQuiescingExecutor}. */
@RunWith(JUnit4.class)
public class ForkJoinQuiescingExecutorTest {

  @Test
  public void testExecuteFromTaskForksInSamePool() throws Exception {
    // Spy as an easy way to track calls to #execute.
    ForkJoinPool forkJoinPool = spy(new ForkJoinPool());
    try {
      ForkJoinQuiescingExecutor underTest =
          ForkJoinQuiescingExecutor.newBuilder().withOwnershipOf(forkJoinPool).build();

      AtomicReference<ForkJoinPool> subtaskRanIn = new AtomicReference<>();
      Runnable subTask = () -> subtaskRanIn.set(ForkJoinTask.getPool());

      AtomicReference<ForkJoinPool> taskRanIn = new AtomicReference<>();
      underTest.execute(
          () -> {
            taskRanIn.set(ForkJoinTask.getPool());
            underTest.execute(subTask);
          });
      underTest.awaitQuiescence(/*interruptWorkers=*/ false);

      assertThat(taskRanIn.get()).isSameInstanceAs(forkJoinPool);
      assertThat(subtaskRanIn.get()).isSameInstanceAs(forkJoinPool);

      // Confirm only one thing (the first task) was submitted via execute, the other should have
      // gone through the ForkJoinTask#fork() machinery.
      verify(forkJoinPool, times(1)).execute(any(Runnable.class));
    } finally {
      // Avoid leaving dangling threads.
      forkJoinPool.shutdownNow();
    }
  }

  /** Confirm our fork-new-work-if-in-forkjoinpool logic works as expected. */
  @Test
  public void testExecuteFromTaskInDifferentPoolRunsInRightPool() throws Exception {
    ForkJoinPool forkJoinPool = new ForkJoinPool();
    ForkJoinPool otherForkJoinPool = new ForkJoinPool();
    try {
      ForkJoinQuiescingExecutor originalExecutor =
          ForkJoinQuiescingExecutor.newBuilder().withOwnershipOf(forkJoinPool).build();
      ForkJoinQuiescingExecutor otherExecutor =
          ForkJoinQuiescingExecutor.newBuilder().withOwnershipOf(otherForkJoinPool).build();

      AtomicReference<ForkJoinPool> subtaskRanIn = new AtomicReference<>();
      Runnable subTask = () -> subtaskRanIn.set(ForkJoinTask.getPool());

      AtomicReference<ForkJoinPool> taskRanIn = new AtomicReference<>();
      originalExecutor.execute(
          () -> {
            taskRanIn.set(ForkJoinTask.getPool());
            otherExecutor.execute(subTask);
          });

      originalExecutor.awaitQuiescence(/*interruptWorkers=*/ false);
      otherExecutor.awaitQuiescence(/*interruptWorkers=*/ false);

      assertThat(taskRanIn.get()).isSameInstanceAs(forkJoinPool);
      assertThat(subtaskRanIn.get()).isSameInstanceAs(otherForkJoinPool);
    } finally {
      // Avoid leaving dangling threads.
      forkJoinPool.shutdownNow();
      otherForkJoinPool.shutdownNow();
    }
  }

  @Test
  public void testAwaitTerminationShutsDownPool() throws Exception {
    ForkJoinPool forkJoinPool = new ForkJoinPool();
    try {
      ForkJoinQuiescingExecutor underTest =
          ForkJoinQuiescingExecutor.newBuilder().withOwnershipOf(forkJoinPool).build();

      underTest.awaitTermination(/*interruptWorkers=*/ false);

      assertThat(forkJoinPool.isTerminated()).isTrue();
    } finally {
      // Avoid leaving dangling threads.
      forkJoinPool.shutdownNow();
    }
  }
}
