// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import javax.annotation.Nullable;

/** A {@link BlazeModule} that waits for submitted tasks to terminate after every command. */
public class BlockWaitingModule extends BlazeModule {

  /** A task to be submitted. */
  public interface Task {
    void call() throws AbruptExitException;
  }

  /**
   * Wraps an AbruptExitException thrown by a task.
   *
   * <p>This is needed because a task that can throw a checked exception cannot be submitted to
   * {@link ExecutorService}.
   */
  private static class TaskException extends RuntimeException {
    TaskException(AbruptExitException cause) {
      super(cause);
    }
  }

  @Nullable private ExecutorService executorService;
  @Nullable private ArrayList<Future<?>> submittedTasks;

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    checkState(executorService == null, "executorService must be null");
    checkState(submittedTasks == null, "submittedTasks must be null");

    executorService =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("block-waiting-%d").build());

    submittedTasks = new ArrayList<>();
  }

  @SuppressWarnings("FutureReturnValueIgnored")
  public void submit(Task task) {
    checkNotNull(executorService, "executorService must not be null");
    checkNotNull(submittedTasks, "submittedTasks must be null");

    submittedTasks.add(
        executorService.submit(
            () -> {
              try {
                task.call();
              } catch (AbruptExitException e) {
                throw new TaskException(e);
              }
            }));
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    checkNotNull(executorService, "executorService must not be null");

    if (ExecutorUtil.interruptibleShutdown(executorService)) {
      Thread.currentThread().interrupt();
    }

    for (Future<?> f : submittedTasks) {
      try {
        f.get(); // guaranteed to have completed.
      } catch (InterruptedException e) {
        throw new AssertionError("task should not have been interrupted");
      } catch (ExecutionException e) {
        Throwable cause = e.getCause();
        if (cause instanceof TaskException) {
          checkState(cause.getCause() instanceof AbruptExitException);
          throw (AbruptExitException) cause.getCause();
        }
        throw new RuntimeException(e);
      }
    }

    executorService = null;
    submittedTasks = null;
  }
}
