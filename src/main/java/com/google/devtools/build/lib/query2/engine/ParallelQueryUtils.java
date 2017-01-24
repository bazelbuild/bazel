// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.MoreFutures;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.Future;

/** Several utilities to aid in writing {@link QueryExpression#parEvalImpl} implementations. */
public class ParallelQueryUtils {
  /**
   * Encapsulation of a subtask of parallel evaluation of a {@link QueryExpression}. See
   * {@link #executeQueryTasksAndWaitInterruptiblyFailFast}.
   */
  @ThreadSafe
  public interface QueryTask {
    void execute() throws QueryException, InterruptedException;
  }

  /**
   * Executes the given {@link QueryTask}s using the given {@link ForkJoinPool} and interruptibly
   * waits for their completion. Throws the first {@link QueryException} encountered during parallel
   * execution or an {@link InterruptedException} if the calling thread is interrupted.
   *
   * <p>These "fail-fast" semantics are desirable to avoid doing unneeded work when evaluating
   * multiple {@link QueryTask}s in parallel: if serial execution of the tasks would result in a
   * {@link QueryException} then we want parallel execution to do so as well, but there's no need to
   * continue waiting for completion of the tasks after at least one of them results in a
   * {@link QueryException}.
   */
  public static void executeQueryTasksAndWaitInterruptiblyFailFast(
      List<QueryTask> queryTasks,
      ForkJoinPool forkJoinPool) throws QueryException, InterruptedException {
    int numTasks = queryTasks.size();
    if (numTasks == 1) {
      Iterables.getOnlyElement(queryTasks).execute();
      return;
    }
    FailFastCountDownLatch failFastLatch = new FailFastCountDownLatch(numTasks);
    ArrayList<QueryTaskForkJoinTask> forkJoinTasks = new ArrayList<>(numTasks);
    for (QueryTask queryTask : queryTasks) {
      QueryTaskForkJoinTask forkJoinTask = adaptAsForkJoinTask(queryTask, failFastLatch);
      forkJoinTasks.add(forkJoinTask);
      @SuppressWarnings("unused") 
      Future<?> possiblyIgnoredError = forkJoinPool.submit(forkJoinTask);
    }
    failFastLatch.await();
    try {
      MoreFutures.waitForAllInterruptiblyFailFast(forkJoinTasks);
    } catch (ExecutionException e) {
      throw rethrowCause(e);
    }
  }

  private static QueryTaskForkJoinTask adaptAsForkJoinTask(
      QueryTask queryTask,
      FailFastCountDownLatch failFastLatch) {
    return new QueryTaskForkJoinTask(queryTask, failFastLatch);
  }

  private static RuntimeException rethrowCause(ExecutionException e)
      throws QueryException, InterruptedException {
    Throwable cause = e.getCause();
    if (cause instanceof ParallelRuntimeException) {
      ((ParallelRuntimeException) cause).rethrow();
    }
    throw new IllegalStateException(e);
  }

  /**
   * Wrapper around a {@link CountDownLatch} with initial count {@code n} that counts down once on
   * "success" and {@code n} times on "failure".
   *
   * <p>This can be used in a concurrent context to wait until either {@code n} tasks are successful
   * or at least one of them fails.
   */
  @ThreadSafe
  private static class FailFastCountDownLatch {
    private final int n;
    private final CountDownLatch completionLatch;

    private FailFastCountDownLatch(int n) {
      this.n = n;
      this.completionLatch = new CountDownLatch(n);
    }

    private void await() throws InterruptedException {
      completionLatch.await();
    }

    private void countDown(boolean success) {
      if (success) {
        completionLatch.countDown();
      } else {
        for (int i = 0; i < n; i++) {
          completionLatch.countDown();
        }
      }
    }
  }

  // ForkJoinTask#adapt(Callable) wraps thrown checked exceptions as RuntimeExceptions. We avoid
  // having to think about that messiness (which is inconsistent with other Future implementations)
  // by having our own ForkJoinTask subclass and managing checked exceptions ourselves.
  @ThreadSafe
  private static class QueryTaskForkJoinTask extends ForkJoinTask<Void> {
    private final QueryTask queryTask;
    private final FailFastCountDownLatch completionLatch;

    private QueryTaskForkJoinTask(QueryTask queryTask, FailFastCountDownLatch completionLatch) {
      this.queryTask = queryTask;
      this.completionLatch = completionLatch;
    }

    @Override
    public Void getRawResult() {
      return null;
    }

    @Override
    protected void setRawResult(Void value) {
    }

    @Override
    protected boolean exec() {
      boolean successful = false;
      try {
        queryTask.execute();
        successful = true;
        return true;
      } catch (QueryException queryException) {
        throw new ParallelRuntimeQueryException(queryException);
      } catch (InterruptedException interruptedException) {
        throw new ParallelInterruptedQueryException(interruptedException);
      } finally {
        completionLatch.countDown(successful);
      }
    }
  }

  private abstract static class ParallelRuntimeException extends RuntimeException {
    abstract void rethrow() throws QueryException, InterruptedException;
  }

  private static class ParallelRuntimeQueryException extends ParallelRuntimeException {
    private final QueryException queryException;

    private ParallelRuntimeQueryException(QueryException queryException) {
      this.queryException = queryException;
    }

    @Override
    void rethrow() throws QueryException, InterruptedException {
      throw queryException;
    }
  }

  private static class ParallelInterruptedQueryException extends ParallelRuntimeException {
    private final InterruptedException interruptedException;

    private ParallelInterruptedQueryException(InterruptedException interruptedException) {
      this.interruptedException = interruptedException;
    }

    @Override
    void rethrow() throws QueryException, InterruptedException {
      throw interruptedException;
    }
  }
}
