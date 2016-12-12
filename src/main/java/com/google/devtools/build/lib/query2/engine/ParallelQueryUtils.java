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

import com.google.devtools.build.lib.concurrent.MoreFutures;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

/** Several utilities to aid in writing {@link QueryExpression#parEval} implementations. */
public class ParallelQueryUtils {
  /**
   * Encapsulation of a subtask of parallel evaluation of a {@link QueryExpression}. See
   * {@link #executeQueryTasksAndWaitInterruptibly}.
   */
  public interface QueryTask {
    void execute() throws QueryException, InterruptedException;
  }

  /**
   * Executes the given {@link QueryTask}s using the given {@link ForkJoinPool} and interruptibly
   * waits for their completion. Throws the first {@link QueryException} or
   * {@link InterruptedException} encountered during parallel execution.
   */
  public static void executeQueryTasksAndWaitInterruptibly(
      List<QueryTask> queryTasks,
      ForkJoinPool forkJoinPool) throws QueryException, InterruptedException {
    ArrayList<QueryTaskForkJoinTask> forkJoinTasks = new ArrayList<>(queryTasks.size());
    for (QueryTask queryTask : queryTasks) {
      QueryTaskForkJoinTask forkJoinTask = adaptAsForkJoinTask(queryTask);
      forkJoinTasks.add(forkJoinTask);
      forkJoinPool.submit(forkJoinTask);
    }
    try {
      MoreFutures.waitForAllInterruptiblyFailFast(forkJoinTasks);
    } catch (ExecutionException e) {
      throw rethrowCause(e);
    }
  }

  private static QueryTaskForkJoinTask adaptAsForkJoinTask(QueryTask queryTask) {
    return new QueryTaskForkJoinTask(queryTask);
  }

  private static RuntimeException rethrowCause(ExecutionException e)
      throws QueryException, InterruptedException {
    Throwable cause = e.getCause();
    if (cause instanceof ParallelRuntimeException) {
      ((ParallelRuntimeException) cause).rethrow();
    }
    throw new IllegalStateException(e);
  }

  // ForkJoinTask#adapt(Callable) wraps thrown checked exceptions as RuntimeExceptions. We avoid
  // having to think about that messiness (which is inconsistent with other Future implementations)
  // by having our own ForkJoinTask subclass and managing checked exceptions ourselves.
  private static class QueryTaskForkJoinTask extends ForkJoinTask<Void> {
    private final QueryTask queryTask;

    private QueryTaskForkJoinTask(QueryTask queryTask) {
      this.queryTask = queryTask;
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
      try {
        queryTask.execute();
      } catch (QueryException queryException) {
        throw new ParallelRuntimeQueryException(queryException);
      } catch (InterruptedException interruptedException) {
        throw new ParallelInterruptedQueryException(interruptedException);
      }
      return true;
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
