// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.BlockingStack;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * A helper class for performing a custom visitation on the Skyframe graph, using {@link
 * QuiescingExecutor}.
 *
 * <p>The visitor uses an AbstractQueueVisitor backed by a ThreadPoolExecutor with a thread pool NOT
 * part of the global query evaluation pool to avoid starvation.
 *
 * @param <T> the type of objects to visit
 * @param <V> the type of visitation results to process
 */
@ThreadSafe
public abstract class ParallelVisitor<T, V> {
  private final Callback<V> callback;
  private final int visitBatchSize;
  private final int processResultsBatchSize;

  private final VisitingTaskExecutor executor;

  /** A queue to store pending visits. These should be unique wrt {@link #getUniqueValues}. */
  private final LinkedBlockingQueue<T> processingQueue = new LinkedBlockingQueue<>();

  /**
   * The max time interval between two scheduling passes in milliseconds. A scheduling pass is
   * defined as the scheduler thread determining whether to drain all pending visits from the queue
   * and submitting tasks to perform the visits.
   *
   * <p>The choice of 1ms is a result based of experiments. It is an attempted balance due to a few
   * facts about the scheduling interval:
   *
   * <p>1. A large interval adds systematic delay. In an extreme case, a visit which is supposed to
   * take only 1ms now may take 5ms. For most visits which take longer than a few hundred
   * milliseconds, it should not be noticeable.
   *
   * <p>2. A zero-interval config eats too much CPU.
   *
   * <p>Even though the scheduler runs once every 1 ms, it does not try to drain it every time.
   * Pending visits are drained only certain criteria are met.
   */
  private static final long SCHEDULING_INTERVAL_MILLISECONDS = 1;

  /**
   * The minimum number of pending tasks the scheduler tries to hit. The 3x number is set based on
   * experiments. We do not want to schedule tasks too frequently to miss the benefits of large
   * number of keys being grouped by packages. On the other hand, we want to keep all threads in the
   * pool busy to achieve full capacity. A low number here will cause some of the worker threads to
   * go idle at times before the next scheduling cycle.
   *
   * <p>TODO(shazh): Revisit the choice of task target based on real-prod performance.
   */
  private static final long MIN_PENDING_TASKS = 3L * SkyQueryEnvironment.DEFAULT_THREAD_COUNT;

  /**
   * Fail fast on RuntimeExceptions, including {@code RuntimeInterruptedException} and {@code
   * RuntimeQueryException}, which result from InterruptedException and QueryException.
   *
   * <p>Doesn't log for {@code RuntimeInterruptedException}, which is expected when evaluations are
   * interrupted, or {@code RuntimeQueryException}, which happens when expected query failures
   * occur.
   */
  static final ErrorClassifier PARALLEL_VISITOR_ERROR_CLASSIFIER =
      new ErrorClassifier() {
        @Override
        protected ErrorClassification classifyException(Exception e) {
          if (e instanceof RuntimeInterruptedException || e instanceof RuntimeQueryException) {
            return ErrorClassification.CRITICAL;
          } else if (e instanceof RuntimeException) {
            return ErrorClassification.CRITICAL_AND_LOG;
          } else {
            return ErrorClassification.NOT_CRITICAL;
          }
        }
      };

  /** All visitors share a single global fixed thread pool. */
  private static final ExecutorService FIXED_THREAD_POOL_EXECUTOR =
      new ThreadPoolExecutor(
          /*corePoolSize=*/ Math.max(1, SkyQueryEnvironment.DEFAULT_THREAD_COUNT),
          /*maximumPoolSize=*/ Math.max(1, SkyQueryEnvironment.DEFAULT_THREAD_COUNT),
          /*keepAliveTime=*/ 1,
          /*units=*/ TimeUnit.SECONDS,
          /*workQueue=*/ new BlockingStack<Runnable>(),
          new ThreadFactoryBuilder().setNameFormat("parallel-visitor %d").build());

  protected ParallelVisitor(
      Callback<V> callback,
      int visitBatchSize,
      int processResultsBatchSize) {
    this.callback = callback;
    this.visitBatchSize = visitBatchSize;
    this.processResultsBatchSize = processResultsBatchSize;
    this.executor =
        new VisitingTaskExecutor(FIXED_THREAD_POOL_EXECUTOR, PARALLEL_VISITOR_ERROR_CLASSIFIER);
  }

  /** Factory for {@link ParallelVisitor} instances. */
  public interface Factory {
    ParallelVisitor<?, ?> create();
  }

  /**
   * Returns a {@link Callback} which kicks off a parallel visitation when {@link Callback#process}
   * is invoked.
   */
  public static Callback<Target> createParallelVisitorCallback(Factory visitorFactory) {
    return new ParallelVisitorCallback(visitorFactory);
  }

  /** An object to hold keys to visit and keys ready for processing. */
  protected final class Visit {
    private final Iterable<SkyKey> keysToUseForResult;
    private final Iterable<T> keysToVisit;

    public Visit(Iterable<SkyKey> keysToUseForResult, Iterable<T> keysToVisit) {
      this.keysToUseForResult = keysToUseForResult;
      this.keysToVisit = keysToVisit;
    }
  }

  public void visitAndWaitForCompletion(Iterable<SkyKey> keys)
      throws QueryException, InterruptedException {
    getUniqueValues(preprocessInitialVisit(keys)).forEach(processingQueue::add);
    executor.visitAndWaitForCompletion();
  }

  /**
   * Forwards the given {@code keysToUseForResult}'s contribution to the set of results in the full
   * visitation to the given {@link Callback}.
   */
  protected abstract void processPartialResults(
      Iterable<SkyKey> keysToUseForResult, Callback<V> callback)
      throws QueryException, InterruptedException;

  /** Gets the {@link Visit} representing the local visitation of the given {@code values}. */
  protected abstract Visit getVisitResult(Iterable<T> values)
      throws QueryException, InterruptedException;

  /** Gets the equivalent of {@link Visit#keysToVisit} for the entry-level SkyKeys. */
  protected abstract Iterable<T> preprocessInitialVisit(Iterable<SkyKey> keys);

  /**
   * Returns the values that have never been visited before in {@link #getVisitResult}.
   *
   * <p>Used to dedupe visitations before adding them to {@link #processingQueue}.
   */
  protected abstract ImmutableList<T> getUniqueValues(Iterable<T> values) throws QueryException;

  /** Gets tasks to visit pending keys. */
  protected Iterable<Task> getVisitTasks(Collection<T> pendingKeysToVisit) {
    ImmutableList.Builder<Task> builder = ImmutableList.builder();
    for (Iterable<T> keysToVisitBatch : Iterables.partition(pendingKeysToVisit, visitBatchSize)) {
      builder.add(new VisitTask(keysToVisitBatch));
    }

    return builder.build();
  }

  /** A {@link Runnable} which handles {@link QueryException} and {@link InterruptedException}. */
  protected abstract static class Task implements Runnable {

    @Override
    public void run() {
      try {
        process();
      } catch (QueryException e) {
        throw new RuntimeQueryException(e);
      } catch (InterruptedException e) {
        throw new RuntimeInterruptedException(e);
      }
    }

    abstract void process() throws QueryException, InterruptedException;
  }

  class VisitTask extends Task {
    private final Iterable<T> keysToVisit;

    VisitTask(Iterable<T> keysToVisit) {
      this.keysToVisit = keysToVisit;
    }

    @Override
    void process() throws QueryException, InterruptedException {
      Visit visit = getVisitResult(keysToVisit);
      for (Iterable<SkyKey> keysToUseForResultBatch :
          Iterables.partition(visit.keysToUseForResult, processResultsBatchSize)) {
        executor.execute(new GetAndProcessResultsTask(keysToUseForResultBatch));
      }

      getUniqueValues(visit.keysToVisit).forEach(processingQueue::add);
    }
  }

  private class GetAndProcessResultsTask extends Task {
    private final Iterable<SkyKey> keysToUseForResult;

    private GetAndProcessResultsTask(Iterable<SkyKey> keysToUseForResult) {
      this.keysToUseForResult = keysToUseForResult;
    }

    @Override
    protected void process() throws QueryException, InterruptedException {
      processPartialResults(keysToUseForResult, callback);
    }
  }

  /**
   * A custom implementation of {@link QuiescingExecutor} which uses a centralized queue and
   * scheduler for parallel visitations.
   */
  private class VisitingTaskExecutor extends AbstractQueueVisitor {
    private VisitingTaskExecutor(ExecutorService executor, ErrorClassifier errorClassifier) {
      super(
          /*executorService=*/ executor,
          // Leave the thread pool active for other current and future callers.
          /*shutdownOnCompletion=*/ false,
          /*failFastOnException=*/ true,
          /*errorClassifier=*/ errorClassifier);
    }

    private void visitAndWaitForCompletion() throws QueryException, InterruptedException {
      // The scheduler keeps running until either of the following two conditions are met.
      //
      // 1. Errors (QueryException or InterruptedException) occurred and visitations should fail
      //    fast.
      // 2. There is no pending visit in the queue and no pending task running.
      while (!mustJobsBeStopped() && (!processingQueue.isEmpty() || getTaskCount() > 0)) {
        // To achieve maximum efficiency, queue is drained in either of the following two
        // conditions:
        //
        // 1. The number of pending tasks is low. We schedule new tasks to avoid wasting CPU.
        // 2. The process queue size is large.
        if (getTaskCount() < MIN_PENDING_TASKS
            || processingQueue.size() >= SkyQueryEnvironment.BATCH_CALLBACK_SIZE) {

          Collection<T> pendingKeysToVisit = new ArrayList<>(processingQueue.size());
          processingQueue.drainTo(pendingKeysToVisit);
          for (Task task : getVisitTasks(pendingKeysToVisit)) {
            execute(task);
          }
        }

        try {
          Thread.sleep(SCHEDULING_INTERVAL_MILLISECONDS);
        } catch (InterruptedException e) {
          // If the main thread waiting for completion of the visitation is interrupted, we should
          // gracefully terminate all running and pending tasks before exit. If QueryException
          // occurred in any of the worker thread, awaitTerminationAndPropagateErrorsIfAny
          // propagates the QueryException instead of InterruptedException.
          setInterrupted();
          awaitTerminationAndPropagateErrorsIfAny();
        }
      }

      // We reach here either because the visitation is complete, or because an error prevents us
      // from proceeding with the visitation. awaitTerminationAndPropagateErrorsIfAny will either
      // gracefully exit if the visitation is complete, or propagate the exception if error
      // occurred.
      awaitTerminationAndPropagateErrorsIfAny();
    }

    private void awaitTerminationAndPropagateErrorsIfAny()
        throws QueryException, InterruptedException {
      try {
        awaitTermination(/*interruptWorkers=*/ true);
      } catch (RuntimeQueryException e) {
        throw (QueryException) e.getCause();
      } catch (RuntimeInterruptedException e) {
        throw (InterruptedException) e.getCause();
      }
    }
  }

  /**
   * A {@link Callback} whose {@link Callback#process} method kicks off a visitation via a fresh
   * {@link ParallelVisitor} instance.
   */
  private static class ParallelVisitorCallback implements Callback<Target> {
    private final ParallelVisitor.Factory visitorFactory;

    private ParallelVisitorCallback(ParallelVisitor.Factory visitorFactory) {
      this.visitorFactory = visitorFactory;
    }

    @Override
    public void process(Iterable<Target> partialResult)
        throws QueryException, InterruptedException {
      ParallelVisitor<?, ?> visitor = visitorFactory.create();
      // TODO(nharmata): It's not ideal to have an operation like this in #process that blocks on
      // another, potentially expensive computation. Refactor to something like "processAsync".
      visitor.visitAndWaitForCompletion(
          SkyQueryEnvironment.makeTransitiveTraversalKeysStrict(partialResult));
    }
  }

  private static class RuntimeQueryException extends RuntimeException {
    private RuntimeQueryException(QueryException queryException) {
      super(queryException);
    }
  }

  private static class RuntimeInterruptedException extends RuntimeException {
    private RuntimeInterruptedException(InterruptedException interruptedException) {
      super(interruptedException);
    }
  }
}
