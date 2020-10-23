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
package com.google.devtools.build.lib.concurrent;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * A helper class for performing a custom visitation on the Skyframe graph, using {@link
 * QuiescingExecutor}.
 *
 * <p>The visitor uses an AbstractQueueVisitor backed by a ThreadPoolExecutor with a thread pool NOT
 * part of the global query evaluation pool to avoid starvation.
 *
 * <p>The visitation starts with {@link InputT}s via {@link #visitAndWaitForCompletion} which is
 * then converted to {@link VisitKeyT} through {@link #preprocessInitialVisit}.
 *
 * @param <InputT> the type of objects provided to initialize visitation
 * @param <VisitKeyT> the type of objects to visit
 * @param <OutputKeyT> the type of the key used to reference a result value
 * @param <OutputResultT> the type of visitation results to process
 * @param <ExceptionT> the exception type that can be thrown during visitation and the callback
 * @param <CallbackT> the callback type accepting {@code OutputResultT} and may throw {@code
 *     ExceptionT}
 */
@ThreadSafe
public abstract class ParallelVisitor<
    InputT,
    VisitKeyT,
    OutputKeyT,
    OutputResultT,
    ExceptionT extends Exception,
    CallbackT extends BatchCallback<OutputResultT, ExceptionT>> {
  protected final CallbackT callback;
  protected final Class<ExceptionT> exceptionClass;
  private final int visitBatchSize;
  private final int processResultsBatchSize;
  protected final int resultBatchSize;
  private final VisitingTaskExecutor executor;
  private final VisitTaskStatusCallback visitTaskStatusCallback;

  /**
   * A queue to store pending visits. These should be unique wrt {@link
   * #noteAndReturnUniqueVisitationKeys}.
   */
  private final LinkedBlockingQueue<VisitKeyT> visitQueue = new LinkedBlockingQueue<>();

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
  private final long minPendingTasks;

  /**
   * Fail fast on RuntimeExceptions, including {@code RuntimeInterruptedException} and {@code
   * RuntimeCheckedException}, which result from InterruptedException and {@code <ExceptionT>}.
   *
   * <p>Doesn't log for {@code RuntimeInterruptedException}, which is expected when evaluations are
   * interrupted, or {@code RuntimeCheckedException}, which happens when expected visitation
   * failures occur.
   */
  private static final ErrorClassifier PARALLEL_VISITOR_ERROR_CLASSIFIER =
      new ErrorClassifier() {
        @Override
        protected ErrorClassification classifyException(Exception e) {
          if (e instanceof RuntimeInterruptedException
              || e instanceof ParallelVisitor.RuntimeCheckedException) {
            return ErrorClassification.CRITICAL;
          } else if (e instanceof RuntimeException) {
            return ErrorClassification.CRITICAL_AND_LOG;
          } else {
            return ErrorClassification.NOT_CRITICAL;
          }
        }
      };

  protected ParallelVisitor(
      CallbackT callback,
      Class<ExceptionT> exceptionClass,
      int visitBatchSize,
      int processResultsBatchSize,
      long minPendingTasks,
      int batchCallbackSize,
      ExecutorService executor,
      VisitTaskStatusCallback visitTaskStatusCallback) {
    this.callback = callback;
    this.exceptionClass = exceptionClass;
    this.visitBatchSize = visitBatchSize;
    this.processResultsBatchSize = processResultsBatchSize;
    this.resultBatchSize = batchCallbackSize;
    this.visitTaskStatusCallback = visitTaskStatusCallback;
    this.executor =
        new VisitingTaskExecutor(executor, PARALLEL_VISITOR_ERROR_CLASSIFIER, batchCallbackSize);
    this.minPendingTasks = minPendingTasks;
  }

  /** Factory for {@link ParallelVisitor} instances. */
  public interface Factory<
      InputT,
      VisitKeyT,
      OutputKeyT,
      OutputResultT,
      ExceptionT extends Exception,
      CallbackT extends BatchCallback<OutputResultT, ExceptionT>> {
    ParallelVisitor<InputT, VisitKeyT, OutputKeyT, OutputResultT, ExceptionT, CallbackT> create();
  }

  /** A hook for getting notified when a visitation is discovered or completed. */
  public interface VisitTaskStatusCallback {
    void onVisitTaskDiscovered();

    void onVisitTaskCompleted();

    VisitTaskStatusCallback NULL_INSTANCE =
        new VisitTaskStatusCallback() {
          @Override
          public void onVisitTaskDiscovered() {}

          @Override
          public void onVisitTaskCompleted() {}
        };
  }

  protected abstract Iterable<OutputResultT> outputKeysToOutputValues(
      Iterable<OutputKeyT> targetKeys) throws ExceptionT, InterruptedException;

  /**
   * Suitable exception type to use with {@link ParallelVisitor} when no checked exception is
   * appropriate.
   */
  public static final class UnusedException extends RuntimeException {}

  /** An object to hold keys to visit and keys ready for processing. */
  protected final class Visit {
    private final Iterable<OutputKeyT> keysToUseForResult;
    private final Iterable<VisitKeyT> keysToVisit;

    public Visit(Iterable<OutputKeyT> keysToUseForResult, Iterable<VisitKeyT> keysToVisit) {
      this.keysToUseForResult = keysToUseForResult;
      this.keysToVisit = keysToVisit;
    }
  }

  public void visitAndWaitForCompletion(Iterable<InputT> keys)
      throws ExceptionT, InterruptedException {
    noteAndReturnUniqueVisitationKeys(preprocessInitialVisit(keys)).forEach(this::addToVisitQueue);
    executor.visitAndWaitForCompletion();
  }

  /** Gets the {@link Visit} representing the local visitation of the given {@code values}. */
  protected abstract Visit getVisitResult(Iterable<VisitKeyT> values)
      throws ExceptionT, InterruptedException;

  /** Transforms the initial input {@link InputT}s to {@link VisitKeyT} to start the visitation. */
  protected abstract Iterable<VisitKeyT> preprocessInitialVisit(Iterable<InputT> inputs);

  /**
   * Returns the values that have never been visited before in {@link #getVisitResult}.
   *
   * <p>Used to dedupe visitations before adding them to {@link #visitQueue}.
   */
  protected abstract Iterable<VisitKeyT> noteAndReturnUniqueVisitationKeys(
      Iterable<VisitKeyT> prospectiveVisitationKeys) throws ExceptionT;

  /** Gets tasks to visit pending keys. */
  protected Iterable<Task<ExceptionT>> getVisitTasks(Collection<VisitKeyT> pendingKeysToVisit)
      throws InterruptedException, ExceptionT {
    ImmutableList.Builder<Task<ExceptionT>> builder = ImmutableList.builder();
    for (Iterable<VisitKeyT> keysToVisitBatch :
        Iterables.partition(pendingKeysToVisit, visitBatchSize)) {
      builder.add(new VisitTask(keysToVisitBatch, exceptionClass));
    }

    return builder.build();
  }

  private void addToVisitQueue(VisitKeyT visitKey) {
    visitQueue.add(visitKey);
    visitTaskStatusCallback.onVisitTaskDiscovered();
  }

  /** A {@link Runnable} which handles {@link ExceptionT} and {@link InterruptedException}. */
  protected abstract static class Task<ExceptionT extends Exception> implements Runnable {
    protected final Class<ExceptionT> exceptionClass;

    Task(Class<ExceptionT> exceptionClass) {
      this.exceptionClass = exceptionClass;
    }

    @Override
    public void run() {
      try {
        process();
      } catch (InterruptedException e) {
        throw new RuntimeInterruptedException(e);
      } catch (RuntimeException e) {
        // Rethrow all RuntimeExceptions so they aren't caught by the following "catch Exception".
        throw e;
      } catch (Exception e) {
        // We can't "catch (ExceptionT e)" in Java. Instead we catch all checked exceptions and
        // double-check the type at runtime using the real ExceptionT class object.
        Preconditions.checkArgument(
            exceptionClass.isInstance(e),
            "got checked exception type %s, expected %s. Thrown exception: %s\nStack Trace: %s",
            e.getClass(),
            exceptionClass,
            e.getMessage(),
            Throwables.getStackTraceAsString(e));
        throw new RuntimeCheckedException(e);
      }
    }

    abstract void process() throws ExceptionT, InterruptedException;
  }

  /** A task to visit a batch of {@link VisitKeyT keys}. */
  public class VisitTask extends Task<ExceptionT> {
    private final Iterable<VisitKeyT> keysToVisit;

    public VisitTask(Iterable<VisitKeyT> keysToVisit, Class<ExceptionT> exceptionClass) {
      super(exceptionClass);
      this.keysToVisit = keysToVisit;
    }

    @Override
    void process() throws ExceptionT, InterruptedException {
      Visit visit = getVisitResult(keysToVisit);
      for (Iterable<OutputKeyT> keysToUseForResultBatch :
          Iterables.partition(visit.keysToUseForResult, processResultsBatchSize)) {
        executor.execute(
            new GetAndProcessUniqueResultsTask(keysToUseForResultBatch, exceptionClass));
      }
      noteAndReturnUniqueVisitationKeys(visit.keysToVisit)
          .forEach(ParallelVisitor.this::addToVisitQueue);
      keysToVisit.forEach(
          key -> ParallelVisitor.this.visitTaskStatusCallback.onVisitTaskCompleted());
    }
  }

  private class GetAndProcessUniqueResultsTask extends Task<ExceptionT> {
    private final Iterable<OutputKeyT> uniqueKeysToUseForResult;

    private GetAndProcessUniqueResultsTask(
        Iterable<OutputKeyT> uniqueKeysToUseForResult, Class<ExceptionT> exceptionClass) {
      super(exceptionClass);
      this.uniqueKeysToUseForResult = uniqueKeysToUseForResult;
    }

    @Override
    protected void process() throws ExceptionT, InterruptedException {
      callback.process(outputKeysToOutputValues(uniqueKeysToUseForResult));
    }
  }

  /**
   * A custom implementation of {@link QuiescingExecutor} which uses a centralized queue and
   * scheduler for parallel visitations.
   */
  private class VisitingTaskExecutor extends AbstractQueueVisitor {
    private final int batchCallbackSize;

    private VisitingTaskExecutor(
        ExecutorService executor, ErrorClassifier errorClassifier, int batchCallbackSize) {
      super(
          /*executorService=*/ executor,
          // Leave the thread pool active for other current and future callers.
          /*shutdownOnCompletion=*/ false,
          /*failFastOnException=*/ true,
          /*errorClassifier=*/ errorClassifier);
      this.batchCallbackSize = batchCallbackSize;
    }

    private void visitAndWaitForCompletion() throws ExceptionT, InterruptedException {
      // The scheduler keeps running until either of the following two conditions are met.
      //
      // 1. Errors (ExceptionT or InterruptedException) occurred and visitations should fail
      //    fast.
      // 2. There is no pending visit in the queue and no pending task running.
      while (!mustJobsBeStopped() && moreWorkToDo()) {
        // To achieve maximum efficiency, queue is drained in either of the following two
        // conditions:
        //
        // 1. The number of pending tasks is low. We schedule new tasks to avoid wasting CPU.
        // 2. The process queue size is large.
        if (getTaskCount() < minPendingTasks || visitQueue.size() >= batchCallbackSize) {

          Collection<VisitKeyT> pendingKeysToVisit = new ArrayList<>(visitQueue.size());
          visitQueue.drainTo(pendingKeysToVisit);
          for (Task<?> task : getVisitTasks(pendingKeysToVisit)) {
            execute(task);
          }
        }

        try {
          Thread.sleep(SCHEDULING_INTERVAL_MILLISECONDS);
        } catch (InterruptedException e) {
          // If the main thread waiting for completion of the visitation is interrupted, we should
          // gracefully terminate all running and pending tasks before exit. If ExceptionT
          // occurred in any of the worker thread, awaitTerminationAndPropagateErrorsIfAny
          // propagates the ExceptionT instead of InterruptedException.
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

    private boolean moreWorkToDo() {
      // Note that we must check the task count first -- checking the processing queue first has the
      // following race condition:
      // (1) Check processing queue and observe that it is empty
      // (2) A remaining task adds to the processing queue and shuts down
      // (3) We check the task count and observe it is empty
      return getTaskCount() > 0 || !visitQueue.isEmpty();
    }

    // We check against Class<ExceptionT> before creating RuntimeCheckedException.
    @SuppressWarnings("unchecked")
    private void awaitTerminationAndPropagateErrorsIfAny() throws ExceptionT, InterruptedException {
      try {
        awaitTermination(/*interruptWorkers=*/ true);
      } catch (RuntimeCheckedException e) {
        throw (ExceptionT) e.getCause();
      } catch (RuntimeInterruptedException e) {
        throw (InterruptedException) e.getCause();
      }
    }
  }

  private static class RuntimeCheckedException extends RuntimeException {
    private RuntimeCheckedException(Exception checkedException) {
      super(checkedException);
    }
  }

  private static class RuntimeInterruptedException extends RuntimeException {
    private RuntimeInterruptedException(InterruptedException interruptedException) {
      super(interruptedException);
    }
  }
}
