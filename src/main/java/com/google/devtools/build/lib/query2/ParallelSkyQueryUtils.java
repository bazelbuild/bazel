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
package com.google.devtools.build.lib.query2;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.BlockingStack;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.ThreadSafeCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeUniquifier;
import com.google.devtools.build.lib.query2.engine.VariableContext;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Parallel implementations of various functionality in {@link SkyQueryEnvironment}.
 *
 * <p>Special attention is given to memory usage. Naive parallel implementations of query
 * functionality would lead to memory blowup. Instead of dealing with {@link Target}s, we try to
 * deal with {@link SkyKey}s as much as possible to reduce the number of {@link Package}s forcibly
 * in memory at any given time.
 */
// TODO(bazel-team): Be more deliberate about bounding memory usage here.
class ParallelSkyQueryUtils {

  /** The maximum number of keys to visit at once. */
  @VisibleForTesting static final int VISIT_BATCH_SIZE = 10000;

  private ParallelSkyQueryUtils() {
  }

  /**
   * Specialized parallel variant of {@link SkyQueryEnvironment#getAllRdeps} that is appropriate
   * when there is no depth-bound.
   */
  static void getAllRdepsUnboundedParallel(
      SkyQueryEnvironment env,
      QueryExpression expression,
      VariableContext<Target> context,
      ThreadSafeCallback<Target> callback,
      MultisetSemaphore<PackageIdentifier> packageSemaphore)
          throws QueryException, InterruptedException {
    env.eval(
        expression,
        context,
        new SkyKeyBFSVisitorCallback(
            new AllRdepsUnboundedVisitor.Factory(env, callback, packageSemaphore)));
  }

  /** Specialized parallel variant of {@link SkyQueryEnvironment#getRBuildFiles}. */
  static void getRBuildFilesParallel(
      SkyQueryEnvironment env,
      Collection<PathFragment> fileIdentifiers,
      ThreadSafeCallback<Target> callback,
      MultisetSemaphore<PackageIdentifier> packageSemaphore)
          throws QueryException, InterruptedException {
    ThreadSafeUniquifier<SkyKey> keyUniquifier = env.createSkyKeyUniquifier();
    RBuildFilesVisitor visitor =
        new RBuildFilesVisitor(env, keyUniquifier, callback, packageSemaphore);
    visitor.visitAndWaitForCompletion(env.getSkyKeysForFileFragments(fileIdentifiers));
  }

  /** A helper class that computes 'rbuildfiles(<blah>)' via BFS. */
  private static class RBuildFilesVisitor extends AbstractSkyKeyBFSVisitor<SkyKey> {
    private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

    private RBuildFilesVisitor(
        SkyQueryEnvironment env,
        ThreadSafeUniquifier<SkyKey> uniquifier,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(env, uniquifier, callback);
      this.packageSemaphore = packageSemaphore;
    }

    @Override
    protected Visit getVisitResult(Iterable<SkyKey> values) throws InterruptedException {
      Collection<Iterable<SkyKey>> reverseDeps = env.graph.getReverseDeps(values).values();
      Set<SkyKey> keysToUseForResult = CompactHashSet.create();
      Set<SkyKey> keysToVisitNext = CompactHashSet.create();
      for (SkyKey rdep : Iterables.concat(reverseDeps)) {
        if (rdep.functionName().equals(SkyFunctions.PACKAGE)) {
          keysToUseForResult.add(rdep);
          // Every package has a dep on the external package, so we need to include those edges too.
          if (rdep.equals(PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER))) {
            keysToVisitNext.add(rdep);
          }
        } else if (!rdep.functionName().equals(SkyFunctions.PACKAGE_LOOKUP)) {
          // Packages may depend on the existence of subpackages, but these edges aren't relevant to
          // rbuildfiles.
          keysToVisitNext.add(rdep);
        }
      }
      return new Visit(keysToUseForResult, keysToVisitNext);
    }

    @Override
    protected void processResultantTargets(
        Iterable<SkyKey> keysToUseForResult, Callback<Target> callback)
            throws QueryException, InterruptedException {
      Set<PackageIdentifier> pkgIdsNeededForResult =
          ImmutableSet.copyOf(
              Iterables.transform(
                  keysToUseForResult,
                  SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER));
      packageSemaphore.acquireAll(pkgIdsNeededForResult);
      try {
        callback.process(SkyQueryEnvironment.getBuildFilesForPackageValues(
            env.graph.getSuccessfulValues(keysToUseForResult).values()));
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForResult);
      }
    }

    @Override
    protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> keys) {
      return keys;
    }
  }

  /**
   * A helper class that computes 'allrdeps(<blah>)' via BFS.
   *
   * <p>The visitor uses a pair of <node, reverse dep> to keep track the nodes to visit and avoid
   * dealing with targetification of reverse deps until they are needed. The node itself is needed
   * to filter out disallowed deps later. Compared against the approach using a single SkyKey, it
   * consumes 16 more bytes in a 64-bit environment for each edge. However it defers the need to
   * load all the packages which have at least a target as a rdep of the current batch, thus greatly
   * reduces the risk of OOMs. The additional memory usage should not be a large concern here, as
   * even with 10M edges, the memory overhead is around 160M, and the memory can be reclaimed by
   * regular GC.
   */
  private static class AllRdepsUnboundedVisitor
      extends AbstractSkyKeyBFSVisitor<Pair<SkyKey, SkyKey>> {
    private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

    private AllRdepsUnboundedVisitor(
        SkyQueryEnvironment env,
        ThreadSafeUniquifier<Pair<SkyKey, SkyKey>> uniquifier,
        ThreadSafeCallback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(env, uniquifier, callback);
      this.packageSemaphore = packageSemaphore;
    }

    /**
     * A {@link Factory} for {@link AllRdepsUnboundedVisitor} instances, each of which will be used
     * to perform visitation of the reverse transitive closure of the {@link Target}s passed in a
     * single {@link ThreadSafeCallback#process} call. Note that all the created
     * instances share the same {@code ThreadSafeUniquifier<SkyKey>} so that we don't visit the
     * same Skyframe node more than once.
     */
    private static class Factory implements AbstractSkyKeyBFSVisitor.Factory {
      private final SkyQueryEnvironment env;
      private final ThreadSafeUniquifier<Pair<SkyKey, SkyKey>> uniquifier;
      private final ThreadSafeCallback<Target> callback;
      private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

      private Factory(
        SkyQueryEnvironment env,
        ThreadSafeCallback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
        this.env = env;
        this.uniquifier = env.createReverseDepSkyKeyUniquifier();
        this.callback = callback;
        this.packageSemaphore = packageSemaphore;
      }

      @Override
      public AbstractSkyKeyBFSVisitor<Pair<SkyKey, SkyKey>> create() {
        return new AllRdepsUnboundedVisitor(env, uniquifier, callback, packageSemaphore);
      }
    }

    @Override
    protected Visit getVisitResult(Iterable<Pair<SkyKey, SkyKey>> keys)
        throws InterruptedException {
      Collection<SkyKey> filteredKeys = new ArrayList<>();

      // Build a raw reverse dep map from pairs of SkyKeys to filter out the disallowed deps.
      Map<SkyKey, Collection<SkyKey>> reverseDepsMap = Maps.newHashMap();
      for (Pair<SkyKey, SkyKey> reverseDepPair : keys) {
        // First-level nodes do not have a parent node (they may have one in Skyframe but we do not
        // need to retrieve them.
        if (reverseDepPair.first == null) {
          filteredKeys.add(Preconditions.checkNotNull(reverseDepPair.second));
          continue;
        }

        if (!reverseDepsMap.containsKey(reverseDepPair.first)) {
          reverseDepsMap.put(reverseDepPair.first, new LinkedList<SkyKey>());
        }

        reverseDepsMap.get(reverseDepPair.first).add(reverseDepPair.second);
      }

      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
          env.makePackageKeyToTargetKeyMap(Iterables.concat(reverseDepsMap.values()));
      Set<PackageIdentifier> pkgIdsNeededForTargetification =
          ImmutableSet.copyOf(
              Iterables.transform(
                  packageKeyToTargetKeyMap.keySet(),
                  SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER));
      packageSemaphore.acquireAll(pkgIdsNeededForTargetification);

      try {
        // Filter out disallowed deps. We cannot defer the targetification any further as we do not
        // want to retrieve the rdeps of unwanted nodes (targets).
        if (!reverseDepsMap.isEmpty()) {
          Collection<Target> filteredTargets =
              env.filterRawReverseDepsOfTransitiveTraversalKeys(
                  reverseDepsMap, packageKeyToTargetKeyMap);
          filteredKeys.addAll(
              Collections2.transform(filteredTargets, SkyQueryEnvironment.TARGET_TO_SKY_KEY));
        }
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForTargetification);
      }

      // Retrieve the reverse deps as SkyKeys and defer the targetification and filtering to next
      // recursive visitation.
      Map<SkyKey, Iterable<SkyKey>> unfilteredReverseDeps = env.graph.getReverseDeps(filteredKeys);

      ImmutableList.Builder<Pair<SkyKey, SkyKey>> builder = ImmutableList.builder();
      for (Map.Entry<SkyKey, Iterable<SkyKey>> rdeps : unfilteredReverseDeps.entrySet()) {
        for (SkyKey rdep : rdeps.getValue()) {
          Label label = SkyQueryEnvironment.SKYKEY_TO_LABEL.apply(rdep);
          if (label != null) {
            builder.add(Pair.of(rdeps.getKey(), rdep));
          }
        }
      }

      return new Visit(/*keysToUseForResult=*/ filteredKeys, /*keysToVisit=*/ builder.build());
    }

    @Override
    protected void processResultantTargets(
        Iterable<SkyKey> keysToUseForResult, Callback<Target> callback)
            throws QueryException, InterruptedException {
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
          env.makePackageKeyToTargetKeyMap(keysToUseForResult);
      Set<PackageIdentifier> pkgIdsNeededForResult =
          ImmutableSet.copyOf(
            Iterables.transform(
                packageKeyToTargetKeyMap.keySet(),
                SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER));
      packageSemaphore.acquireAll(pkgIdsNeededForResult);
      try {
        callback.process(
            env.makeTargetsFromPackageKeyToTargetKeyMap(packageKeyToTargetKeyMap).values());
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForResult);
      }
    }

    @Override
    protected Iterable<Pair<SkyKey, SkyKey>> preprocessInitialVisit(Iterable<SkyKey> keys) {
      return Iterables.transform(
          keys,
          new Function<SkyKey, Pair<SkyKey, SkyKey>>() {
            @Override
            public Pair<SkyKey, SkyKey> apply(SkyKey key) {
              // Set parent of first-level nodes to null. They are handled specially in
              // AllRdepsUnboundedVisitor#getVisitResult and will not be filtered later.
              return Pair.of(null, key);
            }
          });
    }

    @Override
    protected Iterable<Task> getVisitTasks(Collection<Pair<SkyKey, SkyKey>> pendingKeysToVisit) {
      // Group pending visits by package.
      ListMultimap<PackageIdentifier, Pair<SkyKey, SkyKey>> visitsByPackage =
          ArrayListMultimap.create();
      for (Pair<SkyKey, SkyKey> visit : pendingKeysToVisit) {
        Label label = SkyQueryEnvironment.SKYKEY_TO_LABEL.apply(visit.second);
        if (label != null) {
          visitsByPackage.put(label.getPackageIdentifier(), visit);
        }
      }

      ImmutableList.Builder<Task> builder = ImmutableList.builder();

      // A couple notes here:
      // (i)  ArrayListMultimap#values returns the values grouped by key, which is exactly what we
      //      want.
      // (ii) ArrayListMultimap#values returns a Collection view, so we make a copy to avoid
      //      accidentally retaining the entire ArrayListMultimap object.
      for (Iterable<Pair<SkyKey, SkyKey>> keysToVisitBatch :
          Iterables.partition(ImmutableList.copyOf(visitsByPackage.values()), VISIT_BATCH_SIZE)) {
        builder.add(new VisitTask(keysToVisitBatch));
      }

      return builder.build();
    }
  }

  /**
   * A {@link ThreadSafeCallback} whose {@link ThreadSafeCallback#process} method kicks off a BFS
   * visitation via a fresh {@link AbstractSkyKeyBFSVisitor} instance.
   */
  private static class SkyKeyBFSVisitorCallback implements ThreadSafeCallback<Target> {
    private final AbstractSkyKeyBFSVisitor.Factory visitorFactory;

    private SkyKeyBFSVisitorCallback(AbstractSkyKeyBFSVisitor.Factory visitorFactory) {
      this.visitorFactory = visitorFactory;
    }

    @Override
    public void process(Iterable<Target> partialResult)
        throws QueryException, InterruptedException {
      AbstractSkyKeyBFSVisitor<?> visitor = visitorFactory.create();
      visitor.visitAndWaitForCompletion(
          SkyQueryEnvironment.makeTransitiveTraversalKeysStrict(partialResult));
    }
  }

  /**
   * A helper class for performing a custom BFS visitation on the Skyframe graph, using {@link
   * QuiescingExecutor}.
   *
   * <p>The visitor uses an AbstractQueueVisitor backed by a ThreadPoolExecutor with a thread pool
   * NOT part of the global query evaluation pool to avoid starvation.
   */
  @ThreadSafe
  private abstract static class AbstractSkyKeyBFSVisitor<T> {
    protected final SkyQueryEnvironment env;
    private final ThreadSafeUniquifier<T> uniquifier;
    private final Callback<Target> callback;

    private final BFSVisitingTaskExecutor executor;

    /** A queue to store pending visits. */
    private final LinkedBlockingQueue<T> processingQueue = new LinkedBlockingQueue<>();

    /**
     * The max time interval between two scheduling passes in milliseconds. A scheduling pass is
     * defined as the scheduler thread determining whether to drain all pending visits from the
     * queue and submitting tasks to perform the visits.
     *
     * <p>The choice of 1ms is a result based of experiments. It is an attempted balance due to a
     * few facts about the scheduling interval:
     *
     * <p>1. A large interval adds systematic delay. In an extreme case, a BFS visit which is
     * supposed to take only 1ms now may take 5ms. For most BFS visits which take longer than a few
     * hundred milliseconds, it should not be noticeable.
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
     * number of keys being grouped by packages. On the other hand, we want to keep all threads in
     * the pool busy to achieve full capacity. A low number here will cause some of the worker
     * threads to go idle at times before the next scheduling cycle.
     *
     * <p>TODO(shazh): Revisit the choice of task target based on real-prod performance.
     */
    private static final long MIN_PENDING_TASKS = 3 * SkyQueryEnvironment.DEFAULT_THREAD_COUNT;

    /**
     * Fail fast on RuntimeExceptions, including {code RuntimeInterruptedException} and {@code
     * RuntimeQueryException}, which are resulted from InterruptedException and QueryException.
     */
    static final ErrorClassifier SKYKEY_BFS_VISITOR_ERROR_CLASSIFIER =
        new ErrorClassifier() {
          @Override
          protected ErrorClassification classifyException(Exception e) {
            return (e instanceof RuntimeException)
                ? ErrorClassification.CRITICAL_AND_LOG
                : ErrorClassification.NOT_CRITICAL;
          }
        };

    /** All BFS visitors share a single global fixed thread pool. */
    private static final ExecutorService FIXED_THREAD_POOL_EXECUTOR =
        new ThreadPoolExecutor(
            // Must be at least 2 worker threads in the pool (1 for the scheduler thread).
            /*corePoolSize=*/ Math.max(2, SkyQueryEnvironment.DEFAULT_THREAD_COUNT),
            /*maximumPoolSize=*/ Math.max(2, SkyQueryEnvironment.DEFAULT_THREAD_COUNT),
            /*keepAliveTime=*/ 1,
            /*units=*/ TimeUnit.SECONDS,
            /*workQueue=*/ new BlockingStack<Runnable>(),
            new ThreadFactoryBuilder().setNameFormat("skykey-bfs-visitor %d").build());

    private AbstractSkyKeyBFSVisitor(
        SkyQueryEnvironment env, ThreadSafeUniquifier<T> uniquifier, Callback<Target> callback) {
      this.env = env;
      this.uniquifier = uniquifier;
      this.callback = callback;
      this.executor =
          new BFSVisitingTaskExecutor(
              FIXED_THREAD_POOL_EXECUTOR, SKYKEY_BFS_VISITOR_ERROR_CLASSIFIER);
    }

    /** Factory for {@link AbstractSkyKeyBFSVisitor} instances. */
    private static interface Factory {
      AbstractSkyKeyBFSVisitor<?> create();
    }

    protected final class Visit {
      private final Iterable<SkyKey> keysToUseForResult;
      private final Iterable<T> keysToVisit;

      private Visit(Iterable<SkyKey> keysToUseForResult, Iterable<T> keysToVisit) {
        this.keysToUseForResult = keysToUseForResult;
        this.keysToVisit = keysToVisit;
      }
    }

    void visitAndWaitForCompletion(Iterable<SkyKey> keys)
        throws QueryException, InterruptedException {
      processingQueue.addAll(ImmutableList.copyOf(preprocessInitialVisit(keys)));
      executor.bfsVisitAndWaitForCompletion();
    }

    /**
     * Forwards the given {@code keysToUseForResult}'s contribution to the set of {@link Target}s in
     * the full visitation to the given {@link Callback}.
     */
    protected abstract void processResultantTargets(
        Iterable<SkyKey> keysToUseForResult, Callback<Target> callback)
        throws QueryException, InterruptedException;

    /** Gets the {@link Visit} representing the local visitation of the given {@code values}. */
    protected abstract Visit getVisitResult(Iterable<T> values) throws InterruptedException;

    /** Gets the first {@link Visit} representing the entry-level SkyKeys. */
    protected abstract Iterable<T> preprocessInitialVisit(Iterable<SkyKey> keys);

    protected Iterable<Task> getVisitTasks(Collection<T> pendingKeysToVisit) {
      ImmutableList.Builder<Task> builder = ImmutableList.builder();
      for (Iterable<T> keysToVisitBatch :
          Iterables.partition(pendingKeysToVisit, VISIT_BATCH_SIZE)) {
        builder.add(new VisitTask(keysToVisitBatch));
      }

      return builder.build();
    }

    abstract static class Task implements Runnable {

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

      private VisitTask(Iterable<T> keysToVisit) {
        this.keysToVisit = keysToVisit;
      }

      @Override
      void process() throws InterruptedException {
        ImmutableList<T> uniqueKeys = uniquifier.unique(keysToVisit);
        if (uniqueKeys.isEmpty()) {
          return;
        }

        Visit visit = getVisitResult(uniqueKeys);
        for (Iterable<SkyKey> keysToUseForResultBatch :
            Iterables.partition(
                visit.keysToUseForResult, SkyQueryEnvironment.BATCH_CALLBACK_SIZE)) {
          executor.execute(new GetAndProcessResultsTask(keysToUseForResultBatch));
        }

        processingQueue.addAll(ImmutableList.copyOf(visit.keysToVisit));
      }
    }

    private class GetAndProcessResultsTask extends Task {
      private final Iterable<SkyKey> keysToUseForResult;

      private GetAndProcessResultsTask(Iterable<SkyKey> keysToUseForResult) {
        this.keysToUseForResult = keysToUseForResult;
      }

      @Override
      protected void process() throws QueryException, InterruptedException {
        processResultantTargets(keysToUseForResult, callback);
      }
    }

    /**
     * A custom implementation of {@link QuiescingExecutor} which uses a centralized queue and
     * scheduler for parallel BFS visitations.
     */
    private class BFSVisitingTaskExecutor extends AbstractQueueVisitor {
      private BFSVisitingTaskExecutor(ExecutorService executor, ErrorClassifier errorClassifier) {
        super(
            /*concurrent=*/ true,
            /*executorService=*/ executor,
            // Leave the thread pool active for other current and future callers.
            /*shutdownOnCompletion=*/ false,
            /*failFastOnException=*/ true,
            /*errorClassifier=*/ errorClassifier);
      }

      private void bfsVisitAndWaitForCompletion() throws QueryException, InterruptedException {
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
            // occured in any of the worker thread, awaitTerminationAndPropagateErrorsIfAny
            // propagates the QueryException instead of InterruptedException.
            setInterrupted();
            awaitTerminationAndPropagateErrorsIfAny();
            throw e;
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

