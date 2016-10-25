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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MoreFutures;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

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
      ForkJoinPool forkJoinPool)
          throws QueryException, InterruptedException {
    env.eval(
        expression,
        context,
        new SkyKeyBFSVisitorCallback(
            new AllRdepsUnboundedVisitor.Factory(env, callback, forkJoinPool)));
  }

  /** Specialized parallel variant of {@link SkyQueryEnvironment#getRBuildFiles}. */
  static void getRBuildFilesParallel(
      SkyQueryEnvironment env,
      Collection<PathFragment> fileIdentifiers,
      ThreadSafeCallback<Target> callback,
      ForkJoinPool forkJoinPool)
          throws QueryException, InterruptedException {
    ThreadSafeUniquifier<SkyKey> keyUniquifier = env.createSkyKeyUniquifier();
    RBuildFilesVisitor visitor = new RBuildFilesVisitor(env, forkJoinPool, keyUniquifier, callback);
    visitor.visitAndWaitForCompletion(env.getSkyKeysForFileFragments(fileIdentifiers));
  }

  /** A helper class that computes 'rbuildfiles(<blah>)' via BFS. */
  private static class RBuildFilesVisitor extends AbstractSkyKeyBFSVisitor<SkyKey> {
    private final SkyQueryEnvironment env;

    private RBuildFilesVisitor(
        SkyQueryEnvironment env,
        ForkJoinPool forkJoinPool,
        ThreadSafeUniquifier<SkyKey> uniquifier,
        Callback<Target> callback) {
      super(forkJoinPool, uniquifier, callback);
      this.env = env;
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
    protected Iterable<Target> getTargetsToAddToResult(Iterable<SkyKey> keysToUseForResult)
        throws InterruptedException {
      return SkyQueryEnvironment.getBuildFilesForPackageValues(
          env.graph.getSuccessfulValues(keysToUseForResult).values());
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
    private final SkyQueryEnvironment env;

    private AllRdepsUnboundedVisitor(
        SkyQueryEnvironment env,
        ForkJoinPool forkJoinPool,
        ThreadSafeUniquifier<Pair<SkyKey, SkyKey>> uniquifier,
        ThreadSafeCallback<Target> callback) {
      super(forkJoinPool, uniquifier, callback);
      this.env = env;
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
      private final ForkJoinPool forkJoinPool;
      private final ThreadSafeUniquifier<Pair<SkyKey, SkyKey>> uniquifier;
      private final ThreadSafeCallback<Target> callback;

      private Factory(
        SkyQueryEnvironment env,
        ThreadSafeCallback<Target> callback,
        ForkJoinPool forkJoinPool) {
        this.env = env;
        this.forkJoinPool = forkJoinPool;
        this.uniquifier = env.createReverseDepSkyKeyUniquifier();
        this.callback = callback;
      }

      @Override
      public AbstractSkyKeyBFSVisitor<Pair<SkyKey, SkyKey>> create() {
        return new AllRdepsUnboundedVisitor(env, forkJoinPool, uniquifier, callback);
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

      // Filter out disallowed deps. We cannot defer the targetification any further as we do not
      // want to retrieve the rdeps of unwanted nodes (targets).
      if (!reverseDepsMap.isEmpty()) {
        Collection<Target> filteredTargets =
            env.filterRawReverseDepsOfTransitiveTraversalKeys(reverseDepsMap);
        filteredKeys.addAll(
            Collections2.transform(filteredTargets, SkyQueryEnvironment.TARGET_TO_SKY_KEY));
      }

      // Retrieve the reverse deps as SkyKeys and defer the targetification and filtering to next
      // recursive visitation.
      Map<SkyKey, Iterable<SkyKey>> unfilteredReverseDeps = env.graph.getReverseDeps(filteredKeys);

      // Build a collection of Pairs and group by package id so we can partition them efficiently
      // later.
      ArrayListMultimap<PackageIdentifier, Pair<SkyKey, SkyKey>> rdepsByPackage =
          ArrayListMultimap.create();
      for (Map.Entry<SkyKey, Iterable<SkyKey>> rdeps : unfilteredReverseDeps.entrySet()) {
        for (SkyKey rdep : rdeps.getValue()) {
          Label label = SkyQueryEnvironment.SKYKEY_TO_LABEL.apply(rdep);
          if (label != null) {
            rdepsByPackage.put(label.getPackageIdentifier(), Pair.of(rdeps.getKey(), rdep));
          }
        }
      }

      // A couple notes here:
      // (i)  ArrayListMultimap#values returns the values grouped by key, which is exactly what we
      //      want.
      // (ii) ArrayListMultimap#values returns a Collection view, so we make a copy to avoid
      //      accidentally retaining the entire ArrayListMultimap object.
      Iterable<Pair<SkyKey, SkyKey>> keysToVisit = ImmutableList.copyOf(rdepsByPackage.values());

      // TODO(shazh): Use a global pool to store keys to be returned and keys to be processed, and
      // assign them to VisitTasks. It allows us to better optimize package retrieval.
      return new Visit(/*keysToUseForResult=*/ filteredKeys, /*keysToVisit=*/ keysToVisit);
    }

    @Override
    protected Iterable<Target> getTargetsToAddToResult(Iterable<SkyKey> keysToUseForResult)
        throws InterruptedException {
      return env.makeTargetsFromSkyKeys(keysToUseForResult).values();
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
   * ForkJoinQuiescingExecutor}.
   *
   * <p>The choice of {@link ForkJoinPool} over, say, AbstractQueueVisitor backed by a
   * ThreadPoolExecutor, is very deliberate. {@link SkyKeyBFSVisitorCallback#process} kicks off a
   * visitation and blocks on completion of it. But this visitation may never complete if there are
   * a bounded number of threads in the global thread pool used for query evaluation!
   */
  @ThreadSafe
  private abstract static class AbstractSkyKeyBFSVisitor<T> {
    private final ForkJoinPool forkJoinPool;
    private final ThreadSafeUniquifier<T> uniquifier;
    private final Callback<Target> callback;
    /** The maximum number of keys to visit at once. */
    private static final int VISIT_BATCH_SIZE = 10000;

    private AbstractSkyKeyBFSVisitor(
        ForkJoinPool forkJoinPool, ThreadSafeUniquifier<T> uniquifier, Callback<Target> callback) {
      this.forkJoinPool = forkJoinPool;
      this.uniquifier = uniquifier;
      this.callback = callback;
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
      Iterable<ForkJoinTask<?>> tasks =
          getTasks(
              new Visit(
                  /*keysToUseForResult=*/ ImmutableList.<SkyKey>of(),
                  /*keysToVisit=*/ preprocessInitialVisit(keys)));
      for (ForkJoinTask<?> task : tasks) {
        forkJoinPool.execute(task);
      }
      try {
        MoreFutures.waitForAllInterruptiblyFailFast(tasks);
      } catch (ExecutionException ee) {
        Throwable cause = ee.getCause();
        if (cause instanceof RuntimeQueryException) {
          throw (QueryException) cause.getCause();
        } else if (cause instanceof RuntimeInterruptedException) {
          throw (InterruptedException) cause.getCause();
        } else {
          throw new IllegalStateException(cause);
        }
      }
    }

    private abstract static class AbstractInternalRecursiveAction extends RecursiveAction {
      protected abstract void computeImpl() throws QueryException, InterruptedException;

      @Override
      public final void compute() {
        try {
          computeImpl();
        } catch (QueryException queryException) {
          throw new RuntimeQueryException(queryException);
        } catch (InterruptedException interruptedException) {
          throw new RuntimeInterruptedException(interruptedException);
        }
      }
    }

    private class VisitTask extends AbstractInternalRecursiveAction {
      private final Iterable<T> keysToVisit;

      private VisitTask(Iterable<T> keysToVisit) {
        this.keysToVisit = keysToVisit;
      }

      @Override
      protected void computeImpl() throws InterruptedException {
        ImmutableList<T> uniqueKeys = uniquifier.unique(keysToVisit);
        if (uniqueKeys.isEmpty()) {
          return;
        }
        Iterable<ForkJoinTask<?>> tasks = getTasks(getVisitResult(uniqueKeys));
        for (ForkJoinTask<?> task : tasks) {
          task.fork();
        }
        for (ForkJoinTask<?> task : tasks) {
          task.join();
        }
      }
    }

    private class GetAndProcessResultsTask extends AbstractInternalRecursiveAction {
      private final Iterable<SkyKey> keysToUseForResult;

      private GetAndProcessResultsTask(Iterable<SkyKey> keysToUseForResult) {
        this.keysToUseForResult = keysToUseForResult;
      }

      @Override
      protected void computeImpl() throws QueryException, InterruptedException {
        callback.process(getTargetsToAddToResult(keysToUseForResult));
      }
    }

    private Iterable<ForkJoinTask<?>> getTasks(Visit visit) {
      // Split the given visit request into ForkJoinTasks for visiting keys and ForkJoinTasks for
      // getting and outputting results, each of which obeys the separate batch limits.
      // TODO(bazel-team): Attempt to group work on targets within the same package.
      ImmutableList.Builder<ForkJoinTask<?>> tasksBuilder = ImmutableList.builder();
      for (Iterable<T> keysToVisitBatch :
          Iterables.partition(visit.keysToVisit, VISIT_BATCH_SIZE)) {
        tasksBuilder.add(new VisitTask(keysToVisitBatch));
      }
      for (Iterable<SkyKey> keysToUseForResultBatch : Iterables.partition(
          visit.keysToUseForResult, SkyQueryEnvironment.BATCH_CALLBACK_SIZE)) {
        tasksBuilder.add(new GetAndProcessResultsTask(keysToUseForResultBatch));
      }
      return tasksBuilder.build();
    }

    /**
     * Gets the given {@code keysToUseForResult}'s contribution to the set of {@link Target}s in the
     * full visitation.
     */
    protected abstract Iterable<Target> getTargetsToAddToResult(
        Iterable<SkyKey> keysToUseForResult) throws InterruptedException;

    /** Gets the {@link Visit} representing the local visitation of the given {@code values}. */
    protected abstract Visit getVisitResult(Iterable<T> values) throws InterruptedException;

    /** Gets the first {@link Visit} representing the entry-level SkyKeys. */
    protected abstract Iterable<T> preprocessInitialVisit(Iterable<SkyKey> keys);
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

