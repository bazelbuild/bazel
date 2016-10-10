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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.concurrent.ForkJoinQuiescingExecutor;
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
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
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
  private static class RBuildFilesVisitor extends AbstractSkyKeyBFSVisitor {
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
  }

  /** A helper class that computes 'allrdeps(<blah>)' via BFS. */
  private static class AllRdepsUnboundedVisitor extends AbstractSkyKeyBFSVisitor {
    private final SkyQueryEnvironment env;

    private AllRdepsUnboundedVisitor(
        SkyQueryEnvironment env,
        ForkJoinPool forkJoinPool,
        ThreadSafeUniquifier<SkyKey> uniquifier,
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
      private final ThreadSafeUniquifier<SkyKey> uniquifier;
      private final ThreadSafeCallback<Target> callback;

      private Factory(
        SkyQueryEnvironment env,
        ThreadSafeCallback<Target> callback,
        ForkJoinPool forkJoinPool) {
        this.env = env;
        this.forkJoinPool = forkJoinPool;
        this.uniquifier = env.createSkyKeyUniquifier();
        this.callback = callback;
      }

      @Override
      public AbstractSkyKeyBFSVisitor create() {
        return new AllRdepsUnboundedVisitor(env, forkJoinPool, uniquifier, callback);
      }
    }

    @Override
    protected Visit getVisitResult(Iterable<SkyKey> keys) throws InterruptedException {
      // TODO(bazel-team): Defer some of this work to the next recursive visitation. Instead, have
      // this visitation merely get the Skyframe-land rdeps.

      // Note that this does more than merely get the Skyframe-land rdeps:
      // (i)  It only returns rdeps that have corresponding Targets.
      // (ii) It only returns rdeps whose corresponding Targets have a valid dependency edge to
      //      their direct dep.
      Iterable<SkyKey> keysToVisit = SkyQueryEnvironment.makeTransitiveTraversalKeysStrict(
          env.getReverseDepsOfTransitiveTraversalKeys(keys));
      return new Visit(
          /*keysToUseForResult=*/ keys,
          /*keysToVisit=*/ keysToVisit);
    }

    @Override
    protected Iterable<Target> getTargetsToAddToResult(Iterable<SkyKey> keysToUseForResult)
        throws InterruptedException {
      return env.makeTargetsFromSkyKeys(keysToUseForResult).values();
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
      AbstractSkyKeyBFSVisitor visitor = visitorFactory.create();
      visitor.visitAndWaitForCompletion(
          SkyQueryEnvironment.makeTransitiveTraversalKeysStrict(partialResult));
    }
  }

  /**
   * A helper class for performing a custom BFS visitation on the Skyframe graph, using
   * {@link ForkJoinQuiescingExecutor}.
   *
   * <p>The choice of {@link ForkJoinPool} over, say, AbstractQueueVisitor backed by a
   * ThreadPoolExecutor, is very deliberate. {@link SkyKeyBFSVisitorCallback#process} kicks off
   * a visitation and blocks on completion of it. But this visitation may never complete if there
   * are a bounded number of threads in the global thread pool used for query evaluation!
   */
  @ThreadSafe
  private abstract static class AbstractSkyKeyBFSVisitor {
    private final ForkJoinPool forkJoinPool;
    private final ThreadSafeUniquifier<SkyKey> uniquifier;
    private final Callback<Target> callback;
    /** The maximum number of keys to visit at once. */
    private static final int VISIT_BATCH_SIZE = 10000;

    private AbstractSkyKeyBFSVisitor(
        ForkJoinPool forkJoinPool,
        ThreadSafeUniquifier<SkyKey> uniquifier,
        Callback<Target> callback) {
      this.forkJoinPool = forkJoinPool;
      this.uniquifier = uniquifier;
      this.callback = callback;
    }

    /** Factory for {@link AbstractSkyKeyBFSVisitor} instances. */
    private static interface Factory {
      AbstractSkyKeyBFSVisitor create();
    }

    protected static final class Visit {
      private final Iterable<SkyKey> keysToUseForResult;
      private final Iterable<SkyKey> keysToVisit;

      private Visit(Iterable<SkyKey> keysToUseForResult, Iterable<SkyKey> keysToVisit) {
        this.keysToUseForResult = keysToUseForResult;
        this.keysToVisit = keysToVisit;
      }
    }

    void visitAndWaitForCompletion(Iterable<SkyKey> keys)
        throws QueryException, InterruptedException {
      Iterable<ForkJoinTask<?>> tasks = getTasks(new Visit(
          /*keysToUseForResult=*/ ImmutableList.<SkyKey>of(),
          /*keysToVisit=*/ keys));
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
      private final Iterable<SkyKey> keysToVisit;

      private VisitTask(Iterable<SkyKey> keysToVisit) {
        this.keysToVisit = keysToVisit;
      }

      @Override
      protected void computeImpl() throws InterruptedException {
        ImmutableList<SkyKey> uniqueKeys = uniquifier.unique(keysToVisit);
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
      for (Iterable<SkyKey> keysToVisitBatch
          : Iterables.partition(visit.keysToVisit, VISIT_BATCH_SIZE)) {
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
    protected abstract Visit getVisitResult(Iterable<SkyKey> values) throws InterruptedException;
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

