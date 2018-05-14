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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllCallback;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.query2.engine.VariableContext;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Parallel implementations of various functionality in {@link SkyQueryEnvironment}.
 *
 * <p>Special attention is given to memory usage. Naive parallel implementations of query
 * functionality would lead to memory blowup. Instead of dealing with {@link Target}s, we try to
 * deal with {@link SkyKey}s as much as possible to reduce the number of {@link Package}s forcibly
 * in memory at any given time.
 */
// TODO(bazel-team): Be more deliberate about bounding memory usage here.
public class ParallelSkyQueryUtils {

  /** The maximum number of keys to visit at once. */
  @VisibleForTesting static final int VISIT_BATCH_SIZE = 10000;

  private ParallelSkyQueryUtils() {
  }

  static QueryTaskFuture<Void> getAllRdepsUnboundedParallel(
      SkyQueryEnvironment env,
      QueryExpression expression,
      VariableContext<Target> context,
      Callback<Target> callback,
      MultisetSemaphore<PackageIdentifier> packageSemaphore) {
    return env.eval(
        expression,
        context,
        ParallelVisitor.createParallelVisitorCallback(
            new RdepsUnboundedVisitor.Factory(
                env,
                /*universe=*/ Predicates.alwaysTrue(),
                callback,
                packageSemaphore)));
  }

  static QueryTaskFuture<Void> getAllRdepsBoundedParallel(
      SkyQueryEnvironment env,
      QueryExpression expression,
      int depth,
      VariableContext<Target> context,
      Callback<Target> callback,
      MultisetSemaphore<PackageIdentifier> packageSemaphore) {
    return env.eval(
        expression,
        context,
        ParallelVisitor.createParallelVisitorCallback(
            new RdepsBoundedVisitor.Factory(
                env,
                depth,
                /*universe=*/ Predicates.alwaysTrue(),
                callback,
                packageSemaphore)));
  }

  static QueryTaskFuture<Void> getRdepsInUniverseUnboundedParallel(
      SkyQueryEnvironment env,
      QueryExpression expression,
      Predicate<SkyKey> universe,
      VariableContext<Target> context,
      Callback<Target> callback,
      MultisetSemaphore<PackageIdentifier> packageSemaphore) {
    return env.eval(
        expression,
        context,
        ParallelVisitor.createParallelVisitorCallback(
            new RdepsUnboundedVisitor.Factory(env, universe, callback, packageSemaphore)));
  }

  static QueryTaskFuture<Predicate<SkyKey>> getDTCSkyKeyPredicateFuture(
      SkyQueryEnvironment env,
      QueryExpression expression,
      VariableContext<Target> context,
      int processResultsBatchSize,
      int concurrencyLevel) {
    QueryTaskFuture<ThreadSafeMutableSet<Target>> universeValueFuture =
        QueryUtil.evalAll(env, context, expression);

    Function<ThreadSafeMutableSet<Target>, QueryTaskFuture<Predicate<SkyKey>>>
        getTransitiveClosureAsyncFunction =
        universeValue -> {
          ThreadSafeAggregateAllSkyKeysCallback aggregateAllCallback =
              new ThreadSafeAggregateAllSkyKeysCallback(concurrencyLevel);
          return env.executeAsync(
              () -> {
                Callback<Target> visitorCallback =
                    ParallelVisitor.createParallelVisitorCallback(
                        new TransitiveTraversalValueDTCVisitor.Factory(
                            env,
                            env.createSkyKeyUniquifier(),
                            processResultsBatchSize,
                            aggregateAllCallback));
                visitorCallback.process(universeValue);
                return Predicates.in(aggregateAllCallback.getResult());
              });
        };

    return env.transformAsync(universeValueFuture, getTransitiveClosureAsyncFunction);
  }

  static QueryTaskFuture<Void> getRdepsInUniverseBoundedParallel(
      SkyQueryEnvironment env,
      QueryExpression expression,
      int depth,
      Predicate<SkyKey> universe,
      VariableContext<Target> context,
      Callback<Target> callback,
      MultisetSemaphore<PackageIdentifier> packageSemaphore) {
    return env.eval(
        expression,
        context,
        ParallelVisitor.createParallelVisitorCallback(
            new RdepsBoundedVisitor.Factory(
                env,
                depth,
                universe,
                callback,
                packageSemaphore)));
  }

  /** Specialized parallel variant of {@link SkyQueryEnvironment#getRBuildFiles}. */
  static void getRBuildFilesParallel(
      SkyQueryEnvironment env,
      Collection<PathFragment> fileIdentifiers,
      Callback<Target> callback) throws QueryException, InterruptedException {
    Uniquifier<SkyKey> keyUniquifier = env.createSkyKeyUniquifier();
    RBuildFilesVisitor visitor =
        new RBuildFilesVisitor(env, keyUniquifier, callback);
    visitor.visitAndWaitForCompletion(env.getFileStateKeysForFileFragments(fileIdentifiers));
  }

  /** A {@link ParallelVisitor} whose visitations occur on {@link SkyKey}s. */
  public abstract static class AbstractSkyKeyParallelVisitor<T> extends ParallelVisitor<SkyKey, T> {
    private final Uniquifier<SkyKey> uniquifier;

    protected AbstractSkyKeyParallelVisitor(
        Uniquifier<SkyKey> uniquifier,
        Callback<T> callback,
        int visitBatchSize,
        int processResultsBatchSize) {
      super(callback, visitBatchSize, processResultsBatchSize);
      this.uniquifier = uniquifier;
    }

    @Override
    protected ImmutableList<SkyKey> getUniqueValues(Iterable<SkyKey> values) {
      return uniquifier.unique(values);
    }
  }

  /** A helper class that computes 'rbuildfiles(<blah>)' via BFS. */
  private static class RBuildFilesVisitor extends AbstractSkyKeyParallelVisitor<Target> {
    // Each target in the full output of 'rbuildfiles' corresponds to BUILD file InputFile of a
    // unique package. So the processResultsBatchSize we choose to pass to the ParallelVisitor ctor
    // influences how many packages each leaf task doing processPartialResults will have to
    // deal with at once. A value of 100 was chosen experimentally.
    private static final int PROCESS_RESULTS_BATCH_SIZE = 100;
    private final SkyQueryEnvironment env;

    private RBuildFilesVisitor(
        SkyQueryEnvironment env,
        Uniquifier<SkyKey> uniquifier,
        Callback<Target> callback) {
      super(uniquifier, callback, VISIT_BATCH_SIZE, PROCESS_RESULTS_BATCH_SIZE);
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
        } else if (!rdep.functionName().equals(SkyFunctions.PACKAGE_LOOKUP)
            && !rdep.functionName().equals(SkyFunctions.GLOB)) {
          // Packages may depend on the existence of subpackages, but these edges aren't relevant to
          // rbuildfiles. They may also depend on files transitively through globs, but these cannot
          // be included in load statements and so we don't traverse through these either.
          keysToVisitNext.add(rdep);
        }
      }
      return new Visit(keysToUseForResult, keysToVisitNext);
    }

    @Override
    protected void processPartialResults(
        Iterable<SkyKey> keysToUseForResult, Callback<Target> callback)
        throws QueryException, InterruptedException {
      env.getBuildFileTargetsForPackageKeysAndProcessViaCallback(keysToUseForResult, callback);
    }

    @Override
    protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> keys) {
      return keys;
    }
  }

  private static class DepAndRdep {
    @Nullable
    private final SkyKey dep;
    private final SkyKey rdep;

    private DepAndRdep(@Nullable SkyKey dep, SkyKey rdep) {
      this.dep = dep;
      this.rdep = rdep;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof DepAndRdep)) {
        return false;
      }
      DepAndRdep other = (DepAndRdep) obj;
      return Objects.equals(dep, other.dep) && rdep.equals(other.rdep);
    }

    @Override
    public int hashCode() {
      // N.B. - We deliberately use a garbage-free hashCode implementation (rather than e.g.
      // Objects#hash). Depending on the structure of the graph being traversed, this method can
      // be very hot.
      return 31 * Objects.hashCode(dep) + rdep.hashCode();
    }
  }

  private abstract static class AbstractRdepsVisitor<T> extends ParallelVisitor<T, Target> {
    private static final int PROCESS_RESULTS_BATCH_SIZE = SkyQueryEnvironment.BATCH_CALLBACK_SIZE;

    protected final SkyQueryEnvironment env;
    protected final MultisetSemaphore<PackageIdentifier> packageSemaphore;

    protected AbstractRdepsVisitor(
        SkyQueryEnvironment env,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(callback, VISIT_BATCH_SIZE, PROCESS_RESULTS_BATCH_SIZE);
      this.env = env;
      this.packageSemaphore = packageSemaphore;
    }

    @Override
    protected void processPartialResults(
        Iterable<SkyKey> keysToUseForResult, Callback<Target> callback)
        throws QueryException, InterruptedException {
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
          env.makePackageKeyToTargetKeyMap(keysToUseForResult);
      Set<PackageIdentifier> pkgIdsNeededForResult =
          packageKeyToTargetKeyMap
              .keySet()
              .stream()
              .map(SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER)
              .collect(toImmutableSet());
      packageSemaphore.acquireAll(pkgIdsNeededForResult);
      try {
        callback.process(
            env.makeTargetsFromPackageKeyToTargetKeyMap(packageKeyToTargetKeyMap).values());
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForResult);
      }
    }

    protected abstract SkyKey getRdepOfVisit(T visit);

    @Override
    protected Iterable<Task> getVisitTasks(Collection<T> pendingVisits) {
      // Group pending visitation by the package of the rdep, since we'll be targetfying the
      // rdep during the visitation.
      ListMultimap<PackageIdentifier, T> visitsByPackage = ArrayListMultimap.create();
      for (T visit : pendingVisits) {
        Label label = SkyQueryEnvironment.SKYKEY_TO_LABEL.apply(getRdepOfVisit(visit));
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
      for (Iterable<T> visitBatch :
          Iterables.partition(ImmutableList.copyOf(visitsByPackage.values()), VISIT_BATCH_SIZE)) {
        builder.add(new VisitTask(visitBatch));
      }

      return builder.build();
    }
  }

  /**
   * A helper class that computes unbounded 'allrdeps(<expr>)' or
   * 'rdeps(<precomputed-universe>, <expr>)' via BFS.
   *
   * <p>The visitor uses {@link DepAndRdep} to keep track the nodes to visit and avoid dealing with
   * targetification of reverse deps until they are needed. The rdep node itself is needed to filter
   * out disallowed deps later. Compared against the approach using a single SkyKey, it consumes 16
   * more bytes in a 64-bit environment for each edge. However it defers the need to load all the
   * packages which have at least a target as a rdep of the current batch, thus greatly reduces the
   * risk of OOMs. The additional memory usage should not be a large concern here, as even with 10M
   * edges, the memory overhead is around 160M, and the memory can be reclaimed by regular GC.
   */
  private static class RdepsUnboundedVisitor extends AbstractRdepsVisitor<DepAndRdep> {
    /**
     * A {@link Uniquifier} for visitations. Solely used for {@link #getUniqueValues}, which
     * actually isn't that useful. See the method javadoc.
     */
    private final Uniquifier<DepAndRdep> depAndRdepUniquifier;
    /**
     * A {@link Uniquifier} for *valid* visitations of rdeps. {@code env}'s dependency filter might
     * mean that some rdep edges are invalid, meaning that any individual {@link DepAndRdep}
     * visitation may actually be invalid. Because the same rdep can be reached through more than
     * one reverse edge, it'd be incorrect to naively dedupe visitations solely based on the rdep.
     */
    private final Uniquifier<SkyKey> validRdepUniquifier;
    private final Predicate<SkyKey> universe;

    private RdepsUnboundedVisitor(
        SkyQueryEnvironment env,
        Uniquifier<DepAndRdep> depAndRdepUniquifier,
        Uniquifier<SkyKey> validRdepUniquifier,
        Predicate<SkyKey> universe,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(env, callback, packageSemaphore);
      this.depAndRdepUniquifier = depAndRdepUniquifier;
      this.validRdepUniquifier = validRdepUniquifier;
      this.universe = universe;
    }

    /**
     * A {@link Factory} for {@link RdepsUnboundedVisitor} instances, each of which will be used
     * to perform visitation of the reverse transitive closure of the {@link Target}s passed in a
     * single {@link Callback#process} call. Note that all the created instances share the same
     * {@link Uniquifier} so that we don't visit the same Skyframe node more than once.
     */
    private static class Factory implements ParallelVisitor.Factory {
      private final SkyQueryEnvironment env;
      private final Uniquifier<DepAndRdep> depAndRdepUniquifier;
      private final Uniquifier<SkyKey> validRdepUniquifier;
      private final Predicate<SkyKey> universe;
      private final Callback<Target> callback;
      private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

      private Factory(
        SkyQueryEnvironment env,
        Predicate<SkyKey> universe,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
        this.env = env;
        this.universe = universe;
        this.depAndRdepUniquifier = new UniquifierImpl<>(depAndRdep -> depAndRdep);
        this.validRdepUniquifier = env.createSkyKeyUniquifier();
        this.callback = callback;
        this.packageSemaphore = packageSemaphore;
      }

      @Override
      public ParallelVisitor<DepAndRdep, Target> create() {
        return new RdepsUnboundedVisitor(
            env, depAndRdepUniquifier, validRdepUniquifier, universe, callback, packageSemaphore);
      }
    }

    @Override
    protected Visit getVisitResult(Iterable<DepAndRdep> depAndRdeps) throws InterruptedException {
      Collection<SkyKey> validRdeps = new ArrayList<>();

      // Multimap of dep to all the reverse deps in this visitation. Used to filter out the
      // disallowed deps.
      Multimap<SkyKey, SkyKey> reverseDepMultimap = ArrayListMultimap.create();
      for (DepAndRdep depAndRdep : depAndRdeps) {
        // The "roots" of our visitation (see #preprocessInitialVisit) have a null 'dep' field.
        if (depAndRdep.dep == null) {
          validRdeps.add(depAndRdep.rdep);
        } else {
          reverseDepMultimap.put(depAndRdep.dep, depAndRdep.rdep);
        }
      }

      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
          env.makePackageKeyToTargetKeyMap(Iterables.concat(reverseDepMultimap.values()));
      Set<PackageIdentifier> pkgIdsNeededForTargetification =
          packageKeyToTargetKeyMap
              .keySet()
              .stream()
              .map(SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER)
              .collect(toImmutableSet());
      packageSemaphore.acquireAll(pkgIdsNeededForTargetification);

      try {
        // Filter out disallowed deps. We cannot defer the targetification any further as we do not
        // want to retrieve the rdeps of unwanted nodes (targets).
        if (!reverseDepMultimap.isEmpty()) {
          Collection<Target> filteredTargets =
              env.filterRawReverseDepsOfTransitiveTraversalKeys(
                  reverseDepMultimap.asMap(), packageKeyToTargetKeyMap);
          filteredTargets
              .stream()
              .map(SkyQueryEnvironment.TARGET_TO_SKY_KEY)
              .forEachOrdered(validRdeps::add);
        }
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForTargetification);
      }

      ImmutableList<SkyKey> uniqueValidRdeps = validRdeps.stream()
          .filter(validRdepUniquifier::unique)
          .collect(ImmutableList.toImmutableList());

      // Retrieve the reverse deps as SkyKeys and defer the targetification and filtering to next
      // recursive visitation.
      ImmutableList.Builder<DepAndRdep> depAndRdepsToVisitBuilder = ImmutableList.builder();
      env.graph.getReverseDeps(uniqueValidRdeps).entrySet()
          .forEach(reverseDepsEntry -> depAndRdepsToVisitBuilder.addAll(
              Iterables.transform(
                  Iterables.filter(
                      reverseDepsEntry.getValue(),
                      Predicates.and(SkyQueryEnvironment.IS_TTV, universe)),
                  rdep -> new DepAndRdep(reverseDepsEntry.getKey(), rdep))));

      return new Visit(
          /*keysToUseForResult=*/ uniqueValidRdeps,
          /*keysToVisit=*/ depAndRdepsToVisitBuilder.build());
    }

    @Override
    protected Iterable<DepAndRdep> preprocessInitialVisit(Iterable<SkyKey> keys) {
      return Iterables.transform(
          Iterables.filter(keys, k -> universe.apply(k)),
          key -> new DepAndRdep(null, key));
    }

    @Override
    protected SkyKey getRdepOfVisit(DepAndRdep visit) {
      return visit.rdep;
    }

    @Override
    protected ImmutableList<DepAndRdep> getUniqueValues(Iterable<DepAndRdep> depAndRdeps) {
      // See the javadoc for 'validRdepUniquifier'.
      //
      // N.B. - Except for the visitation roots, 'depAndRdepUniquifier' is actually completely
      // unneeded in practice for ensuring literal unique {@link DepAndRdep} visitations. Valid rdep
      // visitations are deduped in 'getVisitResult' using 'validRdepUniquifier', so there's
      // actually no way the same DepAndRdep visitation can ever be returned from 'getVisitResult'.
      // Still, we include an implementation of 'getUniqueValues' that is correct in isolation so as
      // to not be depending on implementation details of 'ParallelVisitor'.
      //
      // Even so, there's value in not visiting a rdep if it's already been visited *validly*
      // before. We use the intentionally racy {@link Uniquifier#uniquePure} to attempt to do this.
      return depAndRdepUniquifier.unique(
          Iterables.filter(
              depAndRdeps,
              depAndRdep -> validRdepUniquifier.uniquePure(depAndRdep.rdep)));
    }
  }

  private static class DepAndRdepAtDepth {
    private final DepAndRdep depAndRdep;
    private final int rdepDepth;

    private DepAndRdepAtDepth(DepAndRdep depAndRdep, int rdepDepth) {
      this.depAndRdep = depAndRdep;
      this.rdepDepth = rdepDepth;
    }
  }

  /**
   * A helper class that computes bounded 'allrdeps(<expr>, <depth>)' or
   * 'rdeps(<precomputed-universe>, <expr>, <depth>)' via BFS.
   *
   * <p>This is very similar to {@link RdepsUnboundedVisitor}. A lot of the same concerns apply here
   * but there are additional subtle concerns about the correctness of the bounded traversal: just
   * like for the sequential implementation of bounded allrdeps, we use {@link MinDepthUniquifier}.
   */
  private static class RdepsBoundedVisitor extends AbstractRdepsVisitor<DepAndRdepAtDepth> {
    private final int depth;
    private final Uniquifier<DepAndRdepAtDepth> depAndRdepAtDepthUniquifier;
    private final MinDepthUniquifier<SkyKey> validRdepMinDepthUniquifier;
    private final Predicate<SkyKey> universe;

    private RdepsBoundedVisitor(
        SkyQueryEnvironment env,
        int depth,
        Uniquifier<DepAndRdepAtDepth> depAndRdepAtDepthUniquifier,
        MinDepthUniquifier<SkyKey> validRdepMinDepthUniquifier,
        Predicate<SkyKey> universe,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(env, callback, packageSemaphore);
      this.depth = depth;
      this.depAndRdepAtDepthUniquifier = depAndRdepAtDepthUniquifier;
      this.validRdepMinDepthUniquifier = validRdepMinDepthUniquifier;
      this.universe = universe;
    }

    private static class Factory implements ParallelVisitor.Factory {
      private final SkyQueryEnvironment env;
      private final int depth;
      private final Uniquifier<DepAndRdepAtDepth> depAndRdepAtDepthUniquifier;
      private final MinDepthUniquifier<SkyKey> validRdepMinDepthUniquifier;
      private final Predicate<SkyKey> universe;
      private final Callback<Target> callback;
      private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

      private Factory(
          SkyQueryEnvironment env,
          int depth,
          Predicate<SkyKey> universe,
          Callback<Target> callback,
          MultisetSemaphore<PackageIdentifier> packageSemaphore) {
        this.env = env;
        this.depth = depth;
        this.universe = universe;
        this.depAndRdepAtDepthUniquifier =
            new UniquifierImpl<>(depAndRdepAtDepth -> depAndRdepAtDepth);
        this.validRdepMinDepthUniquifier = env.createMinDepthSkyKeyUniquifier();
        this.callback = callback;
        this.packageSemaphore = packageSemaphore;
      }

      @Override
      public ParallelVisitor<DepAndRdepAtDepth, Target> create() {
        return new RdepsBoundedVisitor(
            env,
            depth,
            depAndRdepAtDepthUniquifier,
            validRdepMinDepthUniquifier,
            universe,
            callback,
            packageSemaphore);
      }
    }

    @Override
    protected Visit getVisitResult(Iterable<DepAndRdepAtDepth> depAndRdepAtDepths)
        throws InterruptedException {
      Map<SkyKey, Integer> shallowestRdepDepthMap = new HashMap<>();
      depAndRdepAtDepths.forEach(
          depAndRdepAtDepth -> shallowestRdepDepthMap.merge(
              depAndRdepAtDepth.depAndRdep.rdep, depAndRdepAtDepth.rdepDepth, Integer::min));

      Collection<SkyKey> validRdeps = new ArrayList<>();

      // Multimap of dep to all the reverse deps in this visitation. Used to filter out the
      // disallowed deps.
      Multimap<SkyKey, SkyKey> reverseDepMultimap = ArrayListMultimap.create();
      for (DepAndRdepAtDepth depAndRdepAtDepth : depAndRdepAtDepths) {
        // The "roots" of our visitation (see #preprocessInitialVisit) have a null 'dep' field.
        if (depAndRdepAtDepth.depAndRdep.dep == null) {
          validRdeps.add(depAndRdepAtDepth.depAndRdep.rdep);
        } else {
          reverseDepMultimap.put(
              depAndRdepAtDepth.depAndRdep.dep, depAndRdepAtDepth.depAndRdep.rdep);
        }
      }

      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
          env.makePackageKeyToTargetKeyMap(Iterables.concat(reverseDepMultimap.values()));
      Set<PackageIdentifier> pkgIdsNeededForTargetification =
          packageKeyToTargetKeyMap
              .keySet()
              .stream()
              .map(SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER)
              .collect(toImmutableSet());
      packageSemaphore.acquireAll(pkgIdsNeededForTargetification);

      try {
        // Filter out disallowed deps. We cannot defer the targetification any further as we do not
        // want to retrieve the rdeps of unwanted nodes (targets).
        if (!reverseDepMultimap.isEmpty()) {
          Collection<Target> filteredTargets =
              env.filterRawReverseDepsOfTransitiveTraversalKeys(
                  reverseDepMultimap.asMap(), packageKeyToTargetKeyMap);
          filteredTargets
              .stream()
              .map(SkyQueryEnvironment.TARGET_TO_SKY_KEY)
              .forEachOrdered(validRdeps::add);
        }
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForTargetification);
      }

      ImmutableList<SkyKey> uniqueValidRdeps = validRdeps.stream()
          .filter(validRdep -> validRdepMinDepthUniquifier.uniqueAtDepthLessThanOrEqualTo(
              validRdep, shallowestRdepDepthMap.get(validRdep)))
          .collect(ImmutableList.toImmutableList());

      // Don't bother getting the rdeps of the rdeps that are already at the depth bound.
      Iterable<SkyKey> uniqueValidRdepsBelowDepthBound = Iterables.filter(
          uniqueValidRdeps,
          uniqueValidRdep -> shallowestRdepDepthMap.get(uniqueValidRdep) < depth);

      // Retrieve the reverse deps as SkyKeys and defer the targetification and filtering to next
      // recursive visitation.
      Map<SkyKey, Iterable<SkyKey>> unfilteredRdepsOfRdeps =
          env.graph.getReverseDeps(uniqueValidRdepsBelowDepthBound);

      ImmutableList.Builder<DepAndRdepAtDepth> depAndRdepAtDepthsToVisitBuilder =
          ImmutableList.builder();
      unfilteredRdepsOfRdeps.entrySet().forEach(entry -> {
        SkyKey rdep = entry.getKey();
        int depthOfRdepOfRdep = shallowestRdepDepthMap.get(rdep) + 1;
        Streams.stream(entry.getValue())
            .filter(Predicates.and(SkyQueryEnvironment.IS_TTV, universe))
            .forEachOrdered(rdepOfRdep -> {
          depAndRdepAtDepthsToVisitBuilder.add(
              new DepAndRdepAtDepth(new DepAndRdep(rdep, rdepOfRdep), depthOfRdepOfRdep));
        });
      });

      return new Visit(
          /*keysToUseForResult=*/ uniqueValidRdeps,
          /*keysToVisit=*/ depAndRdepAtDepthsToVisitBuilder.build());
    }

    @Override
    protected Iterable<DepAndRdepAtDepth> preprocessInitialVisit(Iterable<SkyKey> keys) {
      return Iterables.transform(
          Iterables.filter(keys, k -> universe.apply(k)),
          key -> new DepAndRdepAtDepth(new DepAndRdep(null, key), 0));
    }

    @Override
    protected SkyKey getRdepOfVisit(DepAndRdepAtDepth visit) {
      return visit.depAndRdep.rdep;
    }

    @Override
    protected ImmutableList<DepAndRdepAtDepth> getUniqueValues(
        Iterable<DepAndRdepAtDepth> depAndRdepAtDepths) {
      // See the comment in RdepsUnboundedVisitor#getUniqueValues.
      return depAndRdepAtDepthUniquifier.unique(
          Iterables.filter(
              depAndRdepAtDepths,
              depAndRdepAtDepth -> validRdepMinDepthUniquifier.uniqueAtDepthLessThanOrEqualToPure(
                  depAndRdepAtDepth.depAndRdep.rdep, depAndRdepAtDepth.rdepDepth)));
    }
  }

  /**
   * Helper class that computes the TTV-only DTC of some given TTV keys, via BFS following all
   * TTV->TTV dep edges.
   */
  private static class TransitiveTraversalValueDTCVisitor extends ParallelVisitor<SkyKey, SkyKey> {
    private final SkyQueryEnvironment env;
    private final Uniquifier<SkyKey> uniquifier;

    private TransitiveTraversalValueDTCVisitor(
        SkyQueryEnvironment env,
        Uniquifier<SkyKey> uniquifier,
        int processResultsBatchSize,
        AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> aggregateAllCallback) {
      super(aggregateAllCallback, VISIT_BATCH_SIZE, processResultsBatchSize);
      this.env = env;
      this.uniquifier = uniquifier;
    }

    private static class Factory implements ParallelVisitor.Factory {
      private final SkyQueryEnvironment env;
      private final Uniquifier<SkyKey> uniquifier;
      private final AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> aggregateAllCallback;
      private final int processResultsBatchSize;

      private Factory(
          SkyQueryEnvironment env,
          Uniquifier<SkyKey> uniquifier,
          int processResultsBatchSize,
          AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> aggregateAllCallback) {
        this.env = env;
        this.uniquifier = uniquifier;
        this.processResultsBatchSize = processResultsBatchSize;
        this.aggregateAllCallback = aggregateAllCallback;
      }

      @Override
      public ParallelVisitor<SkyKey, SkyKey> create() {
        return new TransitiveTraversalValueDTCVisitor(
            env, uniquifier, processResultsBatchSize, aggregateAllCallback);
      }
    }

    @Override
    protected void processPartialResults(
        Iterable<SkyKey> keysToUseForResult, Callback<SkyKey> callback)
        throws QueryException, InterruptedException {
      callback.process(keysToUseForResult);
    }

    @Override
    protected Visit getVisitResult(Iterable<SkyKey> ttvKeys) throws InterruptedException {
      Multimap<SkyKey, SkyKey> deps = env.getDirectDepsOfSkyKeys(ttvKeys);
      return new Visit(
          /*keysToUseForResult=*/ deps.keySet(),
          /*keysToVisit=*/ deps.values().stream()
          .filter(SkyQueryEnvironment.IS_TTV)
          .collect(ImmutableList.toImmutableList()));
    }

    @Override
    protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> keys) {
      // ParallelVisitorCallback passes in TTV keys.
      Preconditions.checkState(Iterables.all(keys, SkyQueryEnvironment.IS_TTV), keys);
      return keys;
    }

    @Override
    protected ImmutableList<SkyKey> getUniqueValues(Iterable<SkyKey> values) {
      return uniquifier.unique(values);
    }
  }

  /** Thread-safe {@link AggregateAllCallback} backed by a concurrent {@link Set}. */
  @ThreadSafe
  private static class ThreadSafeAggregateAllSkyKeysCallback
      implements AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> {

    private final Set<SkyKey> results;

    private ThreadSafeAggregateAllSkyKeysCallback(int concurrencyLevel) {
      this.results =
          Collections.newSetFromMap(
              new ConcurrentHashMap<>(
                  /*initialCapacity=*/ concurrencyLevel, /*loadFactor=*/ 0.75f));
    }

    @Override
    public void process(Iterable<SkyKey> partialResult)
        throws QueryException, InterruptedException {
      Iterables.addAll(results, partialResult);
    }

    @Override
    public ImmutableSet<SkyKey> getResult() {
      return ImmutableSet.copyOf(results);
    }
  }
}

