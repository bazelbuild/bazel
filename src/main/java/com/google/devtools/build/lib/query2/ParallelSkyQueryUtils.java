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
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.query2.engine.VariableContext;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
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

  /**
   * Specialized parallel variant of {@link SkyQueryEnvironment#getAllRdeps} that is appropriate
   * when there is no depth-bound.
   */
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
            new AllRdepsUnboundedVisitor.Factory(env, callback, packageSemaphore)));
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

  /**
   * A helper class that computes 'allrdeps(<blah>)' via BFS.
   *
   * <p>The visitor uses {@link DepAndRdep} to keep track the nodes to visit and avoid dealing with
   * targetification of reverse deps until they are needed. The rdep node itself is needed to filter
   * out disallowed deps later. Compared against the approach using a single SkyKey, it consumes 16
   * more bytes in a 64-bit environment for each edge. However it defers the need to load all the
   * packages which have at least a target as a rdep of the current batch, thus greatly reduces the
   * risk of OOMs. The additional memory usage should not be a large concern here, as even with 10M
   * edges, the memory overhead is around 160M, and the memory can be reclaimed by regular GC.
   */
  private static class AllRdepsUnboundedVisitor extends ParallelVisitor<DepAndRdep, Target> {
    private static final int PROCESS_RESULTS_BATCH_SIZE = SkyQueryEnvironment.BATCH_CALLBACK_SIZE;
    private final SkyQueryEnvironment env;
    private final MultisetSemaphore<PackageIdentifier> packageSemaphore;
    /**
     * A {@link Uniquifier} for visitations. Solely used for {@link #getUniqueValues}, which
     * actually isn't that useful. See the method javadoc.
     */
    private final Uniquifier<DepAndRdep> depAndRdepUniquifier;
    /**
     * A {@link Uniquifier} for *valid* visitations of rdeps. {@code env}'s dependency filter might
     * mean that some rdep edges are invalid, meaning that any individual {@link DepAndRdep}
     * visitation may actually be invalid. Because the same rdep can be reached through more than
     * one reverse edge, It'd be incorrectly to naively dedupe visitations solely based on the rdep.
     */
    private final Uniquifier<SkyKey> validRdepUniquifier;

    private AllRdepsUnboundedVisitor(
        SkyQueryEnvironment env,
        Uniquifier<DepAndRdep> depAndRdepUniquifier,
        Uniquifier<SkyKey> validRdepUniquifier,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(callback, VISIT_BATCH_SIZE, PROCESS_RESULTS_BATCH_SIZE);
      this.env = env;
      this.depAndRdepUniquifier = depAndRdepUniquifier;
      this.validRdepUniquifier = validRdepUniquifier;
      this.packageSemaphore = packageSemaphore;
    }

    /**
     * A {@link Factory} for {@link AllRdepsUnboundedVisitor} instances, each of which will be used
     * to perform visitation of the reverse transitive closure of the {@link Target}s passed in a
     * single {@link Callback#process} call. Note that all the created instances share the same
     * {@link Uniquifier} so that we don't visit the same Skyframe node more than once.
     */
    private static class Factory implements ParallelVisitor.Factory {
      private final SkyQueryEnvironment env;
      private final Uniquifier<DepAndRdep> depAndRdepUniquifier;
      private final Uniquifier<SkyKey> validRdepUniquifier;
      private final Callback<Target> callback;
      private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

      private Factory(
        SkyQueryEnvironment env,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
        this.env = env;
        this.depAndRdepUniquifier = new UniquifierImpl<>(depAndRdep -> depAndRdep);
        this.validRdepUniquifier = env.createSkyKeyUniquifier();
        this.callback = callback;
        this.packageSemaphore = packageSemaphore;
      }

      @Override
      public ParallelVisitor<DepAndRdep, Target> create() {
        return new AllRdepsUnboundedVisitor(
            env, depAndRdepUniquifier, validRdepUniquifier, callback, packageSemaphore);
      }
    }

    @Override
    protected Visit getVisitResult(Iterable<DepAndRdep> depAndRdeps) throws InterruptedException {
      Collection<SkyKey> filteredUniqueKeys = new ArrayList<>();

      // Build a raw reverse dep map from pairs of SkyKeys to filter out the disallowed deps.
      Map<SkyKey, Collection<SkyKey>> reverseDepsMap = Maps.newHashMap();
      for (DepAndRdep depAndRdep : depAndRdeps) {
        // The "roots" of our visitation (see #preprocessInitialVisit) have a null 'dep' field.
        if (depAndRdep.dep == null && validRdepUniquifier.unique(depAndRdep.rdep)) {
          filteredUniqueKeys.add(depAndRdep.rdep);
          continue;
        }

        reverseDepsMap.computeIfAbsent(depAndRdep.dep, k -> new LinkedList<SkyKey>());

        reverseDepsMap.get(depAndRdep.dep).add(depAndRdep.rdep);
      }

      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
          env.makePackageKeyToTargetKeyMap(Iterables.concat(reverseDepsMap.values()));
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
        if (!reverseDepsMap.isEmpty()) {
          Collection<Target> filteredTargets =
              env.filterRawReverseDepsOfTransitiveTraversalKeys(
                  reverseDepsMap, packageKeyToTargetKeyMap);
          filteredTargets
              .stream()
              .map(SkyQueryEnvironment.TARGET_TO_SKY_KEY)
              .forEachOrdered(
                  rdep -> {
                    if (validRdepUniquifier.unique(rdep)) {
                      filteredUniqueKeys.add(rdep);
                    }
                  });
          }
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForTargetification);
      }

      // Retrieve the reverse deps as SkyKeys and defer the targetification and filtering to next
      // recursive visitation.
      Map<SkyKey, Iterable<SkyKey>> unfilteredReverseDeps =
          env.graph.getReverseDeps(filteredUniqueKeys);

      ImmutableList.Builder<DepAndRdep> builder = ImmutableList.builder();
      for (Map.Entry<SkyKey, Iterable<SkyKey>> rdeps : unfilteredReverseDeps.entrySet()) {
        for (SkyKey rdep : rdeps.getValue()) {
          Label label = SkyQueryEnvironment.SKYKEY_TO_LABEL.apply(rdep);
          if (label != null) {
            builder.add(new DepAndRdep(rdeps.getKey(), rdep));
          }
        }
      }

      return new Visit(
          /*keysToUseForResult=*/ filteredUniqueKeys, /*keysToVisit=*/ builder.build());
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

    @Override
    protected Iterable<DepAndRdep> preprocessInitialVisit(Iterable<SkyKey> keys) {
      return Iterables.transform(keys, key -> new DepAndRdep(null, key));
    }

    @Override
    protected Iterable<Task> getVisitTasks(Collection<DepAndRdep> pendingKeysToVisit) {
      // Group pending (dep, rdep) visits by the package of the rdep, since we'll be targetfying the
      // rdep during the visitation.
      ListMultimap<PackageIdentifier, DepAndRdep> visitsByPackage =
          ArrayListMultimap.create();
      for (DepAndRdep depAndRdep : pendingKeysToVisit) {
        Label label = SkyQueryEnvironment.SKYKEY_TO_LABEL.apply(depAndRdep.rdep);
        if (label != null) {
          visitsByPackage.put(label.getPackageIdentifier(), depAndRdep);
        }
      }

      ImmutableList.Builder<Task> builder = ImmutableList.builder();

      // A couple notes here:
      // (i)  ArrayListMultimap#values returns the values grouped by key, which is exactly what we
      //      want.
      // (ii) ArrayListMultimap#values returns a Collection view, so we make a copy to avoid
      //      accidentally retaining the entire ArrayListMultimap object.
      for (Iterable<DepAndRdep> depAndRdepBatch :
          Iterables.partition(ImmutableList.copyOf(visitsByPackage.values()), VISIT_BATCH_SIZE)) {
        builder.add(new VisitTask(depAndRdepBatch));
      }

      return builder.build();
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
      // Even so, there's value in not visiting a rdep if it's already been visiting *validly*
      // before. We use the intentionally racy {@link Uniquifier#uniquePure} to attempt to do this.
      return depAndRdepUniquifier.unique(
          Iterables.filter(
              depAndRdeps,
              depAndRdep -> validRdepUniquifier.uniquePure(depAndRdep.rdep)));
    }
  }
}

