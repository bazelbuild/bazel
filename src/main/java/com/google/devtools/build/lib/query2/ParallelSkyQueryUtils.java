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
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
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
      Callback<Target> callback,
      MultisetSemaphore<PackageIdentifier> packageSemaphore)
          throws QueryException, InterruptedException {
    Uniquifier<SkyKey> keyUniquifier = env.createSkyKeyUniquifier();
    RBuildFilesVisitor visitor =
        new RBuildFilesVisitor(env, keyUniquifier, callback, packageSemaphore);
    visitor.visitAndWaitForCompletion(env.getSkyKeysForFileFragments(fileIdentifiers));
  }

  /** A helper class that computes 'rbuildfiles(<blah>)' via BFS. */
  private static class RBuildFilesVisitor extends ParallelVisitor<SkyKey> {
    private final SkyQueryEnvironment env;
    private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

    private RBuildFilesVisitor(
        SkyQueryEnvironment env,
        Uniquifier<SkyKey> uniquifier,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(uniquifier, callback, VISIT_BATCH_SIZE);
      this.env = env;
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
          Streams.stream(keysToUseForResult)
              .map(SkyQueryEnvironment.PACKAGE_SKYKEY_TO_PACKAGE_IDENTIFIER)
              .collect(toImmutableSet());
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
  private static class AllRdepsUnboundedVisitor extends ParallelVisitor<Pair<SkyKey, SkyKey>> {
    private final SkyQueryEnvironment env;
    private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

    private AllRdepsUnboundedVisitor(
        SkyQueryEnvironment env,
        Uniquifier<Pair<SkyKey, SkyKey>> uniquifier,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
      super(uniquifier, callback, VISIT_BATCH_SIZE);
      this.env = env;
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
      private final Uniquifier<Pair<SkyKey, SkyKey>> uniquifier;
      private final Callback<Target> callback;
      private final MultisetSemaphore<PackageIdentifier> packageSemaphore;

      private Factory(
        SkyQueryEnvironment env,
        Callback<Target> callback,
        MultisetSemaphore<PackageIdentifier> packageSemaphore) {
        this.env = env;
        this.uniquifier = env.createReverseDepSkyKeyUniquifier();
        this.callback = callback;
        this.packageSemaphore = packageSemaphore;
      }

      @Override
      public ParallelVisitor<Pair<SkyKey, SkyKey>> create() {
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

        reverseDepsMap.computeIfAbsent(reverseDepPair.first, k -> new LinkedList<SkyKey>());

        reverseDepsMap.get(reverseDepPair.first).add(reverseDepPair.second);
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
              .forEachOrdered(filteredKeys::add);
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
    protected Iterable<Pair<SkyKey, SkyKey>> preprocessInitialVisit(Iterable<SkyKey> keys) {
      return Iterables.transform(keys, key -> Pair.of(null, key));
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
}

