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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.ParallelSkyQueryUtils.DepAndRdep;
import com.google.devtools.build.lib.query2.ParallelSkyQueryUtils.DepAndRdepAtDepth;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * A helper class that computes bounded 'allrdeps(<expr>, <depth>)' or
 * 'rdeps(<precomputed-universe>, <expr>, <depth>)' via BFS.
 *
 * <p>This is very similar to {@link RdepsUnboundedVisitor}. A lot of the same concerns apply here
 * but there are additional subtle concerns about the correctness of the bounded traversal: just
 * like for the sequential implementation of bounded allrdeps, we use {@link MinDepthUniquifier}.
 */
class RdepsBoundedVisitor extends AbstractTargetOuputtingVisitor<DepAndRdepAtDepth> {
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
      Callback<Target> callback) {
    super(env, callback);
    this.depth = depth;
    this.depAndRdepAtDepthUniquifier = depAndRdepAtDepthUniquifier;
    this.validRdepMinDepthUniquifier = validRdepMinDepthUniquifier;
    this.universe = universe;
  }

  static class Factory implements ParallelVisitor.Factory<DepAndRdepAtDepth, SkyKey, Target> {
    private final SkyQueryEnvironment env;
    private final int depth;
    private final Uniquifier<DepAndRdepAtDepth> depAndRdepAtDepthUniquifier;
    private final MinDepthUniquifier<SkyKey> validRdepMinDepthUniquifier;
    private final Predicate<SkyKey> universe;
    private final Callback<Target> callback;

    Factory(
        SkyQueryEnvironment env, int depth, Predicate<SkyKey> universe, Callback<Target> callback) {
      this.env = env;
      this.depth = depth;
      this.universe = universe;
      this.depAndRdepAtDepthUniquifier =
          new UniquifierImpl<>(
              depAndRdepAtDepth -> depAndRdepAtDepth, env.getQueryEvaluationParallelismLevel());
      this.validRdepMinDepthUniquifier = env.createMinDepthSkyKeyUniquifier();
      this.callback = callback;
    }

    @Override
    public ParallelVisitor<DepAndRdepAtDepth, SkyKey, Target> create() {
      return new RdepsBoundedVisitor(
          env, depth, depAndRdepAtDepthUniquifier, validRdepMinDepthUniquifier, universe, callback);
    }
  }

  @Override
  protected Visit getVisitResult(Iterable<DepAndRdepAtDepth> depAndRdepAtDepths)
      throws InterruptedException {
    Map<SkyKey, Integer> shallowestRdepDepthMap = new HashMap<>();
    depAndRdepAtDepths.forEach(
        depAndRdepAtDepth ->
            shallowestRdepDepthMap.merge(
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
        reverseDepMultimap.put(depAndRdepAtDepth.depAndRdep.dep, depAndRdepAtDepth.depAndRdep.rdep);
      }
    }

    Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
        SkyQueryEnvironment.makePackageKeyToTargetKeyMap(
            Iterables.concat(reverseDepMultimap.values()));
    Set<PackageIdentifier> pkgIdsNeededForTargetification =
        SkyQueryEnvironment.getPkgIdsNeededForTargetification(packageKeyToTargetKeyMap);
    MultisetSemaphore<PackageIdentifier> packageSemaphore = getPackageSemaphore();
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

    ImmutableList<SkyKey> uniqueValidRdeps =
        validRdeps
            .stream()
            .filter(
                validRdep ->
                    validRdepMinDepthUniquifier.uniqueAtDepthLessThanOrEqualTo(
                        validRdep, shallowestRdepDepthMap.get(validRdep)))
            .collect(ImmutableList.toImmutableList());

    // Don't bother getting the rdeps of the rdeps that are already at the depth bound.
    Iterable<SkyKey> uniqueValidRdepsBelowDepthBound =
        Iterables.filter(
            uniqueValidRdeps,
            uniqueValidRdep -> shallowestRdepDepthMap.get(uniqueValidRdep) < depth);

    // Retrieve the reverse deps as SkyKeys and defer the targetification and filtering to next
    // recursive visitation.
    Map<SkyKey, Iterable<SkyKey>> unfilteredRdepsOfRdeps =
        env.graph.getReverseDeps(uniqueValidRdepsBelowDepthBound);

    ImmutableList.Builder<DepAndRdepAtDepth> depAndRdepAtDepthsToVisitBuilder =
        ImmutableList.builder();
    unfilteredRdepsOfRdeps
        .entrySet()
        .forEach(
            entry -> {
              SkyKey rdep = entry.getKey();
              int depthOfRdepOfRdep = shallowestRdepDepthMap.get(rdep) + 1;
              Streams.stream(entry.getValue())
                  .filter(Predicates.and(SkyQueryEnvironment.IS_TTV, universe))
                  .forEachOrdered(
                      rdepOfRdep -> {
                        depAndRdepAtDepthsToVisitBuilder.add(
                            new DepAndRdepAtDepth(
                                new DepAndRdep(rdep, rdepOfRdep), depthOfRdepOfRdep));
                      });
            });

    return new Visit(
        /*keysToUseForResult=*/ uniqueValidRdeps,
        /*keysToVisit=*/ depAndRdepAtDepthsToVisitBuilder.build());
  }

  @Override
  protected SkyKey visitationKeyToOutputKey(DepAndRdepAtDepth visitationKey) {
    return visitationKey.depAndRdep.rdep;
  }

  @Override
  protected Iterable<DepAndRdepAtDepth> noteAndReturnUniqueVisitationKeys(
      Iterable<DepAndRdepAtDepth> prospectiveVisitationKeys) throws QueryException {
    // See the comment in RdepsUnboundedVisitor#getUniqueValues.
    return depAndRdepAtDepthUniquifier.unique(
        Iterables.filter(
            prospectiveVisitationKeys,
            depAndRdepAtDepth ->
                validRdepMinDepthUniquifier.uniqueAtDepthLessThanOrEqualToPure(
                    depAndRdepAtDepth.depAndRdep.rdep, depAndRdepAtDepth.rdepDepth)));
  }

  @Override
  protected Iterable<DepAndRdepAtDepth> preprocessInitialVisit(Iterable<SkyKey> skyKeys) {
    return Iterables.transform(
        skyKeys, key -> new DepAndRdepAtDepth(new DepAndRdep(/*dep=*/ null, key), 0));
  }
}
