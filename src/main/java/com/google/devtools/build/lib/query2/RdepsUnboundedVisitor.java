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
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.ParallelSkyQueryUtils.DepAndRdep;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Set;

/**
 * A helper class that computes unbounded 'allrdeps(<expr>)' or 'rdeps(<precomputed-universe>,
 * <expr>)' via BFS.
 *
 * <p>The visitor uses {@link DepAndRdep} to keep track the nodes to visit and avoid dealing with
 * targetification of reverse deps until they are needed. The rdep node itself is needed to filter
 * out disallowed deps later. Compared against the approach using a single SkyKey, it consumes 16
 * more bytes in a 64-bit environment for each edge. However it defers the need to load all the
 * packages which have at least a target as a rdep of the current batch, thus greatly reduces the
 * risk of OOMs. The additional memory usage should not be a large concern here, as even with 10M
 * edges, the memory overhead is around 160M, and the memory can be reclaimed by regular GC.
 */
class RdepsUnboundedVisitor extends AbstractTargetOuputtingVisitor<DepAndRdep> {
  /**
   * A {@link Uniquifier} for visitations. Solely used for {@link
   * #noteAndReturnUniqueVisitationKeys}, which actually isn't that useful. See the method javadoc.
   */
  private final Uniquifier<DepAndRdep> depAndRdepUniquifier;
  /**
   * A {@link Uniquifier} for *valid* visitations of rdeps. {@code env}'s dependency filter might
   * mean that some rdep edges are invalid, meaning that any individual {@link DepAndRdep}
   * visitation may actually be invalid. Because the same rdep can be reached through more than one
   * reverse edge, it'd be incorrect to naively dedupe visitations solely based on the rdep.
   */
  private final Uniquifier<SkyKey> validRdepUniquifier;

  private final Predicate<SkyKey> unfilteredUniverse;

  RdepsUnboundedVisitor(
      SkyQueryEnvironment env,
      Uniquifier<DepAndRdep> depAndRdepUniquifier,
      Uniquifier<SkyKey> validRdepUniquifier,
      Predicate<SkyKey> unfilteredUniverse,
      Callback<Target> callback) {
    super(env, callback);
    this.depAndRdepUniquifier = depAndRdepUniquifier;
    this.validRdepUniquifier = validRdepUniquifier;
    this.unfilteredUniverse = unfilteredUniverse;
  }

  /**
   * A {@link Factory} for {@link RdepsUnboundedVisitor} instances, each of which will be used to
   * perform visitation of the reverse transitive closure of the {@link Target}s passed in a single
   * {@link Callback#process} call. Note that all the created instances share the same {@link
   * Uniquifier} so that we don't visit the same Skyframe node more than once.
   */
  static class Factory implements ParallelVisitor.Factory<DepAndRdep, SkyKey, Target> {
    private final SkyQueryEnvironment env;
    private final Uniquifier<DepAndRdep> depAndRdepUniquifier;
    private final Uniquifier<SkyKey> validRdepUniquifier;
    private final Predicate<SkyKey> unfilteredUniverse;
    private final Callback<Target> callback;

    Factory(
        SkyQueryEnvironment env, Predicate<SkyKey> unfilteredUniverse, Callback<Target> callback) {
      this.env = env;
      this.unfilteredUniverse = unfilteredUniverse;
      this.depAndRdepUniquifier =
          new UniquifierImpl<>(depAndRdep -> depAndRdep, env.getQueryEvaluationParallelismLevel());
      this.validRdepUniquifier = env.createSkyKeyUniquifier();
      this.callback = callback;
    }

    @Override
    public ParallelVisitor<DepAndRdep, SkyKey, Target> create() {
      return new RdepsUnboundedVisitor(
          env, depAndRdepUniquifier, validRdepUniquifier, unfilteredUniverse, callback);
    }
  }

  @Override
  protected Visit getVisitResult(Iterable<DepAndRdep> depAndRdeps)
      throws QueryException, InterruptedException {
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

    ImmutableList.Builder<SkyKey> uniqueValidRdepsbuilder = ImmutableList.builder();
    for (SkyKey rdep : validRdeps) {
      if (validRdepUniquifier.unique(rdep)) {
        uniqueValidRdepsbuilder.add(rdep);
      }
    }
    ImmutableList<SkyKey> uniqueValidRdeps = uniqueValidRdepsbuilder.build();

    // Retrieve the reverse deps as SkyKeys and defer the targetification and filtering to next
    // recursive visitation. Because the universe given to us is unfiltered, we definitely still
    // need to filter out disallowed edges, but cannot do so before targetification occurs. This
    // means we may be wastefully visiting nodes via disallowed edges.
    ImmutableList.Builder<DepAndRdep> depAndRdepsToVisitBuilder = ImmutableList.builder();
    env.graph
        .getReverseDeps(uniqueValidRdeps)
        .entrySet()
        .forEach(
            reverseDepsEntry ->
                depAndRdepsToVisitBuilder.addAll(
                    Iterables.transform(
                        Iterables.filter(
                            reverseDepsEntry.getValue(),
                            Predicates.and(SkyQueryEnvironment.IS_TTV, unfilteredUniverse)),
                        rdep -> new DepAndRdep(reverseDepsEntry.getKey(), rdep))));

    return new Visit(
        /*keysToUseForResult=*/ uniqueValidRdeps,
        /*keysToVisit=*/ depAndRdepsToVisitBuilder.build());
  }

  @Override
  protected SkyKey visitationKeyToOutputKey(DepAndRdep visitationKey) {
    return visitationKey.rdep;
  }

  @Override
  protected Iterable<DepAndRdep> noteAndReturnUniqueVisitationKeys(
      Iterable<DepAndRdep> prospectiveVisitationKeys) throws QueryException {
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
            prospectiveVisitationKeys,
            depAndRdep -> validRdepUniquifier.uniquePure(depAndRdep.rdep)));
  }

  @Override
  protected Iterable<DepAndRdep> preprocessInitialVisit(Iterable<SkyKey> skyKeys) {
    return Iterables.transform(
        Iterables.filter(skyKeys, k -> unfilteredUniverse.apply(k)),
        key -> new DepAndRdep(/*dep=*/ null, key));
  }
}
