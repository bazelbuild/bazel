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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.ParallelVisitorUtils.ParallelQueryVisitor;
import com.google.devtools.build.lib.query2.ParallelVisitorUtils.QueryVisitorFactory;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import java.util.Set;

/**
 * A helper class that computes unbounded 'deps(<expr>)' via BFS. Re-use logic from {@link
 * AbstractTargetOuputtingVisitor} to grab relevant semaphore locks when processing nodes as well as
 * batching visits by packages.
 */
class DepsUnboundedVisitor extends AbstractTargetOuputtingVisitor<SkyKey> {
  /**
   * A {@link Uniquifier} for valid deps. Only used prior to visiting the deps. Deps filtering is
   * done in the {@link DepsUnboundedVisitor#getVisitResult} stage.
   */
  private final Uniquifier<SkyKey> validDepUniquifier;

  private final boolean depsNeedFiltering;
  private final QueryExpressionContext<Target> context;
  private final QueryExpression caller;

  DepsUnboundedVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> validDepUniquifier,
      Callback<Target> callback,
      boolean depsNeedFiltering,
      QueryExpressionContext<Target> context,
      QueryExpression caller) {
    super(env, callback);
    this.validDepUniquifier = validDepUniquifier;
    this.depsNeedFiltering = depsNeedFiltering;
    this.context = context;
    this.caller = caller;
  }

  /**
   * A {@link Factory} for {@link DepsUnboundedVisitor} instances, each of which will be used to
   * perform visitation of the DTC of the {@link SkyKey}s passed in a single {@link
   * Callback#process} call. Note that all the created instances share the same {@link Uniquifier}
   * so that we don't visit the same Skyframe node more than once.
   */
  static class Factory implements QueryVisitorFactory<SkyKey, SkyKey, Target> {
    private final SkyQueryEnvironment env;
    private final Uniquifier<SkyKey> validDepUniquifier;
    private final Callback<Target> callback;
    private final boolean depsNeedFiltering;
    private final QueryExpressionContext<Target> context;
    private final QueryExpression caller;

    Factory(
        SkyQueryEnvironment env,
        Callback<Target> callback,
        boolean depsNeedFiltering,
        QueryExpressionContext<Target> context,
        QueryExpression caller) {
      this.env = env;
      this.validDepUniquifier = env.createSkyKeyUniquifier();
      this.callback = callback;
      this.depsNeedFiltering = depsNeedFiltering;
      this.context = context;
      this.caller = caller;
    }

    @Override
    public ParallelQueryVisitor<SkyKey, SkyKey, Target> create() {
      return new DepsUnboundedVisitor(
          env, validDepUniquifier, callback, depsNeedFiltering, context, caller);
    }
  }

  @Override
  protected Visit getVisitResult(Iterable<SkyKey> keys)
      throws InterruptedException, QueryException {
    if (depsNeedFiltering) {
      // We have to targetify the keys here in order to determine the allowed dependencies.
      Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
          SkyQueryEnvironment.makePackageKeyToTargetKeyMap(keys);
      Set<PackageIdentifier> pkgIdsNeededForTargetification =
          SkyQueryEnvironment.getPkgIdsNeededForTargetification(packageKeyToTargetKeyMap);
      MultisetSemaphore<PackageIdentifier> packageSemaphore = getPackageSemaphore();
      packageSemaphore.acquireAll(pkgIdsNeededForTargetification);
      Iterable<SkyKey> depsAsSkyKeys;
      try {
        Iterable<Target> depsAsTargets =
            env.getFwdDeps(
                env.getTargetKeyToTargetMapForPackageKeyToTargetKeyMap(
                    packageKeyToTargetKeyMap).values(),
                context);
        depsAsSkyKeys = Iterables.transform(depsAsTargets, Target::getLabel);
      } finally {
        packageSemaphore.releaseAll(pkgIdsNeededForTargetification);
      }

      return new Visit(
          /*keysToUseForResult=*/ keys,
          /*keysToVisit=*/ depsAsSkyKeys);
    }

    return new Visit(
        keys, ImmutableSet.copyOf(Iterables.concat(env.getFwdDepLabels(keys).values())));
  }

  @Override
  protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> visitationKeys) {
    return visitationKeys;
  }

  @Override
  protected SkyKey visitationKeyToOutputKey(SkyKey visitationKey) {
    return visitationKey;
  }

  @Override
  protected Iterable<SkyKey> noteAndReturnUniqueVisitationKeys(
      Iterable<SkyKey> prospectiveVisitationKeys) throws QueryException {
    return validDepUniquifier.unique(prospectiveVisitationKeys);
  }

  @Override
  protected void handleMissingTargets(
      Map<? extends SkyKey, Target> keysWithTargets, Set<SkyKey> targetKeys)
      throws QueryException, InterruptedException {
    env.reportUnsuccessfulOrMissingTargets(keysWithTargets, targetKeys, caller);
  }
}
