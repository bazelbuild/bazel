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

import static com.google.devtools.build.lib.query2.SkyQueryEnvironment.IS_TTV;
import static com.google.devtools.build.lib.query2.SkyQueryEnvironment.SKYKEY_TO_LABEL;

import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
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

  DepsUnboundedVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> validDepUniquifier,
      Callback<Target> callback,
      boolean depsNeedFiltering,
      QueryExpressionContext<Target> context) {
    super(env, callback);
    this.validDepUniquifier = validDepUniquifier;
    this.depsNeedFiltering = depsNeedFiltering;
    this.context = context;
  }

  /**
   * A {@link Factory} for {@link DepsUnboundedVisitor} instances, each of which will be used to
   * perform visitation of the DTC of the {@link SkyKey}s passed in a single {@link
   * Callback#process} call. Note that all the created instances share the same {@link Uniquifier}
   * so that we don't visit the same Skyframe node more than once.
   */
  static class Factory implements ParallelVisitor.Factory<SkyKey, SkyKey, Target> {
    private final SkyQueryEnvironment env;
    private final Uniquifier<SkyKey> validDepUniquifier;
    private final Callback<Target> callback;
    private final boolean depsNeedFiltering;
    private final QueryExpressionContext<Target> context;

    Factory(
        SkyQueryEnvironment env,
        Callback<Target> callback,
        boolean depsNeedFiltering,
        QueryExpressionContext<Target> context) {
      this.env = env;
      this.validDepUniquifier = env.createSkyKeyUniquifier();
      this.callback = callback;
      this.depsNeedFiltering = depsNeedFiltering;
      this.context = context;
    }

    @Override
    public ParallelVisitor<SkyKey, SkyKey, Target> create() {
      return new DepsUnboundedVisitor(
          env,
          validDepUniquifier,
          callback,
          depsNeedFiltering,
          context);
    }
  }

  @Override
  protected Visit getVisitResult(Iterable<SkyKey> keys) throws InterruptedException {
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

    // We need to explicitly check that all requested TTVs are actually in the graph.
    Map<SkyKey, Iterable<SkyKey>> depMap = env.graph.getDirectDeps(keys);
    checkIfMissingTargets(keys, depMap);
    Iterable<SkyKey> deps = Iterables.filter(Iterables.concat(depMap.values()), IS_TTV);
    return new Visit(keys, deps);
  }

  private void checkIfMissingTargets(Iterable<SkyKey> keys, Map<SkyKey, Iterable<SkyKey>> depMap) {
    if (depMap.size() != Iterables.size(keys)) {
      Iterable<Label> missingTargets =
          Iterables.transform(
              Iterables.filter(keys, Predicates.not(Predicates.in(depMap.keySet()))),
              SKYKEY_TO_LABEL);
      env.getEventHandler()
          .handle(Event.warn("Targets were missing from graph: " + missingTargets));
    }
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
}
