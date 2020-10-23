// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.ParallelSkyQueryUtils.DepAndRdep;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Set;

class RdepsVisitorUtils {
  private RdepsVisitorUtils() {}

  static Iterable<SkyKey> getMaybeFilteredRdeps(
      Iterable<DepAndRdep> depAndRdeps, SkyQueryEnvironment env) throws InterruptedException {
    return env.hasDependencyFilter() ? getFilteredRdeps(depAndRdeps, env) : getRdeps(depAndRdeps);
  }

  private static Iterable<SkyKey> getFilteredRdeps(
      Iterable<DepAndRdep> depAndRdeps, SkyQueryEnvironment env) throws InterruptedException {
    ArrayList<SkyKey> filteredRdeps = new ArrayList<>();

    Multimap<SkyKey, SkyKey> reverseDepMultimap = ArrayListMultimap.create();
    for (DepAndRdep depAndRdep : depAndRdeps) {
      if (depAndRdep.dep == null) {
        filteredRdeps.add(depAndRdep.rdep);
      } else {
        reverseDepMultimap.put(depAndRdep.dep, depAndRdep.rdep);
      }
    }

    Multimap<SkyKey, SkyKey> packageKeyToTargetKeyMap =
        SkyQueryEnvironment.makePackageKeyToTargetKeyMap(
            Iterables.concat(reverseDepMultimap.values()));
    Set<PackageIdentifier> pkgIdsNeededForTargetification =
        SkyQueryEnvironment.getPkgIdsNeededForTargetification(packageKeyToTargetKeyMap);

    MultisetSemaphore<PackageIdentifier> packageSemaphore = env.getPackageMultisetSemaphore();
    packageSemaphore.acquireAll(pkgIdsNeededForTargetification);
    try {
      if (!reverseDepMultimap.isEmpty()) {
        Collection<Target> filteredTargets =
            env.filterRawReverseDepsOfTransitiveTraversalKeys(
                reverseDepMultimap.asMap(), packageKeyToTargetKeyMap);
        filteredTargets.stream()
            .map(SkyQueryEnvironment.TARGET_TO_SKY_KEY)
            .forEachOrdered(filteredRdeps::add);
      }
    } finally {
      packageSemaphore.releaseAll(pkgIdsNeededForTargetification);
    }

    return filteredRdeps;
  }

  private static Iterable<SkyKey> getRdeps(Iterable<DepAndRdep> depAndRdeps) {
    return Iterables.transform(depAndRdeps, depAndRdep -> depAndRdep.rdep);
  }
}
