// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genquery;

import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.TransitiveTargetKey;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Collection;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Factory for {@link GenQueryPackageProvider} using {@link TransitiveTargetValue}-based Skyframe
 * work to collect required information.
 */
public class GenQueryTtvPackageProviderFactory implements GenQueryPackageProviderFactory {

  /**
   * Precomputes the transitive closure of the scope. Returns two maps: one identifying the
   * successful packages, and the other identifying the valid targets. Breaks in the transitive
   * closure of the scope will cause the query to error out early.
   */
  @Override
  @Nullable
  public GenQueryPackageProvider constructPackageMap(Environment env, ImmutableList<Label> scope)
      throws InterruptedException, BrokenQueryScopeException {
    // It is not necessary for correctness to construct intermediate NestedSets; we could iterate
    // over individual targets in scope immediately. However, creating a composite NestedSet first
    // saves us from iterating over the same sub-NestedSets multiple times.
    NestedSetBuilder<Label> validTargets = NestedSetBuilder.stableOrder();
    Set<SkyKey> successfulPackageKeys = Sets.newHashSetWithExpectedSize(scope.size());
    Collection<SkyKey> transitiveTargetKeys =
        Collections2.transform(scope, TransitiveTargetKey::of);
    SkyframeLookupResult transitiveTargetValues = env.getValuesAndExceptions(transitiveTargetKeys);
    if (env.valuesMissing()) {
      return null;
    }
    for (SkyKey skyKey : transitiveTargetKeys) {
      TransitiveTargetValue ttv;
      try {
        ttv =
            (TransitiveTargetValue)
                transitiveTargetValues.getOrThrow(
                    skyKey, NoSuchPackageException.class, NoSuchTargetException.class);
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        throw BrokenQueryScopeException.of(e);
      }
      if (ttv == null) {
        BugReport.sendNonFatalBugReport(
            new IllegalStateException(
                "TransitiveTargetValue " + skyKey + " was missing, this should never happen"));
        continue;
      }
      if (ttv.encounteredLoadingError()) {
        throw BrokenQueryScopeException.of(ttv.getErrorLoadingTarget());
      }
      validTargets.addTransitive(ttv.getTransitiveTargets());
      for (Label transitiveLabel : ttv.getTransitiveTargets().toList()) {
        successfulPackageKeys.add(transitiveLabel.getPackageIdentifier());
      }
    }

    // Construct the package id to package map for all successful packages.
    SkyframeLookupResult transitivePackages = env.getValuesAndExceptions(successfulPackageKeys);
    if (env.valuesMissing()) {
      // Packages from an untaken select branch could be missing: analysis avoids these, but query
      // does not.
      return null;
    }
    ImmutableMap.Builder<PackageIdentifier, Package> packageMapBuilder = ImmutableMap.builder();
    for (SkyKey skyKey : successfulPackageKeys) {
      PackageValue pkg = (PackageValue) transitivePackages.get(skyKey);
      if (pkg == null) {
        BugReport.sendNonFatalBugReport(
            new IllegalStateException(
                "SkyValue " + skyKey + " was missing, this should never happen"));
        return null;
      }
      Preconditions.checkState(
          !pkg.getPackage().containsErrors(),
          "package %s was found to both have and not have errors.",
          skyKey);
      packageMapBuilder.put(pkg.getPackage().getPackageIdentifier(), pkg.getPackage());
    }
    ImmutableMap<PackageIdentifier, Package> packageMap = packageMapBuilder.buildOrThrow();
    ImmutableMap.Builder<Label, Target> validTargetsMapBuilder = ImmutableMap.builder();
    for (Label label : validTargets.build().toList()) {
      try {
        Target target = packageMap.get(label.getPackageIdentifier()).getTarget(label.getName());
        validTargetsMapBuilder.put(label, target);
      } catch (NoSuchTargetException e) {
        throw new IllegalStateException(e);
      }
    }
    return new GenQueryPackageProvider(packageMap, validTargetsMapBuilder.buildOrThrow());
  }
}
