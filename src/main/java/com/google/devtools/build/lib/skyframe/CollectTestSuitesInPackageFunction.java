// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.skyframe.GraphTraversingHelper;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} to ensure dependencies on other such nodes for dependent packages inside
 * test_suite rules' "tests" attribute.
 */
public class CollectTestSuitesInPackageFunction implements SkyFunction {
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    PackageIdentifier packageId = (PackageIdentifier) skyKey.argument();
    PackageValue packageValue;
    try {
      packageValue = (PackageValue) env.getValueOrThrow(packageId, NoSuchPackageException.class);
    } catch (NoSuchPackageException e) {
      // If the argument is a package that doesn't exist, the aggregator function can return
      // a success value immediately.
      return CollectTestSuitesInPackageValue.INSTANCE;
    }
    if (env.valuesMissing()) {
      return null;
    }
    Package pkg = packageValue.getPackage();
    if (pkg.containsErrors()) {
      env.getListener()
          .handle(
              Event.error(
                  "package contains errors: " + packageId.getPackageFragment().getPathString()));
    }

    // Force a dependency on any CollectTestSuitesInPackage(pkg) where pkg is any test_suite
    // test dependency.
    Set<Label> testTargets =
        pkg.getTargets().values().stream()
            .filter(TargetUtils::isTestSuiteRule)
            .flatMap(
                target ->
                    AggregatingAttributeMapper.of(((Rule) target))
                        .getReachableLabels("tests", /*includeSelectKeys=*/ false).stream())
            .collect(ImmutableSet.toImmutableSet());
    Set<SkyKey> collectTestSuiteInPkgDeps = new HashSet<>();
    for (Label label : testTargets) {
      collectTestSuiteInPkgDeps.add(
          CollectTestSuitesInPackageValue.key(label.getPackageIdentifier()));
    }
    collectTestSuiteInPkgDeps.remove(skyKey);
    return GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
            env, collectTestSuiteInPkgDeps)
        ? null
        : CollectTestSuitesInPackageValue.INSTANCE;
  }
}
