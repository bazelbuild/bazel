// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.TestSuiteExpansionValue.TestSuiteExpansion;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * TestSuiteExpansionFunction takes a list of targets and expands all test suites in those targets.
 */
final class TestSuiteExpansionFunction implements SkyFunction {
  @Override
  public SkyValue compute(SkyKey key, Environment env) {
    TestSuiteExpansion expansion = (TestSuiteExpansion) key.argument();
    ResolvedTargets<Target> targets = labelsToTargets(env, expansion.getTargets(), false);
    List<SkyKey> testsInSuitesKeys = new ArrayList<>();
    for (Target target : targets.getTargets()) {
      if (TargetUtils.isTestSuiteRule(target)) {
        testsInSuitesKeys.add(TestsInSuiteValue.key(target, true));
      }
    }
    Map<SkyKey, SkyValue> testsInSuites = env.getValues(testsInSuitesKeys);
    if (env.valuesMissing()) {
      return null;
    }

    Set<Target> result = new LinkedHashSet<>();
    boolean hasError = targets.hasError();
    for (Target target : targets.getTargets()) {
      if (TargetUtils.isTestRule(target)) {
        result.add(target);
      } else if (TargetUtils.isTestSuiteRule(target)) {
        TestsInSuiteValue value = (TestsInSuiteValue) testsInSuites.get(
            TestsInSuiteValue.key(target, true));
        if (value != null) {
          result.addAll(value.getTargets().getTargets());
          hasError |= value.getTargets().hasError();
        }
      } else {
        result.add(target);
      }
    }
    if (env.valuesMissing()) {
      return null;
    }
    // We use ResolvedTargets in order to associate an error flag; the result should never contain
    // any filtered targets.
    return new TestSuiteExpansionValue(new ResolvedTargets<Target>(result, hasError));
  }

  static ResolvedTargets<Target> labelsToTargets(
      Environment env, ImmutableSet<Label> labels, boolean hasError) {
    Set<PackageIdentifier> pkgIdentifiers = new LinkedHashSet<>();
    for (Label label : labels) {
      pkgIdentifiers.add(label.getPackageIdentifier());
    }
    // Don't bother to check for exceptions - the incoming list should only contain valid targets.
    Map<SkyKey, SkyValue> packages = env.getValues(PackageValue.keys(pkgIdentifiers));
    if (env.valuesMissing()) {
      return null;
    }

    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    builder.mergeError(hasError);
    Map<PackageIdentifier, Package> packageMap = new HashMap<>();
    for (Entry<SkyKey, SkyValue> entry : packages.entrySet()) {
      packageMap.put(
          (PackageIdentifier) entry.getKey().argument(),
          ((PackageValue) entry.getValue()).getPackage());
    }

    for (Label label : labels) {
      Package pkg = packageMap.get(label.getPackageIdentifier());
      if (pkg == null) {
        continue;
      }
      try {
        builder.add(pkg.getTarget(label.getName()));
        if (pkg.containsErrors()) {
          builder.setError();
        }
      } catch (NoSuchTargetException e) {
        builder.setError();
      }
    }
    return builder.build();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
