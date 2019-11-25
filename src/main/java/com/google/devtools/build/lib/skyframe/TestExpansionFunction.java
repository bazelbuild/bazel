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
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestTargetUtils;
import com.google.devtools.build.lib.skyframe.TestExpansionValue.TestExpansionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * TestExpansionFunction takes a single test_suite target and expands all of the tests it contains,
 * possibly recursively.
 */
// TODO(ulfjack): What about test_suite rules that include each other.
final class TestExpansionFunction implements SkyFunction {
  @Override
  public SkyValue compute(SkyKey key, Environment env) throws InterruptedException {
    TestExpansionKey expansion = (TestExpansionKey) key.argument();
    SkyKey packageKey = PackageValue.key(expansion.getLabel().getPackageIdentifier());
    PackageValue pkg = (PackageValue) env.getValue(packageKey);
    if (env.valuesMissing()) {
      return null;
    }
    Rule rule = pkg.getPackage().getRule(expansion.getLabel().getName());
    ResolvedTargets<Label> result = computeExpandedTests(env, rule, expansion.isStrict());
    if (env.valuesMissing()) {
      return null;
    }
    return new TestExpansionValue(result);
  }

  private static Set<Label> toLabels(Set<Target> targets) {
    return targets.stream().map(Target::getLabel).collect(Collectors.toSet());
  }

  /**
   * Populates 'result' with all the tests associated with the specified 'rule'. Throws an exception
   * if any target is missing.
   *
   * <p>CAUTION! Keep this logic consistent with {@code TestSuite}!
   */
  private static ResolvedTargets<Label> computeExpandedTests(
      Environment env, Rule rule, boolean strict) throws InterruptedException {
    Set<Target> result = new HashSet<>();
    boolean hasError = false;

    List<Target> prerequisites = new ArrayList<>();
    // Note that prerequisites can contain input file targets; the test_suite rule does not
    // restrict the set of targets that can appear in tests or suites.
    hasError |= getPrerequisites(env, rule, "tests", prerequisites);

    // 1. Add all tests
    for (Target test : prerequisites) {
      if (TargetUtils.isTestRule(test)) {
        result.add(test);
      } else if (strict && !TargetUtils.isTestSuiteRule(test)) {
        // If strict mode is enabled, then give an error for any non-test, non-test-suite targets.
        // TODO(ulfjack): We need to throw to end the process if we happen to be in --nokeep_going,
        // but we can't know whether or not we are at this point.
        env.getListener()
            .handle(
                Event.error(
                    rule.getLocation(),
                    "in test_suite rule '"
                        + rule.getLabel()
                        + "': expecting a test or a test_suite rule but '"
                        + test.getLabel()
                        + "' is not one."));
        hasError = true;
      }
    }

    // 2. Add implicit dependencies on tests in same package, if any.
    List<Target> implicitTests = new ArrayList<>();
    hasError |= getPrerequisites(env, rule, "$implicit_tests", implicitTests);
    for (Target target : implicitTests) {
      // The Package construction of $implicit_tests ensures that this check never fails, but we
      // add it here anyway for compatibility with future code.
      if (TargetUtils.isTestRule(target)) {
        result.add(target);
      }
    }

    // 3. Filter based on tags, size, env.
    TestTargetUtils.filterTests(rule, result);

    // 4. Expand all rules recursively, collecting labels.
    ResolvedTargets.Builder<Label> labelsBuilder = ResolvedTargets.builder();
    // Don't set filtered targets; they would be removed from the containing test suite.
    labelsBuilder.merge(new ResolvedTargets<>(toLabels(result), ImmutableSet.of(), hasError));

    for (Target suite : prerequisites) {
      if (TargetUtils.isTestSuiteRule(suite)) {
        TestExpansionValue value =
            (TestExpansionValue) env.getValue(TestExpansionValue.key(suite, strict));
        if (value == null) {
          continue;
        }
        labelsBuilder.merge(value.getLabels());
      }
    }

    return labelsBuilder.build();
  }

  /**
   * Adds the set of targets found in the attribute named {@code attrName}, which must be of label
   * or label list type, of the {@code test_suite} rule named {@code testSuite}. Returns true if the
   * method found a problem during the lookup process; the actual error message is reported to the
   * environment.
   */
  private static boolean getPrerequisites(
      Environment env, Rule rule, String attrName, List<Target> targets)
      throws InterruptedException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    List<Label> labels =
        mapper.visitLabels(mapper.getAttributeDefinition(attrName)).stream()
            .map(e -> e.getLabel())
            .collect(Collectors.toList());

    Set<PackageIdentifier> pkgIdentifiers = new LinkedHashSet<>();
    for (Label label : labels) {
      pkgIdentifiers.add(label.getPackageIdentifier());
    }

    Map<SkyKey, ValueOrException<NoSuchPackageException>> packages =
        env.getValuesOrThrow(PackageValue.keys(pkgIdentifiers), NoSuchPackageException.class);
    if (env.valuesMissing()) {
      return false;
    }
    boolean hasError = false;
    Map<PackageIdentifier, Package> packageMap = new HashMap<>();
    for (Map.Entry<SkyKey, ValueOrException<NoSuchPackageException>> entry : packages.entrySet()) {
      try {
        packageMap.put(
            (PackageIdentifier) entry.getKey().argument(),
            ((PackageValue) entry.getValue().get()).getPackage());
      } catch (NoSuchPackageException e) {
        env.getListener().handle(Event.error(e.getMessage()));
        hasError = true;
      }
    }

    for (Label label : labels) {
      Package pkg = packageMap.get(label.getPackageIdentifier());
      if (pkg == null) {
        continue;
      }
      if (pkg.containsErrors()) {
        hasError = true;
        // Abort the build if --nokeep_going.
        try {
          env.getValueOrThrow(
              PackageErrorFunction.key(label.getPackageIdentifier()),
              BuildFileContainsErrorsException.class);
          return false;
        } catch (BuildFileContainsErrorsException e) {
          // PackageErrorFunction always throws this exception, and this fact is used by Skyframe to
          // abort the build. If we get here, it's either because of error bubbling or because we're
          // in --keep_going mode. In either case, we *should* ignore the exception.
        }
      }
      try {
        targets.add(pkg.getTarget(label.getName()));
      } catch (NoSuchTargetException e) {
        env.getListener().handle(Event.error(e.getMessage()));
        hasError = true;
      }
    }
    return hasError;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
