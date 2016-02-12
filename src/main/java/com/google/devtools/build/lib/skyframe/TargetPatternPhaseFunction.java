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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.pkgcache.CompileOneDependencyTransformer;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseRunner;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.skyframe.EnvironmentBackedRecursivePackageProvider.MissingDepException;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue.TargetPatternList;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternSkyKeyOrException;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Takes a list of target patterns corresponding to a command line and turns it into a set of
 * resolved Targets.
 */
final class TargetPatternPhaseFunction implements SkyFunction {

  @Override
  public TargetPatternPhaseValue compute(SkyKey key, Environment env) {
    TargetPatternList options = (TargetPatternList) key.argument();

    // Determine targets to build:
    ResolvedTargets<Target> targets = getTargetsToBuild(env,
        options.getTargetPatterns(), options.getOffset(), options.getCompileOneDependency());

    // If the --build_tests_only option was specified or we want to run tests, we need to determine
    // the list of targets to test. For that, we remove manual tests and apply the command-line
    // filters. Also, if --build_tests_only is specified, then the list of filtered targets will be
    // set as build list as well.
    ResolvedTargets<Target> testTargets = null;
    if (options.getDetermineTests() || options.getBuildTestsOnly()) {
      testTargets = determineTests(env,
          options.getTargetPatterns(), options.getOffset(), options.getTestFilter());
      Preconditions.checkState(env.valuesMissing() || (testTargets != null));
    }

    Map<Label, SkyKey> testExpansionKeys = new LinkedHashMap<>();
    if (targets != null) {
      for (Target target : targets.getTargets()) {
        if (TargetUtils.isTestSuiteRule(target)) {
          Label label = target.getLabel();
          SkyKey testExpansionKey = TestSuiteExpansionValue.key(ImmutableSet.of(label));
          testExpansionKeys.put(label, testExpansionKey);
        }
      }
    }
    Map<SkyKey, SkyValue> expandedTests = env.getValues(testExpansionKeys.values());
    if (env.valuesMissing()) {
      return null;
    }

    ImmutableSet<Target> filteredTargets = targets.getFilteredTargets();
    ImmutableSet<Target> testsToRun = null;
    ImmutableSet<Target> testFilteredTargets = ImmutableSet.of();

    if (testTargets != null) {
      // Parse the targets to get the tests.
      if (testTargets.getTargets().isEmpty() && !testTargets.getFilteredTargets().isEmpty()) {
        env.getListener().handle(Event.warn("All specified test targets were excluded by filters"));
      }

      if (options.getBuildTestsOnly()) {
        // Replace original targets to build with test targets, so that only targets that are
        // actually going to be built are loaded in the loading phase. Note that this has a side
        // effect that any test_suite target requested to be built is replaced by the set of *_test
        // targets it represents; for example, this affects the status and the summary reports.
        Set<Target> allFilteredTargets = new HashSet<>();
        allFilteredTargets.addAll(targets.getTargets());
        allFilteredTargets.addAll(targets.getFilteredTargets());
        allFilteredTargets.removeAll(testTargets.getTargets());
        allFilteredTargets.addAll(testTargets.getFilteredTargets());
        testFilteredTargets = ImmutableSet.copyOf(allFilteredTargets);
        filteredTargets = ImmutableSet.of();

        targets = ResolvedTargets.<Target>builder()
            .merge(testTargets)
            .mergeError(targets.hasError())
            .build();
        if (options.getDetermineTests()) {
          testsToRun = testTargets.getTargets();
        }
      } else /*if (determineTests)*/ {
        testsToRun = testTargets.getTargets();
        targets = ResolvedTargets.<Target>builder()
            .merge(targets)
            // Avoid merge() here which would remove the filteredTargets from the targets.
            .addAll(testsToRun)
            .mergeError(testTargets.hasError())
            .build();
        // filteredTargets is correct in this case - it cannot contain tests that got back in
        // through test_suite expansion, because the test determination would also filter those out.
        // However, that's not obvious, and it might be better to explicitly recompute it.
      }
      if (testsToRun != null) {
        // Note that testsToRun can still be null here, if buildTestsOnly && !shouldRunTests.
        if (!targets.getTargets().containsAll(testsToRun)) {
          throw new IllegalStateException(String.format(
              "Internal consistency check failed; some targets are scheduled for test execution "
                  + "but not for building (%s)",
              Sets.difference(testsToRun, targets.getTargets())));
        }
      }
    }

    if (targets.hasError()) {
      env.getListener().handle(Event.warn("Target pattern parsing failed."));
    }

    LoadingPhaseRunner.maybeReportDeprecation(env.getListener(), targets.getTargets());

    boolean preExpansionError = targets.hasError();
    ResolvedTargets.Builder<Target> expandedTargetsBuilder = ResolvedTargets.builder();
    for (Target target : targets.getTargets()) {
      if (TargetUtils.isTestSuiteRule(target)) {
        SkyKey expansionKey =
            Preconditions.checkNotNull(testExpansionKeys.get(target.getLabel()));
        TestSuiteExpansionValue testExpansion =
            (TestSuiteExpansionValue) expandedTests.get(expansionKey);
        expandedTargetsBuilder.merge(testExpansion.getTargets());
      } else {
        expandedTargetsBuilder.add(target);
      }
    }
    ResolvedTargets<Target> expandedTargets = expandedTargetsBuilder.build();
    Set<Target> testSuiteTargets =
        Sets.difference(targets.getTargets(), expandedTargets.getTargets());
    return new TargetPatternPhaseValue(expandedTargets.getTargets(), testsToRun, preExpansionError,
        expandedTargets.hasError(), filteredTargets, testFilteredTargets,
        targets.getTargets(), ImmutableSet.copyOf(testSuiteTargets));
  }

  /**
   * Interpret the command-line arguments.
   *
   * @param targetPatterns the list of command-line target patterns specified by the user
   * @param compileOneDependency if true, enables alternative interpretation of targetPatterns; see
   *     {@link LoadingOptions#compileOneDependency}
   */
  private static ResolvedTargets<Target> getTargetsToBuild(Environment env,
      List<String> targetPatterns, String offset, boolean compileOneDependency) {
    List<SkyKey> patternSkyKeys = new ArrayList<>();
    for (TargetPatternSkyKeyOrException keyOrException :
        TargetPatternValue.keys(targetPatterns, FilteringPolicies.FILTER_MANUAL, offset)) {
      try {
        patternSkyKeys.add(keyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        // Skip.
      }
    }
    Map<SkyKey, ValueOrException<TargetParsingException>> resolvedPatterns =
        env.getValuesOrThrow(patternSkyKeys, TargetParsingException.class);
    if (env.valuesMissing()) {
      return null;
    }

    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    for (SkyKey key : patternSkyKeys) {
      TargetPatternKey pattern = (TargetPatternKey) key.argument();
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(key).get();
      } catch (TargetParsingException e) {
        // TODO(ulfjack): Report to EventBus.
        String rawPattern = pattern.getPattern();
        String errorMessage = e.getMessage();
        env.getListener().handle(Event.error("Skipping '" + rawPattern + "': " + errorMessage));
        builder.setError();
        continue;
      }
      // TODO(ulfjack): This is terribly inefficient.
      ResolvedTargets<Target> asTargets = TestSuiteExpansionFunction.labelsToTargets(
          env, value.getTargets().getTargets(), value.getTargets().hasError());
      if (pattern.isNegative()) {
        builder.filter(Predicates.not(Predicates.in(asTargets.getTargets())));
      } else {
        builder.merge(asTargets);
      }
    }

    ResolvedTargets<Target> result = builder.build();
    if (compileOneDependency) {
      TargetProvider targetProvider = new EnvironmentBackedRecursivePackageProvider(env);
      try {
        return new CompileOneDependencyTransformer(targetProvider)
            .transformCompileOneDependency(env.getListener(), result);
      } catch (MissingDepException e) {
        return null;
      } catch (TargetParsingException e) {
        env.getListener().handle(Event.error(e.getMessage()));
        return ResolvedTargets.failed();
      }
    }
    return result;
  }

  /**
   * Interpret test target labels from the command-line arguments and return the corresponding set
   * of targets, handling the filter flags, and expanding test suites.
   *
   * @param targetPatterns the list of command-line target patterns specified by the user
   * @param testFilter the test filter
   */
  private static ResolvedTargets<Target> determineTests(Environment env,
      List<String> targetPatterns, String offset, TestFilter testFilter) {
    List<SkyKey> patternSkyKeys = new ArrayList<>();
    for (TargetPatternSkyKeyOrException keyOrException :
        TargetPatternValue.keys(targetPatterns, FilteringPolicies.FILTER_TESTS, offset)) {
      try {
        patternSkyKeys.add(keyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        // Skip.
      }
    }
    Map<SkyKey, ValueOrException<TargetParsingException>> resolvedPatterns =
        env.getValuesOrThrow(patternSkyKeys, TargetParsingException.class);
    if (env.valuesMissing()) {
      return null;
    }

    List<SkyKey> expandedSuiteKeys = new ArrayList<>();
    for (SkyKey key : patternSkyKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(key).get();
      } catch (TargetParsingException e) {
        // Skip.
        continue;
      }
      expandedSuiteKeys.add(TestSuiteExpansionValue.key(value.getTargets().getTargets()));
    }
    Map<SkyKey, SkyValue> expandedSuites = env.getValues(expandedSuiteKeys);
    if (env.valuesMissing()) {
      return null;
    }

    ResolvedTargets.Builder<Target> testTargetsBuilder = ResolvedTargets.builder();
    for (SkyKey key : patternSkyKeys) {
      TargetPatternKey pattern = (TargetPatternKey) key.argument();
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(key).get();
      } catch (TargetParsingException e) {
        // This was already reported in getTargetsToBuild (maybe merge the two code paths?).
        continue;
      }

      TestSuiteExpansionValue expandedSuitesValue = (TestSuiteExpansionValue) expandedSuites.get(
          TestSuiteExpansionValue.key(value.getTargets().getTargets()));
      if (pattern.isNegative()) {
        ResolvedTargets<Target> negativeTargets = expandedSuitesValue.getTargets();
        testTargetsBuilder.filter(Predicates.not(Predicates.in(negativeTargets.getTargets())));
        testTargetsBuilder.mergeError(negativeTargets.hasError());
      } else {
        ResolvedTargets<Target> positiveTargets = expandedSuitesValue.getTargets();
        testTargetsBuilder.addAll(positiveTargets.getTargets());
        testTargetsBuilder.mergeError(positiveTargets.hasError());
      }
    }
    testTargetsBuilder.filter(testFilter);
    return testTargetsBuilder.build();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
