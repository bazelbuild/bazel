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

import static com.google.common.collect.ImmutableSetMultimap.flatteningToImmutableSetMultimap;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.AbstractRecursivePackageProvider.MissingDepException;
import com.google.devtools.build.lib.pkgcache.CompileOneDependencyTransformer;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.pkgcache.ParsingFailedEvent;
import com.google.devtools.build.lib.pkgcache.TargetParsingCompleteEvent;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue.TargetPatternPhaseKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternSkyKeyOrException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.ArrayList;
import java.util.Collection;
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

  private final ExternalPackageHelper externalPackageHelper;

  public TargetPatternPhaseFunction(ExternalPackageHelper externalPackageHelper) {
    this.externalPackageHelper = externalPackageHelper;
  }

  @Override
  public TargetPatternPhaseValue compute(SkyKey key, Environment env) throws InterruptedException {
    TargetPatternPhaseKey options = (TargetPatternPhaseKey) key.argument();
    WorkspaceNameValue workspaceName = (WorkspaceNameValue) env.getValue(WorkspaceNameValue.key());
    ImmutableSortedSet<String> notSymlinkedInExecrootDirectories =
        externalPackageHelper.getNotSymlinkedInExecrootDirectories(env);
    if (env.valuesMissing()) {
      return null;
    }

    RepositoryMappingValue repositoryMappingValue =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (repositoryMappingValue == null) {
      return null;
    }

    // Determine targets to build:
    List<String> failedPatterns = new ArrayList<String>();
    List<ExpandedPattern> expandedPatterns =
        getTargetsToBuild(
            env, options, repositoryMappingValue.getRepositoryMapping(), failedPatterns);
    ResolvedTargets<Target> targets =
        env.valuesMissing()
            ? null
            : mergeAll(expandedPatterns, !failedPatterns.isEmpty(), env, options);

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
        if (TargetUtils.isTestSuiteRule(target) && options.isExpandTestSuites()) {
          Label label = target.getLabel();
          SkyKey testExpansionKey = TestsForTargetPatternValue.key(ImmutableSet.of(label));
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
        targets =
            ResolvedTargets.<Target>builder()
                .merge(targets)
                // Merging in all testsToRun guarantees that targets that will be built (because
                // they are tests) are not considered to be "filtered out", even if they were
                // initially filtered out. We can't merge in testTargets because its set of
                // filteredTargets could include targets that we're building but not testing.
                .merge(ResolvedTargets.<Target>builder().addAll(testsToRun).build())
                .mergeError(testTargets.hasError())
                .build();
        filteredTargets = targets.getFilteredTargets();
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

    maybeReportDeprecation(env.getListener(), targets.getTargets());

    ResolvedTargets.Builder<Label> expandedLabelsBuilder = ResolvedTargets.builder();
    for (Target target : targets.getTargets()) {
      if (TargetUtils.isTestSuiteRule(target) && options.isExpandTestSuites()) {
        SkyKey expansionKey =
            Preconditions.checkNotNull(testExpansionKeys.get(target.getLabel()));
        TestsForTargetPatternValue testExpansion =
            (TestsForTargetPatternValue) expandedTests.get(expansionKey);
        expandedLabelsBuilder.merge(testExpansion.getLabels());
      } else {
        expandedLabelsBuilder.add(target.getLabel());
      }
    }
    ResolvedTargets<Label> targetLabels = expandedLabelsBuilder.build();
    ResolvedTargets<Target> expandedTargets =
        TestsForTargetPatternFunction.labelsToTargets(
            env, targetLabels.getTargets(), targetLabels.hasError());
    Set<Target> testSuiteTargets =
        Sets.difference(targets.getTargets(), expandedTargets.getTargets());
    ImmutableSet<Label> testsToRunLabels = null;
    if (testsToRun != null) {
      testsToRunLabels =
          testsToRun.stream().map(Target::getLabel).collect(ImmutableSet.toImmutableSet());
    }
    ImmutableSet<Label> removedTargetLabels =
        testSuiteTargets.stream().map(Target::getLabel).collect(ImmutableSet.toImmutableSet());

    TargetPatternPhaseValue result =
        new TargetPatternPhaseValue(
            targetLabels.getTargets(),
            testsToRunLabels,
            targets.hasError(),
            expandedTargets.hasError(),
            workspaceName.getName(),
            notSymlinkedInExecrootDirectories);

    env.getListener()
        .post(
            new TargetParsingCompleteEvent(
                targets.getTargets(),
                filteredTargets,
                testFilteredTargets,
                options.getTargetPatterns(),
                expandedTargets.getTargets(),
                ImmutableList.copyOf(failedPatterns),
                mapOriginalPatternsToLabels(expandedPatterns, targets.getTargets())));
    env.getListener()
        .post(new LoadingPhaseCompleteEvent(result.getTargetLabels(), removedTargetLabels));
    return result;
  }

  /**
   * Emit a warning when a deprecated target is mentioned on the command line.
   *
   * <p>Note that this does not stop us from emitting "target X depends on deprecated target Y"
   * style warnings for the same target and it is a good thing; <i>depending</i> on a target and
   * <i>wanting</i> to build it are different things.
   */
  private static void maybeReportDeprecation(
      ExtendedEventHandler eventHandler, Collection<Target> targets) {
    for (Rule rule : Iterables.filter(targets, Rule.class)) {
      if (rule.isAttributeValueExplicitlySpecified("deprecation")) {
        eventHandler.handle(Event.warn(rule.getLocation(), String.format(
            "target '%s' is deprecated: %s", rule.getLabel(),
            NonconfigurableAttributeMapper.of(rule).get("deprecation", Type.STRING))));
      }
    }
  }

  /**
   * Interprets the command-line arguments by expanding each pattern to targets and populating the
   * list of {@code failedPatterns}.
   *
   * @param env the Starlark environment
   * @param options the command-line arguments in structured form
   * @param failedPatterns a list into which failed patterns are added
   */
  private static List<ExpandedPattern> getTargetsToBuild(
      Environment env,
      TargetPatternPhaseKey options,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping,
      List<String> failedPatterns)
      throws InterruptedException {

    ImmutableList.Builder<String> canonicalPatterns = new ImmutableList.Builder<>();
    for (String rawPattern : options.getTargetPatterns()) {
      canonicalPatterns.add(TargetPattern.renameRepository(rawPattern, repoMapping));
    }

    List<TargetPatternKey> patternSkyKeys = new ArrayList<>(options.getTargetPatterns().size());
    for (TargetPatternSkyKeyOrException keyOrException :
        TargetPatternValue.keys(
            canonicalPatterns.build(),
            options.getBuildManualTests()
                ? FilteringPolicies.NO_FILTER
                : FilteringPolicies.FILTER_MANUAL,
            options.getOffset())) {
      try {
        patternSkyKeys.add(keyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        failedPatterns.add(keyOrException.getOriginalPattern());
        // We post a PatternExpandingError here - the pattern could not be parsed, so we don't even
        // get to run TargetPatternFunction.
        env.getListener().post(
            PatternExpandingError.failed(keyOrException.getOriginalPattern(), e.getMessage()));
        // We generally skip patterns that don't parse. We report a parsing failed exception to the
        // event bus here, but not in determineTests below, which goes through the same list. Note
        // that the TargetPatternFunction otherwise reports these events (but only if the target
        // pattern could be parsed successfully).
        env.getListener().post(
            new ParsingFailedEvent(keyOrException.getOriginalPattern(), e.getMessage()));
        try {
          env.getValueOrThrow(
              TargetPatternErrorFunction.key(e.getMessage()), TargetParsingException.class);
        } catch (TargetParsingException ignore) {
          // We ignore this. Keep going is active.
        }
        env.getListener().handle(
            Event.error(
                "Skipping '" + keyOrException.getOriginalPattern() + "': " + e.getMessage()));
      }
    }

    Map<SkyKey, ValueOrException<TargetParsingException>> resolvedPatterns =
        env.getValuesOrThrow(patternSkyKeys, TargetParsingException.class);
    List<ExpandedPattern> expandedPatterns = new ArrayList<>(patternSkyKeys.size());

    for (TargetPatternKey pattern : patternSkyKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(pattern).get();
      } catch (TargetParsingException e) {
        String rawPattern = pattern.getPattern();
        String errorMessage = e.getMessage();
        failedPatterns.add(rawPattern);
        env.getListener().post(PatternExpandingError.failed(rawPattern, errorMessage));
        env.getListener().handle(Event.error("Skipping '" + rawPattern + "': " + errorMessage));
        continue;
      }
      if (value == null) {
        continue;
      }
      // TODO(ulfjack): This is terribly inefficient.
      ResolvedTargets<Target> asTargets =
          TestsForTargetPatternFunction.labelsToTargets(
              env, value.getTargets().getTargets(), value.getTargets().hasError());
      if (asTargets == null) {
        continue;
      }
      expandedPatterns.add(ExpandedPattern.of(pattern, asTargets));
    }

    return expandedPatterns;
  }

  /** Merges expansions from all patterns into a single {@link ResolvedTargets} instance. */
  private static ResolvedTargets<Target> mergeAll(
      List<ExpandedPattern> expandedPatterns,
      boolean hasError,
      Environment env,
      TargetPatternPhaseKey options)
      throws InterruptedException {
    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    builder.mergeError(hasError);

    for (ExpandedPattern expansion : expandedPatterns) {
      if (expansion.pattern().isNegative()) {
        builder.filter(Predicates.not(Predicates.in(expansion.resolvedTargets().getTargets())));
      } else {
        builder.merge(expansion.resolvedTargets());
      }
    }

    ResolvedTargets<Target> result = builder
        .filter(TargetUtils.tagFilter(options.getBuildTargetFilter()))
        .build();
    if (options.getCompileOneDependency()) {
      EnvironmentBackedRecursivePackageProvider environmentBackedRecursivePackageProvider =
          new EnvironmentBackedRecursivePackageProvider(env);
      try {
        result =
            new CompileOneDependencyTransformer(environmentBackedRecursivePackageProvider)
                .transformCompileOneDependency(env.getListener(), result);
      } catch (MissingDepException e) {
        return null;
      } catch (TargetParsingException e) {
        try {
          env.getValueOrThrow(
              TargetPatternErrorFunction.key(e.getMessage()), TargetParsingException.class);
        } catch (TargetParsingException ignore) {
          // We ignore this. Keep going is active.
        }
        env.getListener().handle(Event.error(e.getMessage()));
        return ResolvedTargets.failed();
      }
      if (environmentBackedRecursivePackageProvider.encounteredPackageErrors()) {
        result = ResolvedTargets.<Target>builder().merge(result).setError().build();
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
  private static ResolvedTargets<Target> determineTests(
      Environment env, List<String> targetPatterns, String offset, TestFilter testFilter)
      throws InterruptedException {
    List<TargetPatternKey> patternSkyKeys = new ArrayList<>();
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
    for (TargetPatternKey key : patternSkyKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(key).get();
      } catch (TargetParsingException e) {
        // Skip.
        continue;
      }
      expandedSuiteKeys.add(TestsForTargetPatternValue.key(value.getTargets().getTargets()));
    }
    Map<SkyKey, SkyValue> expandedSuites = env.getValues(expandedSuiteKeys);
    if (env.valuesMissing()) {
      return null;
    }

    ResolvedTargets.Builder<Target> testTargetsBuilder = ResolvedTargets.builder();
    for (TargetPatternKey pattern : patternSkyKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(pattern).get();
      } catch (TargetParsingException e) {
        // This was already reported in getTargetsToBuild (maybe merge the two code paths?).
        continue;
      }

      TestsForTargetPatternValue expandedSuitesValue =
          (TestsForTargetPatternValue)
              expandedSuites.get(TestsForTargetPatternValue.key(value.getTargets().getTargets()));
      if (pattern.isNegative()) {
        ResolvedTargets<Target> negativeTargets =
            TestsForTargetPatternFunction.labelsToTargets(
                env,
                expandedSuitesValue.getLabels().getTargets(),
                expandedSuitesValue.getLabels().hasError());
        testTargetsBuilder.filter(Predicates.not(Predicates.in(negativeTargets.getTargets())));
        testTargetsBuilder.mergeError(negativeTargets.hasError());
      } else {
        ResolvedTargets<Target> positiveTargets =
            TestsForTargetPatternFunction.labelsToTargets(
                env,
                expandedSuitesValue.getLabels().getTargets(),
                expandedSuitesValue.getLabels().hasError());
        testTargetsBuilder.addAll(positiveTargets.getTargets());
        testTargetsBuilder.mergeError(positiveTargets.hasError());
      }
    }

    testTargetsBuilder.filter(testFilter);
    return testTargetsBuilder.build();
  }

  private static ImmutableSetMultimap<String, Label> mapOriginalPatternsToLabels(
      List<ExpandedPattern> expandedPatterns, Set<Target> includedTargets) {
    return expandedPatterns
        .stream()
        .filter(expansion -> !expansion.pattern().isNegative())
        .collect(
            flatteningToImmutableSetMultimap(
                expansion -> expansion.pattern().getPattern(),
                expansion ->
                    expansion
                        .resolvedTargets()
                        .getTargets()
                        .stream()
                        .filter(includedTargets::contains)
                        .map(Target::getLabel)));
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** Represents the expansion of a single target pattern. */
  @AutoValue
  abstract static class ExpandedPattern {

    static ExpandedPattern of(TargetPatternKey pattern, ResolvedTargets<Target> resolvedTargets) {
      return new AutoValue_TargetPatternPhaseFunction_ExpandedPattern(pattern, resolvedTargets);
    }

    abstract TargetPatternKey pattern();
    abstract ResolvedTargets<Target> resolvedTargets();
  }
}
