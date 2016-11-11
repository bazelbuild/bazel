// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.DelegatingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestTargetUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Implements the loading phase; responsible for:
 * <ul>
 *   <li>target pattern evaluation
 *   <li>test suite expansion
 *   <li>loading the labels needed to construct the build configuration
 *   <li>loading the labels needed for the analysis with the build configuration
 *   <li>loading the transitive closure of the targets and the configuration labels
 * </ul>
 *
 * <p>In order to ensure correctness of incremental loading and of full cache hits, this class is
 * very restrictive about access to its internal state and to its collaborators. In particular, none
 * of the collaborators of this class may change in incompatible ways, such as changing the relative
 * working directory for the target pattern parser, without notifying this class.
 *
 * <p>For full caching, this class tracks the exact values of all inputs to the loading phase. To
 * maximize caching, it is vital that these change as rarely as possible.
 *
 * <p>The Skyframe-based re-implementation of this class is in TargetPatternPhaseFunction.
 */
public final class LegacyLoadingPhaseRunner extends LoadingPhaseRunner {

  private static final class ParseFailureListenerImpl extends DelegatingEventHandler
      implements ParseFailureListener {
    private final EventBus eventBus;

    private ParseFailureListenerImpl(EventHandler delegate, EventBus eventBus) {
      super(delegate);
      this.eventBus = eventBus;
    }

    @Override
    public void parsingError(String targetPattern, String message) {
      if (eventBus != null) {
        eventBus.post(new ParsingFailedEvent(targetPattern, message));
      }
    }
  }

  private static final Logger LOG = Logger.getLogger(LoadingPhaseRunner.class.getName());

  private final PackageManager packageManager;
  private final TargetPatternEvaluator targetPatternEvaluator;
  private final Set<String> ruleNames;

  public LegacyLoadingPhaseRunner(PackageManager packageManager,
                            Set<String> ruleNames) {
    this.packageManager = packageManager;
    this.targetPatternEvaluator = packageManager.newTargetPatternEvaluator();
    this.ruleNames = ruleNames;
  }

  /**
   * Performs target pattern evaluation, test suite expansion (if requested), and loads the
   * transitive closure of the resulting targets as well as of the targets needed to use the given
   * build configuration provider.
   */
  @Override
  public LoadingResult execute(
      EventHandler eventHandler,
      EventBus eventBus,
      List<String> targetPatterns,
      PathFragment relativeWorkingDirectory,
      LoadingOptions options,
      ListMultimap<String, Label> labelsToLoadUnconditionally,
      boolean keepGoing,
      boolean determineTests,
      @Nullable LoadingCallback callback)
      throws TargetParsingException, LoadingFailedException, InterruptedException {
    LOG.info("Starting pattern evaluation");
    Stopwatch timer = Stopwatch.createStarted();
    if (options.buildTestsOnly && options.compileOneDependency) {
      throw new LoadingFailedException("--compile_one_dependency cannot be used together with "
          + "the --build_tests_only option or the 'bazel test' command ");
    }

    targetPatternEvaluator.updateOffset(relativeWorkingDirectory);
    EventHandler parseFailureListener = new ParseFailureListenerImpl(eventHandler, eventBus);
    // Determine targets to build:
    ResolvedTargets<Target> targets = getTargetsToBuild(parseFailureListener,
        targetPatterns, options.compileOneDependency, options.buildTagFilterList, keepGoing);

    ImmutableSet<Target> filteredTargets = targets.getFilteredTargets();

    boolean buildTestsOnly = options.buildTestsOnly;
    ImmutableSet<Target> testsToRun = null;
    ImmutableSet<Target> testFilteredTargets = ImmutableSet.of();

    // Now we have a list of targets to build. If the --build_tests_only option was specified or we
    // want to run tests, we need to determine the list of targets to test. For that, we remove
    // manual tests and apply the command line filters. Also, if --build_tests_only is specified,
    // then the list of filtered targets will be set as build list as well.
    if (determineTests || buildTestsOnly) {
      // Parse the targets to get the tests.
      ResolvedTargets<Target> testTargets = determineTests(parseFailureListener,
          targetPatterns, options, keepGoing);
      if (testTargets.getTargets().isEmpty() && !testTargets.getFilteredTargets().isEmpty()) {
        eventHandler.handle(Event.warn("All specified test targets were excluded by filters"));
      }

      if (buildTestsOnly) {
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
        if (determineTests) {
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
        Preconditions.checkState(targets.getTargets().containsAll(testsToRun));
      }
    }

    eventBus.post(
        new TargetParsingCompleteEvent(
            targets.getTargets(),
            filteredTargets,
            testFilteredTargets,
            timer.stop().elapsed(TimeUnit.MILLISECONDS),
            targetPatterns));

    if (targets.hasError()) {
      eventHandler.handle(Event.warn("Target pattern parsing failed. Continuing anyway"));
    }
    if (callback != null) {
      callback.notifyTargets(targets.getTargets());
    }
    LoadingPhaseRunner.maybeReportDeprecation(eventHandler, targets.getTargets());

    return doSimpleLoadingPhase(eventHandler, eventBus, targets, testsToRun, keepGoing);
  }

  /**
   * Perform test_suite expansion and emits necessary events and logging messages for legacy
   * support.
   */
  private LoadingResult doSimpleLoadingPhase(
      EventHandler eventHandler,
      EventBus eventBus,
      ResolvedTargets<Target> targets,
      ImmutableSet<Target> testsToRun,
      boolean keepGoing)
      throws InterruptedException, LoadingFailedException {
    Stopwatch timer = preLoadingLogging(eventHandler);

    ImmutableSet<Target> targetsToLoad = targets.getTargets();
    ResolvedTargets<Target> expandedResult;
    try {
      expandedResult = expandTestSuites(eventHandler, targetsToLoad, keepGoing);
    } catch (TargetParsingException e) {
      throw new LoadingFailedException("Loading failed; build aborted", e);
    }

    postLoadingLogging(eventBus, targetsToLoad, expandedResult.getTargets(), timer);
    return new LoadingResult(targets.hasError(), expandedResult.hasError(),
        expandedResult.getTargets(), testsToRun, getWorkspaceName(eventHandler));
  }

  private Stopwatch preLoadingLogging(EventHandler eventHandler) {
    eventHandler.handle(Event.progress("Loading..."));
    LOG.info("Starting loading phase");
    return Stopwatch.createStarted();
  }

  private void postLoadingLogging(EventBus eventBus, ImmutableSet<Target> originalTargetsToLoad,
      ImmutableSet<Target> expandedTargetsToLoad, Stopwatch timer) {
    Set<Target> testSuiteTargets = Sets.difference(originalTargetsToLoad, expandedTargetsToLoad);
    eventBus.post(new LoadingPhaseCompleteEvent(
        expandedTargetsToLoad, ImmutableSet.copyOf(testSuiteTargets),
        packageManager.getStatistics(), timer.stop().elapsed(TimeUnit.MILLISECONDS)));
    LOG.info("Loading phase finished"); 
  }

  private ResolvedTargets<Target> expandTestSuites(EventHandler eventHandler,
      ImmutableSet<Target> targets, boolean keepGoing)
      throws LoadingFailedException, TargetParsingException {
    // We use strict test_suite expansion here to match the analysis-time checks.
    ResolvedTargets<Target> expandedResult = TestTargetUtils.expandTestSuites(
        packageManager, eventHandler, targets, /*strict=*/true, /*keepGoing=*/true);
    if (expandedResult.hasError() && !keepGoing) {
      throw new LoadingFailedException("Could not expand test suite target");
    }
    return expandedResult;
  }

  /**
   * Interpret the command-line arguments.
   *
   * @param targetPatterns the list of command-line target patterns specified by the user
   * @param compileOneDependency if true, enables alternative interpretation of targetPatterns; see
   *     {@link LoadingOptions#compileOneDependency}
   * @throws TargetParsingException if parsing failed and !keepGoing
   */
  private ResolvedTargets<Target> getTargetsToBuild(EventHandler eventHandler,
      List<String> targetPatterns, boolean compileOneDependency,
      List<String> buildTagFilterList, boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    ResolvedTargets<Target> evaluated =
        targetPatternEvaluator.parseTargetPatternList(eventHandler, targetPatterns,
            FilteringPolicies.FILTER_MANUAL, keepGoing);

    ResolvedTargets<Target> result = ResolvedTargets.<Target>builder()
        .merge(evaluated)
        .filter(TargetUtils.tagFilter(buildTagFilterList))
        .build();

    if (compileOneDependency) {
      return new CompileOneDependencyTransformer(packageManager)
          .transformCompileOneDependency(eventHandler, result);
    }

    return result;
  }

  /**
   * Interpret test target labels from the command-line arguments and return the corresponding set
   * of targets, handling the filter flags, and expanding test suites.
   *
   * @param eventHandler the error event eventHandler
   * @param targetPatterns the list of command-line target patterns specified by the user
   * @param options the loading phase options
   * @param keepGoing value of the --keep_going flag
   */
  private ResolvedTargets<Target> determineTests(EventHandler eventHandler,
      List<String> targetPatterns, LoadingOptions options, boolean keepGoing)
          throws TargetParsingException, InterruptedException {
    // Parse the targets to get the tests.
    ResolvedTargets<Target> testTargetsBuilder = targetPatternEvaluator.parseTargetPatternList(
        eventHandler, targetPatterns, FilteringPolicies.FILTER_TESTS, keepGoing);

    ResolvedTargets.Builder<Target> finalBuilder = ResolvedTargets.builder();
    finalBuilder.merge(testTargetsBuilder);
    finalBuilder.filter(TestFilter.forOptions(options, eventHandler, ruleNames));
    return finalBuilder.build();
  }

  private String getWorkspaceName(EventHandler eventHandler)
      throws InterruptedException, LoadingFailedException {
    try {
      return packageManager.getPackage(eventHandler, Label.EXTERNAL_PACKAGE_IDENTIFIER)
          .getWorkspaceName();
    } catch (NoSuchPackageException e) {
      throw new LoadingFailedException("Failed to load //external package", e);
    }
  }
}
