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
package com.google.devtools.build.lib.buildtool;

import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.buildeventstream.AbortedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildtool.buildevent.NoAnalyzeEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/** Performs target pattern eval, configuration creation, loading and analysis. */
public final class AnalysisPhaseRunner {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private AnalysisPhaseRunner() {}

  public static AnalysisResult execute(
      CommandEnvironment env,
      BuildRequest request,
      BuildOptions buildOptions,
      TargetValidator validator)
      throws BuildFailedException, InterruptedException, ViewCreationFailedException,
          TargetParsingException, LoadingFailedException, AbruptExitException,
          InvalidConfigurationException {

    // Target pattern evaluation.
    TargetPatternPhaseValue loadingResult;
    Profiler.instance().markPhase(ProfilePhase.TARGET_PATTERN_EVAL);
    try (SilentCloseable c = Profiler.instance().profile("evaluateTargetPatterns")) {
      loadingResult = evaluateTargetPatterns(env, request, validator);
    }
    env.setWorkspaceName(loadingResult.getWorkspaceName());

    // Compute the heuristic instrumentation filter if needed.
    if (request.needsInstrumentationFilter()) {
      try (SilentCloseable c = Profiler.instance().profile("Compute instrumentation filter")) {
        String instrumentationFilter =
            InstrumentationFilterSupport.computeInstrumentationFilter(
                env.getReporter(),
                // TODO(ulfjack): Expensive. Make this part of the TargetPatternPhaseValue or write
                // a new SkyFunction to compute it?
                loadingResult.getTestsToRun(env.getReporter(), env.getPackageManager()));
        try {
          // We're modifying the buildOptions in place, which is not ideal, but we also don't want
          // to pay the price for making a copy. Maybe reconsider later if this turns out to be a
          // problem (and the performance loss may not be a big deal).
          buildOptions.get(CoreOptions.class).instrumentationFilter =
              new RegexFilter.RegexFilterConverter().convert(instrumentationFilter);
        } catch (OptionsParsingException e) {
          throw new InvalidConfigurationException(e);
        }
      }
    }

    // Exit if there are any pending exceptions from modules.
    env.throwPendingException();

    AnalysisResult analysisResult = null;
    if (request.getBuildOptions().performAnalysisPhase) {
      Profiler.instance().markPhase(ProfilePhase.ANALYZE);

      // The build info factories are immutable during the life time of this server. However, we
      // sometimes clean the graph, which requires re-injecting the value, which requires a hook to
      // do so afterwards, and there is no such hook at the server / workspace level right now. For
      // simplicity, we keep the code here for now.
      env.getSkyframeExecutor()
          .injectExtraPrecomputedValues(
              ImmutableList.of(
                  PrecomputedValue.injected(
                      BuildInfoCollectionFunction.BUILD_INFO_FACTORIES,
                      env.getRuntime().getRuleClassProvider().getBuildInfoFactoriesAsMap())));

      try (SilentCloseable c = Profiler.instance().profile("runAnalysisPhase")) {
        analysisResult =
            runAnalysisPhase(env, request, loadingResult, buildOptions, request.getMultiCpus());
      }

      for (BlazeModule module : env.getRuntime().getBlazeModules()) {
        module.afterAnalysis(env, request, buildOptions, analysisResult.getTargetsToBuild());
      }

      reportTargets(env, analysisResult);

      for (ConfiguredTarget target : analysisResult.getTargetsToSkip()) {
        BuildConfiguration config =
            env.getSkyframeExecutor()
                .getConfiguration(env.getReporter(), target.getConfigurationKey());
        Label label = target.getLabel();
        env.getEventBus()
            .post(
                new AbortedEvent(
                    BuildEventIdUtil.targetCompleted(label, config.getEventId()),
                    AbortReason.SKIPPED,
                    String.format("Target %s build was skipped.", label),
                    label));
      }
    } else {
      env.getReporter().handle(Event.progress("Loading complete."));
      env.getReporter().post(new NoAnalyzeEvent());
      logger.atInfo().log("No analysis requested, so finished");
      FailureDetail failureDetail = BuildView.createFailureDetail(loadingResult, null, null);
      if (failureDetail != null) {
        throw new BuildFailedException(
            failureDetail.getMessage(), DetailedExitCode.of(failureDetail));
      }
    }

    return analysisResult;
  }

  private static TargetPatternPhaseValue evaluateTargetPatterns(
      CommandEnvironment env, final BuildRequest request, final TargetValidator validator)
      throws LoadingFailedException, TargetParsingException, InterruptedException {
    boolean keepGoing = request.getKeepGoing();
    TargetPatternPhaseValue result =
        env.getSkyframeExecutor()
            .loadTargetPatternsWithFilters(
                env.getReporter(),
                request.getTargets(),
                env.getRelativeWorkingDirectory(),
                request.getLoadingOptions(),
                request.getLoadingPhaseThreadCount(),
                keepGoing,
                request.shouldRunTests());
    if (validator != null) {
      Collection<Target> targets =
          result.getTargets(env.getReporter(), env.getSkyframeExecutor().getPackageManager());
      validator.validateTargets(targets, keepGoing);
    }
    return result;
  }

  /**
   * Performs the initial phases 0-2 of the build: Setup, Loading and Analysis.
   *
   * <p>Postcondition: On success, populates the BuildRequest's set of targets to build.
   *
   * @return null if loading / analysis phases were successful; a useful error message if loading or
   *     analysis phase errors were encountered and request.keepGoing.
   * @throws InterruptedException if the current thread was interrupted.
   * @throws ViewCreationFailedException if analysis failed for any reason.
   */
  private static AnalysisResult runAnalysisPhase(
      CommandEnvironment env,
      BuildRequest request,
      TargetPatternPhaseValue loadingResult,
      BuildOptions targetOptions,
      Set<String> multiCpu)
      throws InterruptedException, InvalidConfigurationException, ViewCreationFailedException {
    Stopwatch timer = Stopwatch.createStarted();
    env.getReporter().handle(Event.progress("Loading complete.  Analyzing..."));

    BuildView view =
        new BuildView(
            env.getDirectories(),
            env.getRuntime().getRuleClassProvider(),
            env.getSkyframeExecutor(),
            env.getRuntime().getCoverageReportActionFactory(request));
    AnalysisResult analysisResult =
        view.update(
            loadingResult,
            targetOptions,
            multiCpu,
            request.getAspects(),
            request.getViewOptions(),
            request.getKeepGoing(),
            request.getLoadingPhaseThreadCount(),
            request.getTopLevelArtifactContext(),
            env.getReporter(),
            env.getEventBus());

    Pair<Integer, Integer> tcal = view.getTargetsConfiguredAndLoaded();

    // TODO(bazel-team): Merge these into one event.
    env.getEventBus()
        .post(
            new AnalysisPhaseCompleteEvent(
                analysisResult.getTargetsToBuild(),
                /* targetsLoaded */ tcal.second.intValue(),
                /* targetsConfigured */ tcal.first.intValue(),
                timer.stop().elapsed(TimeUnit.MILLISECONDS),
                view.getAndClearPkgManagerStatistics(),
                view.getActionsConstructed(),
                env.getSkyframeExecutor().wasAnalysisCacheDiscardedAndResetBit()));
    ImmutableSet<BuildConfigurationValue.Key> configurationKeys =
        Stream.concat(
                analysisResult
                    .getTargetsToBuild()
                    .stream()
                    .map(ConfiguredTarget::getConfigurationKey)
                    .distinct(),
                analysisResult.getTargetsToTest() == null
                    ? Stream.empty()
                    : analysisResult
                        .getTargetsToTest()
                        .stream()
                        .map(ConfiguredTarget::getConfigurationKey)
                        .distinct())
            .filter(Objects::nonNull)
            .distinct()
            .collect(ImmutableSet.toImmutableSet());
    Map<BuildConfigurationValue.Key, BuildConfiguration> configurationMap =
        env.getSkyframeExecutor().getConfigurations(env.getReporter(), configurationKeys);
    env.getEventBus()
        .post(
            new TestFilteringCompleteEvent(
                analysisResult.getTargetsToBuild(),
                analysisResult.getTargetsToTest(),
                configurationMap));
    return analysisResult;
  }

  private static void reportTargets(CommandEnvironment env, AnalysisResult analysisResult) {
    Collection<ConfiguredTarget> targetsToBuild = analysisResult.getTargetsToBuild();
    Collection<ConfiguredTarget> targetsToTest = analysisResult.getTargetsToTest();
    if (targetsToTest != null) {
      int testCount = targetsToTest.size();
      int targetCount = targetsToBuild.size() - testCount;
      if (targetCount == 0) {
        env.getReporter()
            .handle(
                Event.info(
                    "Found "
                        + testCount
                        + (testCount == 1 ? " test target..." : " test targets...")));
      } else {
        env.getReporter()
            .handle(
                Event.info(
                    "Found "
                        + targetCount
                        + (targetCount == 1 ? " target and " : " targets and ")
                        + testCount
                        + (testCount == 1 ? " test target..." : " test targets...")));
      }
    } else {
      int targetCount = targetsToBuild.size();
      env.getReporter()
          .handle(
              Event.info(
                  "Found " + targetCount + (targetCount == 1 ? " target..." : " targets...")));
    }
  }
}
