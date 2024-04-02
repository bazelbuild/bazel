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

import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.configurationId;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.buildeventstream.AbortedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildtool.buildevent.NoAnalyzeEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Parser;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.AspectAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TestAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetSkippedEvent;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
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
          InvalidConfigurationException, RepositoryMappingResolutionException {

    // Target pattern evaluation.
    TargetPatternPhaseValue loadingResult;
    Profiler.instance().markPhase(ProfilePhase.TARGET_PATTERN_EVAL);
    try (SilentCloseable c = Profiler.instance().profile("evaluateTargetPatterns")) {
      loadingResult = evaluateTargetPatterns(env, request, validator);
    }
    env.setWorkspaceName(loadingResult.getWorkspaceName());

    // TODO: b/324127375 - Use with incoming --scl_config flag.
    PathFragment unusedProjectFile =
        BuildTool.getProjectFile(
            loadingResult.getTargetLabels(), env.getSkyframeExecutor(), env.getReporter());

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
          throw new InvalidConfigurationException(Code.HEURISTIC_INSTRUMENTATION_FILTER_INVALID, e);
        }
      }
    }

    // Exit if there are any pending exceptions from modules.
    env.throwPendingException();

    AnalysisResult analysisResult = null;
    if (request.getBuildOptions().performAnalysisPhase) {
      Profiler.instance().markPhase(ProfilePhase.ANALYZE);

      try (SilentCloseable c = Profiler.instance().profile("runAnalysisPhase")) {
        analysisResult = runAnalysisPhase(env, request, loadingResult, buildOptions);
      }

      for (BlazeModule module : env.getRuntime().getBlazeModules()) {
        module.afterAnalysis(env, request, buildOptions, analysisResult);
      }

      if (request.shouldRunTests()) {
        reportTargetsWithTests(
            env,
            analysisResult.getTargetsToBuild(),
            Preconditions.checkNotNull(analysisResult.getTargetsToTest()));
      } else {
        reportTargets(env, analysisResult.getTargetsToBuild());
      }

      postAbortedEventsForSkippedTargets(env, analysisResult.getTargetsToSkip());
    } else {
      env.getReporter().handle(Event.progress("Loading complete."));
      env.getReporter().post(new NoAnalyzeEvent());
      logger.atInfo().log("No analysis requested, so finished");
      FailureDetail failureDetail =
          BuildView.createAnalysisFailureDetail(loadingResult, /* skyframeAnalysisResult= */ null);
      if (failureDetail != null) {
        throw new BuildFailedException(
            failureDetail.getMessage(), DetailedExitCode.of(failureDetail));
      }
    }

    return analysisResult;
  }

  static void postAbortedEventsForSkippedTargets(
      CommandEnvironment env, ImmutableSet<ConfiguredTarget> targetsToSkip) {
    for (ConfiguredTarget target : targetsToSkip) {
      BuildConfigurationValue config =
          env.getSkyframeExecutor()
              .getConfiguration(env.getReporter(), target.getConfigurationKey());
      Label label = target.getLabel();
      env.getEventBus()
          .post(
              new AbortedEvent(
                  BuildEventIdUtil.targetCompleted(label, configurationId(config)),
                  AbortReason.SKIPPED,
                  String.format("Target %s build was skipped.", label),
                  label));
    }
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
      BuildOptions targetOptions)
      throws InterruptedException, InvalidConfigurationException,
          RepositoryMappingResolutionException, ViewCreationFailedException {
    Stopwatch timer = Stopwatch.createStarted();
    env.getReporter().handle(Event.progress("Loading complete.  Analyzing..."));

    ImmutableSet<Label> explicitTargetPatterns =
        getExplicitTargetPatterns(
            env,
            request.getTargets(),
            request.getKeepGoing(),
            request.getLoadingPhaseThreadCount());

    BuildView view =
        new BuildView(
            env.getDirectories(),
            env.getRuntime().getRuleClassProvider(),
            env.getSkyframeExecutor(),
            env.getRuntime().getCoverageReportActionFactory(request));
    AnalysisResult analysisResult;
    try {
      analysisResult =
          view.update(
              loadingResult,
              targetOptions,
              explicitTargetPatterns,
              request.getAspects(),
              request.getAspectsParameters(),
              request.getViewOptions(),
              request.getKeepGoing(),
              request.getViewOptions().skipIncompatibleExplicitTargets,
              request.getCheckForActionConflicts(),
              env.getQuiescingExecutors(),
              request.getTopLevelArtifactContext(),
              request.reportIncompatibleTargets(),
              env.getReporter(),
              env.getEventBus(),
              env.getRuntime().getBugReporter(),
              /* includeExecutionPhase= */ false,
              /* skymeldAnalysisOverlapPercentage= */ 0,
              /* resourceManager= */ null,
              /* buildResultListener= */ null,
              /* executionSetupCallback= */ null,
              /* buildConfigurationsCreatedCallback= */ null,
              /* buildDriverKeyTestContext= */ null,
              env.getAdditionalConfigurationChangeEvent());
    } catch (BuildFailedException | TestExecException | AbruptExitException unexpected) {
      throw new IllegalStateException("Unexpected execution exception type: ", unexpected);
    }

    // TODO(bazel-team): Merge these into one event.
    env.getEventBus()
        .post(
            new AnalysisPhaseCompleteEvent(
                analysisResult.getTargetsToBuild(),
                view.getEvaluatedCounts(),
                view.getEvaluatedActionsCounts(),
                timer.stop().elapsed(TimeUnit.MILLISECONDS),
                view.getAndClearPkgManagerStatistics(),
                env.getSkyframeExecutor().wasAnalysisCacheInvalidatedAndResetBit()));
    ImmutableSet<BuildConfigurationKey> configurationKeys =
        Stream.concat(
                analysisResult.getTargetsToBuild().stream()
                    .map(ConfiguredTarget::getConfigurationKey)
                    .distinct(),
                analysisResult.getTargetsToTest() == null
                    ? Stream.empty()
                    : analysisResult.getTargetsToTest().stream()
                        .map(ConfiguredTarget::getConfigurationKey)
                        .distinct())
            .filter(Objects::nonNull)
            .distinct()
            .collect(ImmutableSet.toImmutableSet());
    Map<BuildConfigurationKey, BuildConfigurationValue> configurationMap =
        env.getSkyframeExecutor().getConfigurations(env.getReporter(), configurationKeys);
    env.getEventBus()
        .post(
            new TestFilteringCompleteEvent(
                analysisResult.getTargetsToBuild(),
                analysisResult.getTargetsToTest(),
                analysisResult.getTargetsToSkip(),
                configurationMap));
    postTopLevelStatusEvents(env, analysisResult, configurationMap);

    return analysisResult;
  }

  /** Post the appropriate {@link com.google.devtools.build.lib.skyframe.TopLevelStatusEvents}. */
  private static void postTopLevelStatusEvents(
      CommandEnvironment env,
      AnalysisResult analysisResult,
      Map<BuildConfigurationKey, BuildConfigurationValue> configurationMap) {
    for (ConfiguredTarget configuredTarget : analysisResult.getTargetsToBuild()) {
      env.getEventBus().post(TopLevelTargetAnalyzedEvent.create(configuredTarget));
      if (analysisResult.getTargetsToSkip().contains(configuredTarget)) {
        env.getEventBus().post(TopLevelTargetSkippedEvent.create(configuredTarget));
      }

      if (analysisResult.getTargetsToTest() != null
          && analysisResult.getTargetsToTest().contains(configuredTarget)) {
        env.getEventBus()
            .post(
                TestAnalyzedEvent.create(
                    configuredTarget,
                    configurationMap.get(configuredTarget.getConfigurationKey()),
                    /*isSkipped=*/ analysisResult.getTargetsToSkip().contains(configuredTarget)));
      }
    }

    for (Entry<AspectKey, ConfiguredAspect> entry : analysisResult.getAspectsMap().entrySet()) {
      env.getEventBus().post(AspectAnalyzedEvent.create(entry.getKey(), entry.getValue()));
    }
  }

  static void reportTargetsWithTests(
      CommandEnvironment env,
      Collection<ConfiguredTarget> targetsToBuild,
      Collection<ConfiguredTarget> targetsToTest) {
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
  }

  static void reportTargets(CommandEnvironment env, Collection<ConfiguredTarget> targetsToBuild) {
    int targetCount = targetsToBuild.size();
    env.getReporter()
        .handle(
            Event.info("Found " + targetCount + (targetCount == 1 ? " target..." : " targets...")));
  }

  /**
   * Turns target patterns from the command line into parsed equivalents for single targets.
   *
   * <p>Globbing targets like ":all" and "..." are ignored here and will not be in the returned set.
   *
   * @param env the action's environment.
   * @param requestedTargetPatterns the list of target patterns specified on the command line.
   * @param keepGoing --keep_going command line option.
   * @param loadingPhaseThreads no of threads to be used in execution.
   * @return the set of stringified labels of target patterns that represent single targets. The
   *     stringified labels are in the "unambiguous canonical form".
   * @throws ViewCreationFailedException if a pattern fails to parse for some reason.
   */
  private static ImmutableSet<Label> getExplicitTargetPatterns(
      CommandEnvironment env,
      List<String> requestedTargetPatterns,
      boolean keepGoing,
      int loadingPhaseThreads)
      throws ViewCreationFailedException, RepositoryMappingResolutionException,
          InterruptedException {
    ImmutableSet.Builder<Label> explicitTargetPatterns = ImmutableSet.builder();

    // TODO(andreisolo): Don't re-compute these here as they should be already computed inside the
    //  TargetPatternPhaseValue
    RepositoryMapping mainRepoMapping =
        env.getSkyframeExecutor()
            .getMainRepoMapping(keepGoing, loadingPhaseThreads, env.getReporter());
    TargetPattern.Parser parser =
        new Parser(env.getRelativeWorkingDirectory(), RepositoryName.MAIN, mainRepoMapping);

    for (String requestedTargetPattern : requestedTargetPatterns) {
      if (requestedTargetPattern.startsWith("-")) {
        // Excluded patterns are by definition not explicitly requested so we can move on to the
        // next target pattern.
        continue;
      }

      // Parse the pattern. This should always work because this is at least the second time we're
      // doing it. The previous time is in runAnalysisPhase(). Still, if parsing does fail we
      // propagate the exception up.
      TargetPattern parsedPattern;
      try {
        parsedPattern = parser.parse(requestedTargetPattern);
      } catch (TargetParsingException e) {
        throw new ViewCreationFailedException(
            "Failed to parse target pattern even though it was previously parsed successfully",
            e.getDetailedExitCode().getFailureDetail(),
            e);
      }

      if (parsedPattern.getType() == TargetPattern.Type.SINGLE_TARGET) {
        explicitTargetPatterns.add(parsedPattern.getSingleTargetLabel());
      }
    }

    return ImmutableSet.copyOf(explicitTargetPatterns.build());
  }
}
