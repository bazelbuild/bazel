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

import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.AnalysisAndExecutionResult;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.BuildView.BuildConfigurationsCreated;
import com.google.devtools.build.lib.analysis.BuildView.ExecutionSetup;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.test.CoverageArtifactsKnownEvent;
import com.google.devtools.build.lib.buildtool.buildevent.NoAnalyzeEvent;
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
import com.google.devtools.build.lib.skyframe.BuildResultListener;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView.BuildDriverKeyTestContext;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.AspectAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TestAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetAnalyzedEvent;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Intended drop-in replacement for AnalysisPhaseRunner after we're done with merging Skyframe's
 * analysis and execution phases. This is part of https://github.com/bazelbuild/bazel/issues/14057.
 * Internal: b/147350683.
 *
 * <p>TODO(leba): Consider removing this class altogether to reduce complexity.
 */
public final class AnalysisAndExecutionPhaseRunner {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private AnalysisAndExecutionPhaseRunner() {}

  @Nullable
  static AnalysisAndExecutionResult execute(
      CommandEnvironment env,
      BuildRequest request,
      BuildOptions buildOptions,
      TargetPatternPhaseValue loadingResult,
      ExecutionSetup executionSetupCallback,
      BuildConfigurationsCreated buildConfigurationCreatedCallback,
      BuildDriverKeyTestContext buildDriverKeyTestContext)
      throws BuildFailedException,
          InterruptedException,
          ViewCreationFailedException,
          AbruptExitException,
          InvalidConfigurationException,
          TestExecException,
          RepositoryMappingResolutionException {

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

    AnalysisAndExecutionResult analysisAndExecutionResult = null;
    if (request.getBuildOptions().performAnalysisPhase) {
      Profiler.instance().markPhase(ProfilePhase.ANALYZE_AND_EXECUTE);

      try (SilentCloseable c = Profiler.instance().profile("runAnalysisAndExecutionPhase");
          TopLevelTargetAnalysisWatcher watcher =
              TopLevelTargetAnalysisWatcher.createAndRegisterWithEventBus(
                  env.getRuntime().getBlazeModules(), env, request, buildOptions)) {
        analysisAndExecutionResult =
            runAnalysisAndExecutionPhase(
                env,
                request,
                loadingResult,
                buildOptions,
                executionSetupCallback,
                buildConfigurationCreatedCallback,
                buildDriverKeyTestContext);
      }

      BuildResultListener buildResultListener = env.getBuildResultListener();
      if (request.shouldRunTests()) {
        AnalysisPhaseRunner.reportTargetsWithTests(
            env, buildResultListener.getAnalyzedTargets(), buildResultListener.getAnalyzedTests());
      } else {
        AnalysisPhaseRunner.reportTargets(env, buildResultListener.getAnalyzedTargets());
      }

      AnalysisPhaseRunner.postAbortedEventsForSkippedTargets(
          env, buildResultListener.getSkippedTargets());
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

    return analysisAndExecutionResult;
  }

  static TargetPatternPhaseValue evaluateTargetPatterns(
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
   * Performs all phases of the build: Setup, Loading, Analysis & Execution.
   *
   * <p>Postcondition: On success, populates the BuildRequest's set of targets to build.
   *
   * @return null if the build were successful; a useful error message if errors were encountered
   *     and request.keepGoing.
   * @throws InterruptedException if the current thread was interrupted.
   * @throws ViewCreationFailedException if analysis failed for any reason.
   * @throws InvalidConfigurationException if the configuration can't be determined.
   * @throws BuildFailedException if action execution failed.
   * @throws TestExecException if test execution failed.
   */
  private static AnalysisAndExecutionResult runAnalysisAndExecutionPhase(
      CommandEnvironment env,
      BuildRequest request,
      TargetPatternPhaseValue loadingResult,
      BuildOptions targetOptions,
      ExecutionSetup executionSetupCallback,
      BuildConfigurationsCreated buildConfigurationCreatedCallback,
      BuildDriverKeyTestContext buildDriverKeyTestContext)
      throws InterruptedException,
          InvalidConfigurationException,
          ViewCreationFailedException,
          BuildFailedException,
          TestExecException,
          RepositoryMappingResolutionException,
          AbruptExitException {
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
    // TODO(b/199053098) TestFilteringCompleteEvent.
    return (AnalysisAndExecutionResult)
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
            /* includeExecutionPhase= */ true,
            request.getBuildOptions().skymeldAnalysisOverlapPercentage,
            env.getLocalResourceManager(),
            env.getBuildResultListener(),
            executionSetupCallback,
            buildConfigurationCreatedCallback,
            buildDriverKeyTestContext,
            env.getAdditionalConfigurationChangeEvent());
  }

  /**
   * Turns target patterns from the command line into parsed equivalents for single targets.
   *
   * <p>Globbing targets like ":all" and "..." are ignored here and will not be in the returned set.
   *
   * @param env the action's environment.
   * @param requestedTargetPatterns the list of target patterns specified on the command line.
   * @param keepGoing --keep_going command line option
   * @param loadingPhaseThreads no of threads to be used in execution
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
      // doing it. The previous time is in runAnalysisAndExecutionPhase(). Still, if parsing does
      // fail we propagate the exception up.
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

  private static class TopLevelTargetAnalysisWatcher implements AutoCloseable {

    private final Iterable<BlazeModule> blazeModules;
    private final CommandEnvironment env;
    private final BuildRequest buildRequest;
    private final BuildOptions buildOptions;
    private TopLevelTargetAnalysisWatcher(
        Iterable<BlazeModule> blazeModules,
        CommandEnvironment env,
        BuildRequest buildRequest,
        BuildOptions buildOptions) {
      this.blazeModules = blazeModules;
      this.env = env;
      this.buildRequest = buildRequest;
      this.buildOptions = buildOptions;
    }

    /** Creates an AnalysisOperationWatcher and registers it with the provided eventBus. */
    public static TopLevelTargetAnalysisWatcher createAndRegisterWithEventBus(
        Iterable<BlazeModule> blazeModules,
        CommandEnvironment env,
        BuildRequest buildRequest,
        BuildOptions buildOptions) {
      TopLevelTargetAnalysisWatcher watcher =
          new TopLevelTargetAnalysisWatcher(blazeModules, env, buildRequest, buildOptions);
      env.getEventBus().register(watcher);
      return watcher;
    }

    @Subscribe
    public void handleTopLevelTargetAnalysisConcluded(TopLevelTargetAnalyzedEvent e)
        throws ViewCreationFailedException, InterruptedException {
      for (BlazeModule blazeModule : blazeModules) {
        blazeModule.afterTopLevelTargetAnalysis(
            env, buildRequest, buildOptions, e.configuredTarget());
      }
    }

    @Subscribe
    public void handleAspectAnalyzed(AspectAnalyzedEvent e) {
      for (BlazeModule blazeModule : blazeModules) {
        blazeModule.afterSingleAspectAnalysis(buildRequest, e.configuredAspect());
      }
    }

    @Subscribe
    public void handleTestAnalyzed(TestAnalyzedEvent e) {
      for (BlazeModule blazeModule : blazeModules) {
        blazeModule.afterSingleTestAnalysis(buildRequest, e.configuredTarget());
      }
    }

    @Subscribe
    public void handleKnownCoverageArtifacts(CoverageArtifactsKnownEvent e) {
      for (BlazeModule blazeModule : blazeModules) {
        blazeModule.coverageArtifactsKnown(e.coverageArtifacts());
      }
    }

    @Override
    public void close() {
      env.getEventBus().unregister(this);
    }
  }
}
