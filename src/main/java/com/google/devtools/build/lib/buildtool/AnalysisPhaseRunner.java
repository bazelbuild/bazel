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
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.analysis.StaticallyLinkedMarkerProvider;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.buildeventstream.AbortedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildtool.buildevent.NoAnalyzeEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.util.AbruptExitException;
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

  protected CommandEnvironment env;

  public AnalysisPhaseRunner(CommandEnvironment env) {
    this.env = env;
  }

  public AnalysisResult execute(
      BuildRequest request, BuildOptions buildOptions, TargetValidator validator)
      throws BuildFailedException, InterruptedException, ViewCreationFailedException,
          TargetParsingException, LoadingFailedException, AbruptExitException,
          InvalidConfigurationException {

    // Target pattern evaluation.
    TargetPatternPhaseValue loadingResult;
    Profiler.instance().markPhase(ProfilePhase.LOAD);
    try (SilentCloseable c = Profiler.instance().profile("evaluateTargetPatterns")) {
      loadingResult = evaluateTargetPatterns(request, validator);
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
          buildOptions.get(BuildConfiguration.Options.class).instrumentationFilter =
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
      try (SilentCloseable c = Profiler.instance().profile("runAnalysisPhase")) {
        analysisResult =
            runAnalysisPhase(request, loadingResult, buildOptions, request.getMultiCpus());
      }

      // Check licenses.
      // We check licenses if the first target configuration has license checking enabled. Right
      // now, it is not possible to have multiple target configurations with different settings
      // for this flag, which allows us to take this short cut.
      boolean checkLicenses = buildOptions.get(BuildConfiguration.Options.class).checkLicenses;
      if (checkLicenses) {
        Profiler.instance().markPhase(ProfilePhase.LICENSE);
        try (SilentCloseable c = Profiler.instance().profile("validateLicensingForTargets")) {
          validateLicensingForTargets(analysisResult.getTargetsToBuild(), request.getKeepGoing());
        }
      }

      reportTargets(analysisResult);

      for (ConfiguredTarget target : analysisResult.getTargetsToSkip()) {
        BuildConfiguration config =
            env.getSkyframeExecutor()
                .getConfiguration(env.getReporter(), target.getConfigurationKey());
        Label label = target.getLabel();
        env.getEventBus()
            .post(
                new AbortedEvent(
                    BuildEventId.targetCompleted(label, config.getEventId()),
                    AbortReason.SKIPPED,
                    String.format("Target %s build was skipped.", label),
                    label));
      }
    } else {
      env.getReporter().handle(Event.progress("Loading complete."));
      env.getReporter().post(new NoAnalyzeEvent());
      logger.atInfo().log("No analysis requested, so finished");
      String errorMessage = BuildView.createErrorMessage(loadingResult, null);
      if (errorMessage != null) {
        throw new BuildFailedException(errorMessage);
      }
    }

    return analysisResult;
  }

  private final TargetPatternPhaseValue evaluateTargetPatterns(
      final BuildRequest request, final TargetValidator validator)
      throws LoadingFailedException, TargetParsingException, InterruptedException {
    boolean keepGoing = request.getKeepGoing();
    TargetPatternPhaseValue result =
        env.getSkyframeExecutor()
            .loadTargetPatterns(
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
  private AnalysisResult runAnalysisPhase(
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

    // TODO(bazel-team): Merge these into one event.
    env.getEventBus()
        .post(
            new AnalysisPhaseCompleteEvent(
                analysisResult.getTargetsToBuild(),
                view.getTargetsLoaded(),
                view.getTargetsConfigured(),
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

  private void reportTargets(AnalysisResult analysisResult) {
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

  /**
   * Takes a set of configured targets, and checks if the distribution methods declared for the
   * targets are compatible with the constraints imposed by their prerequisites' licenses.
   *
   * @param configuredTargets the targets to check
   * @param keepGoing if false, and a licensing error is encountered, both generates an error
   *     message on the reporter, <em>and</em> throws an exception. If true, then just generates a
   *     message on the reporter.
   * @throws ViewCreationFailedException if the license checking failed (and not --keep_going)
   */
  private void validateLicensingForTargets(
      Iterable<ConfiguredTarget> configuredTargets, boolean keepGoing)
      throws ViewCreationFailedException {
    for (ConfiguredTarget configuredTarget : configuredTargets) {
      Target target = null;
      try {
        target = env.getPackageManager().getTarget(env.getReporter(), configuredTarget.getLabel());
      } catch (NoSuchPackageException | NoSuchTargetException | InterruptedException e) {
        env.getReporter().handle(Event.error("Failed to get target to validate license"));
        throw new ViewCreationFailedException(
            "Build aborted due to issue getting targets to validate licenses", e);
      }

      if (TargetUtils.isTestRule(target)) {
        continue; // Tests are exempt from license checking
      }

      final Set<DistributionType> distribs = target.getDistributions();
      StaticallyLinkedMarkerProvider markerProvider =
          configuredTarget.getProvider(StaticallyLinkedMarkerProvider.class);
      boolean staticallyLinked = markerProvider != null && markerProvider.isLinkedStatically();

      LicensesProvider provider = configuredTarget.getProvider(LicensesProvider.class);
      if (provider != null) {
        NestedSet<TargetLicense> licenses = provider.getTransitiveLicenses();
        for (TargetLicense targetLicense : licenses) {
          if (!targetLicense
              .getLicense()
              .checkCompatibility(
                  distribs,
                  target,
                  targetLicense.getLabel(),
                  env.getReporter(),
                  staticallyLinked)) {
            if (!keepGoing) {
              throw new ViewCreationFailedException("Build aborted due to licensing error");
            }
          }
        }
      } else if (target instanceof InputFile) {
        // Input file targets do not provide licenses because they do not
        // depend on the rule where their license is taken from. This is usually
        // not a problem, because the transitive collection of licenses always
        // hits the rule they come from, except when the input file is a
        // top-level target. Thus, we need to handle that case specially here.
        //
        // See FileTarget#getLicense for more information about the handling of
        // license issues with File targets.
        License license = target.getLicense();
        if (!license.checkCompatibility(
            distribs, target, configuredTarget.getLabel(), env.getReporter(), staticallyLinked)) {
          if (!keepGoing) {
            throw new ViewCreationFailedException("Build aborted due to licensing error");
          }
        }
      }
    }
  }
}
