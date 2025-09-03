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

package com.google.devtools.build.lib.bazel.coverage;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.primitives.Booleans.falseFirst;
import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.ImportantOutputHandler.ImportantOutputException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.IncompatiblePlatformProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.test.CoverageReportActionFactory.CoverageReportActionsWrapper;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestProvider.TestParams;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.exec.SpawnStrategyResolver;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;
import java.util.Comparator;
import javax.annotation.Nullable;

/**
 * A class to create the coverage report generator action.
 *
 * <p>The coverage report action is created after every test shard action is created, at the very
 * end of the analysis phase. There is only one coverage report action per coverage command
 * invocation. It can also be viewed as a single sink node of the action graph.
 *
 * <p>Its inputs are the individual coverage.dat files from the test outputs (each shard produces
 * one) and the baseline coverage artifacts. Note that each ConfiguredTarget among the transitive
 * dependencies of the top level test targets may provide baseline coverage artifacts.
 *
 * <p>The coverage report generation can have two phases, though they both run in the same action.
 * The source code of the coverage report tool {@code lcov_merger} is in the {@code
 * testing/coverage/lcov_merger} directory. The deployed binaries used by Blaze are under {@code
 * tools/coverage}.
 *
 * <p>The first phase is merging the individual coverage files into a single report file. The
 * location of this file is reported by Blaze. This phase always happens if the {@code
 * --combined_report=lcov} or {@code --combined_report=html}.
 *
 * <p>The second phase is generating an html report. It only happens if {@code
 * --combined_report=html}. The action generates an html output file potentially for every tested
 * source file into the report. Since this set of files is unknown in the analysis phase (the tool
 * figures it out from the contents of the merged coverage report file) the action always runs
 * locally when {@code --combined_report=html}.
 */
public final class CoverageReportActionBuilder {

  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpu(/* memoryMb= */ 750, /* cpu= */ 1);

  private static final Comparator<ActionOwner> ACTION_OWNER_COMPARATOR =
      comparing(
              (ActionOwner actionOwner) -> actionOwner.getExecProperties().isEmpty(), falseFirst())
          .thenComparing(ActionOwner::getLabel)
          .thenComparing(ActionOwner::getConfigurationChecksum);

  // SpawnActions can't be used because they need the AnalysisEnvironment and this action is
  // created specially at the very end of the analysis phase when we don't have it anymore.
  @Immutable
  private static final class CoverageReportAction extends AbstractAction
      implements NotifyOnActionCacheHit {
    private final ImmutableList<String> command;
    private final String locationMessage;

    CoverageReportAction(
        ActionOwner owner,
        NestedSet<Artifact> inputs,
        ImmutableSet<Artifact> outputs,
        ImmutableList<String> command,
        String locationMessage) {
      super(owner, inputs, outputs);
      this.command = command;
      this.locationMessage = locationMessage;
    }

    @Override
    public ActionResult execute(ActionExecutionContext ctx)
        throws ActionExecutionException, InterruptedException {
      Spawn spawn =
          new BaseSpawn(command, ImmutableMap.of(), ImmutableMap.of(), this, LOCAL_RESOURCES);
      try {
        ImmutableList<SpawnResult> spawnResults =
            ctx.getContext(SpawnStrategyResolver.class).exec(spawn, ctx);
        informImportantOutputHandler(ctx);
        ctx.getEventHandler().handle(Event.info(locationMessage));
        return ActionResult.create(spawnResults);
      } catch (ExecException e) {
        throw ActionExecutionException.fromExecException(e, this);
      }
    }

    private void informImportantOutputHandler(ActionExecutionContext ctx)
        throws EnvironmentalExecException, InterruptedException {
      var importantOutputHandler = ctx.getContext(ImportantOutputHandler.class);
      if (importantOutputHandler == null) {
        return;
      }

      ImmutableList<Path> coverageReportOutputs =
          getOutputs().stream()
              .map(o -> ctx.getPathResolver().toPath(o))
              .collect(toImmutableList());
      try (var ignored =
          GoogleAutoProfilerUtils.profiledAndLogged(
              "Informing important output handler of coverage report",
              ProfilerTask.INFO,
              ImportantOutputHandler.LOG_THRESHOLD)) {
        importantOutputHandler.processTestOutputs(coverageReportOutputs);
      } catch (ImportantOutputException e) {
        throw new EnvironmentalExecException(e, e.getFailureDetail());
      }
    }

    @Override
    public String getMnemonic() {
      return "CoverageReport";
    }

    @Override
    protected String getRawProgressMessage() {
      return "Coverage report generation";
    }

    @Override
    protected void computeKey(
        ActionKeyContext actionKeyContext,
        @Nullable InputMetadataProvider inputMetadataProvider,
        Fingerprint fp) {
      fp.addStrings(command);
    }

    @Override
    public boolean actionCacheHit(ActionCachedContext context) {
      context.getEventHandler().handle(Event.info(locationMessage));
      return true;
    }
  }

  public CoverageReportActionBuilder() {}

  /** Returns the coverage report action. May return null in case of an error. */
  @Nullable
  public CoverageReportActionsWrapper createCoverageActionsWrapper(
      EventHandler reporter,
      BlazeDirectories directories,
      Collection<ConfiguredTarget> configuredTargets,
      Collection<ConfiguredTarget> targetsToTest,
      ArtifactFactory factory,
      ActionKeyContext actionKeyContext,
      ArtifactOwner artifactOwner,
      String workspaceName,
      CoverageHelper coverageHelper,
      Artifact htmlReport)
      throws InterruptedException {
    if (targetsToTest == null || targetsToTest.isEmpty()) {
      return null;
    }
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    FilesToRunProvider reportGenerator = null;
    ActionOwner actionOwner = null;
    for (ConfiguredTarget target : targetsToTest) {
      // Skip incompatible tests.
      if (target.get(IncompatiblePlatformProvider.PROVIDER) != null) {
        continue;
      }
      TestParams testParams = target.getProvider(TestProvider.class).getTestParams();
      builder.addAll(testParams.getCoverageArtifacts());
      // targetsToTest has non-deterministic order, so we ensure that we pick the same action owner
      // and matching report generator each time by picking the owner that's lexicographically
      // largest. We prefer an owner with exec properties set in case the action is run remotely.
      if (reportGenerator == null
          || ACTION_OWNER_COMPARATOR.compare(testParams.getActionOwnerForCoverage(), actionOwner)
              > 0) {
        reportGenerator = testParams.getCoverageReportGenerator();
        actionOwner = testParams.getActionOwnerForCoverage();
      }
    }
    // If all tests are incompatible, there's nothing to do.
    if (reportGenerator == null) {
      return null;
    }
    checkNotNull(actionOwner);
    NestedSet<Artifact> baselineCoverageArtifacts = getBaselineCoverageArtifacts(configuredTargets);
    NestedSet<Artifact> coverageArtifacts =
        builder.addTransitive(baselineCoverageArtifacts).build();
    if (!coverageArtifacts.isEmpty()) {
      Artifact baselineLcovArtifact =
          factory.getDerivedArtifact(
              TestRunnerAction.COVERAGE_TMP_ROOT.getRelative("baseline_lcov_files.tmp"),
              directories.getBuildDataDirectory(workspaceName),
              artifactOwner);
      Action baselineLcovFileAction =
          generateLcovFileWriteAction(baselineLcovArtifact, baselineCoverageArtifacts, actionOwner);
      Action baselineReportAction =
          generateCoverageReportAction(
              new CoverageArgs(
                  directories,
                  baselineCoverageArtifacts,
                  baselineLcovArtifact,
                  factory,
                  artifactOwner,
                  reportGenerator,
                  workspaceName,
                  /* htmlReport= */ null,
                  actionOwner),
              coverageHelper,
              configuredTargets,
              "_baseline_report.dat");
      Artifact coverageLcovArtifact =
          factory.getDerivedArtifact(
              TestRunnerAction.COVERAGE_TMP_ROOT.getRelative("coverage_lcov_files.tmp"),
              directories.getBuildDataDirectory(workspaceName),
              artifactOwner);
      Action coverageLcovFileAction =
          generateLcovFileWriteAction(coverageLcovArtifact, coverageArtifacts, actionOwner);
      Action coverageReportAction =
          generateCoverageReportAction(
              new CoverageArgs(
                  directories,
                  coverageArtifacts,
                  coverageLcovArtifact,
                  factory,
                  artifactOwner,
                  reportGenerator,
                  workspaceName,
                  htmlReport,
                  actionOwner),
              coverageHelper,
              configuredTargets,
              "_coverage_report.dat");
      return new CoverageReportActionsWrapper(
          baselineReportAction,
          coverageReportAction,
          ImmutableList.of(baselineLcovFileAction, coverageLcovFileAction),
          actionKeyContext);
    } else {
      reporter.handle(
          Event.error("Cannot generate coverage report - no coverage information was collected"));
      return null;
    }
  }

  private static NestedSet<Artifact> getBaselineCoverageArtifacts(
      Collection<ConfiguredTarget> configuredTargets) {
    var baselineCoverageArtifacts = NestedSetBuilder.<Artifact>stableOrder();
    for (ConfiguredTarget target : configuredTargets) {
      InstrumentedFilesInfo provider = target.get(InstrumentedFilesInfo.STARLARK_CONSTRUCTOR);
      if (provider != null) {
        baselineCoverageArtifacts.addTransitive(provider.getBaselineCoverageArtifacts());
      }
    }
    return baselineCoverageArtifacts.build();
  }

  private static LazyWritePathsFileAction generateLcovFileWriteAction(
      Artifact lcovArtifact, NestedSet<Artifact> coverageArtifacts, ActionOwner actionOwner) {
    return new LazyWritePathsFileAction(
        actionOwner,
        lcovArtifact,
        coverageArtifacts,
        /* filesToIgnore= */ ImmutableSet.of(),
        /* includeDerivedArtifacts= */ true);
  }

  /** A helper interface for product-specific coverage support. */
  public interface CoverageHelper {
    /** Returns the arguments for the coverage report generator action. */
    ImmutableList<String> getArgs(CoverageArgs args, Artifact lcovOutput);

    /** Returns a message describing the location of the coverage report. */
    String getLocationMessage(CoverageArgs args, Artifact lcovOutput);

    /** Returns additional inputs for the coverage report action. */
    default NestedSet<Artifact> getInstrumentedFiles(
        CoverageArgs args, Collection<ConfiguredTarget> configuredTargets) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  private static CoverageReportAction generateCoverageReportAction(
      CoverageArgs args,
      CoverageHelper coverageHelper,
      Collection<ConfiguredTarget> configuredTargets,
      String basename) {
    ArtifactRoot root = args.directories().getBuildDataDirectory(args.workspaceName());
    Artifact lcovOutput =
        args.factory()
            .getDerivedArtifact(
                TestRunnerAction.COVERAGE_TMP_ROOT.getRelative(basename),
                root,
                args.artifactOwner());
    Artifact reportGeneratorExec = args.reportGenerator().getExecutable();
    RunfilesSupport runfilesSupport = args.reportGenerator().getRunfilesSupport();
    Artifact runfilesTree =
        runfilesSupport != null ? runfilesSupport.getRunfilesTreeArtifact() : null;
    ImmutableList<String> actionArgs = coverageHelper.getArgs(args, lcovOutput);

    NestedSetBuilder<Artifact> inputsBuilder =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(args.coverageArtifacts())
            .add(reportGeneratorExec)
            .add(args.lcovArtifact());
    if (runfilesTree != null) {
      inputsBuilder.add(runfilesTree);
    }

    ImmutableSet.Builder<Artifact> outputsBuilder =
        ImmutableSet.<Artifact>builder().add(lcovOutput);
    if (args.htmlReport() != null) {
      outputsBuilder.add(args.htmlReport());
      inputsBuilder.addTransitive(coverageHelper.getInstrumentedFiles(args, configuredTargets));
    }
    return new CoverageReportAction(
        args.actionOwner(),
        inputsBuilder.build(),
        outputsBuilder.build(),
        actionArgs,
        coverageHelper.getLocationMessage(args, lcovOutput));
  }
}
