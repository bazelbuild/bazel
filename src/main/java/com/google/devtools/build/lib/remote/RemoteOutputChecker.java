// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.packages.TargetUtils.isTestRuleName;
import static com.google.devtools.build.lib.skyframe.CoverageReportValue.COVERAGE_REPORT_KEY;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.ProviderCollection;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.clock.Clock;
import java.util.Set;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** A {@link RemoteArtifactChecker} that checks the TTL of remote metadata. */
public class RemoteOutputChecker implements RemoteArtifactChecker {
  private enum CommandMode {
    UNKNOWN,
    BUILD,
    TEST,
    RUN,
    COVERAGE;
  }

  private final Clock clock;
  private final CommandMode commandMode;
  private final boolean downloadToplevel;
  private final ImmutableList<Pattern> patternsToDownload;
  private final Set<ActionInput> toplevelArtifactsToDownload = Sets.newConcurrentHashSet();
  private final Set<ActionInput> inputsToDownload = Sets.newConcurrentHashSet();

  public RemoteOutputChecker(
      Clock clock,
      String commandName,
      boolean downloadToplevel,
      ImmutableList<Pattern> patternsToDownload) {
    this.clock = clock;
    switch (commandName) {
      case "build":
        this.commandMode = CommandMode.BUILD;
        break;
      case "test":
        this.commandMode = CommandMode.TEST;
        break;
      case "run":
        this.commandMode = CommandMode.RUN;
        break;
      case "coverage":
        this.commandMode = CommandMode.COVERAGE;
        break;
      default:
        this.commandMode = CommandMode.UNKNOWN;
    }
    this.downloadToplevel = downloadToplevel;
    this.patternsToDownload = patternsToDownload;
  }

  // TODO(chiwang): Code path reserved for skymeld.
  public void afterTopLevelTargetAnalysis(
      ConfiguredTarget configuredTarget,
      Supplier<TopLevelArtifactContext> topLevelArtifactContextSupplier) {
    addTopLevelTarget(configuredTarget, configuredTarget, topLevelArtifactContextSupplier);
  }

  public void afterAnalysis(AnalysisResult analysisResult) {
    for (var target : analysisResult.getTargetsToBuild()) {
      addTopLevelTarget(target, target, analysisResult::getTopLevelContext);
    }
    for (var aspect : analysisResult.getAspectsMap().values()) {
      addTopLevelTarget(aspect, /* configuredTarget= */ null, analysisResult::getTopLevelContext);
    }
    var targetsToTest = analysisResult.getTargetsToTest();
    if (targetsToTest != null) {
      for (var target : targetsToTest) {
        addTargetUnderTest(target, analysisResult.getArtifactsToBuild());
      }
    }
  }

  private void addTopLevelTarget(
      ProviderCollection target,
      @Nullable ConfiguredTarget configuredTarget,
      Supplier<TopLevelArtifactContext> topLevelArtifactContextSupplier) {
    if (shouldAddTopLevelTarget(configuredTarget)) {
      var topLevelArtifactContext = topLevelArtifactContextSupplier.get();
      var artifactsToBuild =
          TopLevelArtifactHelper.getAllArtifactsToBuild(target, topLevelArtifactContext)
              .getImportantArtifacts();
      toplevelArtifactsToDownload.addAll(artifactsToBuild.toList());

      addRunfiles(target);
    }
  }

  private void addRunfiles(ProviderCollection buildTarget) {
    var runfilesProvider = buildTarget.getProvider(FilesToRunProvider.class);
    if (runfilesProvider == null) {
      return;
    }
    var runfilesSupport = runfilesProvider.getRunfilesSupport();
    if (runfilesSupport == null) {
      return;
    }
    var runfiles = runfilesSupport.getRunfiles();
    for (Artifact runfile : runfiles.getArtifacts().toList()) {
      if (runfile.isSourceArtifact()) {
        continue;
      }
      toplevelArtifactsToDownload.add(runfile);
    }
    for (var symlink : runfiles.getSymlinks().toList()) {
      var artifact = symlink.getArtifact();
      if (artifact.isSourceArtifact()) {
        continue;
      }
      toplevelArtifactsToDownload.add(artifact);
    }
    for (var symlink : runfiles.getRootSymlinks().toList()) {
      var artifact = symlink.getArtifact();
      if (artifact.isSourceArtifact()) {
        continue;
      }
      toplevelArtifactsToDownload.add(artifact);
    }
  }

  private void addTargetUnderTest(
      ProviderCollection target, ImmutableSet<Artifact> artifactsToBuild) {
    TestProvider testProvider = checkNotNull(target.getProvider(TestProvider.class));
    if (downloadToplevel && commandMode == CommandMode.TEST) {
      // In test mode, download the outputs of the test runner action.
      toplevelArtifactsToDownload.addAll(testProvider.getTestParams().getOutputs());
    }
    if (commandMode == CommandMode.COVERAGE) {
      // In coverage mode, download the per-test and aggregated coverage files.
      // Do this even for MINIMAL, since coverage (unlike test) doesn't produce any observable
      // results other than outputs.
      toplevelArtifactsToDownload.addAll(testProvider.getTestParams().getCoverageArtifacts());
      for (Artifact artifactToBuild : artifactsToBuild) {
        if (artifactToBuild.getArtifactOwner().equals(COVERAGE_REPORT_KEY)) {
          toplevelArtifactsToDownload.add(artifactToBuild);
        }
      }
    }
  }

  public void addInputToDownload(ActionInput file) {
    inputsToDownload.add(file);
  }

  private boolean shouldAddTopLevelTarget(@Nullable ConfiguredTarget configuredTarget) {
    switch (commandMode) {
      case RUN:
        // Always download outputs of toplevel targets in run mode.
        return true;
      case COVERAGE:
      case TEST:
        // Do not download test binary in test/coverage mode.
        if (configuredTarget instanceof RuleConfiguredTarget) {
          var ruleConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
          var isTestRule = isTestRuleName(ruleConfiguredTarget.getRuleClassString());
          return !isTestRule && downloadToplevel;
        }
        return downloadToplevel;
      default:
        return downloadToplevel;
    }
  }

  private boolean shouldDownloadOutputForToplevel(ActionInput output) {
    return shouldDownloadOutputFor(output, toplevelArtifactsToDownload);
  }

  private boolean shouldDownloadOutputForLocalAction(ActionInput output) {
    return shouldDownloadOutputFor(output, inputsToDownload);
  }

  private boolean shouldDownloadOutputForRegex(ActionInput output) {
    if (output instanceof Artifact && ((Artifact) output).isTreeArtifact()) {
      return false;
    }

    for (var pattern : patternsToDownload) {
      if (pattern.matcher(output.getExecPathString()).matches()) {
        return true;
      }
    }

    return false;
  }

  private static boolean shouldDownloadOutputFor(
      ActionInput output, Set<ActionInput> artifactCollection) {
    if (output instanceof TreeFileArtifact) {
      if (artifactCollection.contains(((Artifact) output).getParent())) {
        return true;
      }
    } else if (artifactCollection.contains(output)) {
      return true;
    }

    return false;
  }

  /**
   * Returns {@code true} if Bazel should download this {@link ActionInput} during spawn execution.
   */
  public boolean shouldDownloadOutput(ActionInput output) {
    return shouldDownloadOutputForToplevel(output)
        || shouldDownloadOutputForLocalAction(output)
        || shouldDownloadOutputForRegex(output);
  }

  @Override
  public boolean shouldTrustRemoteArtifact(ActionInput file, RemoteFileArtifactValue metadata) {
    if (shouldDownloadOutput(file)) {
      // If Bazel should download this file, but it does not exist locally, returns false to rerun
      // the generating action to trigger the download (just like in the normal build, when local
      // outputs are missing).
      return false;
    }

    return metadata.isAlive(clock.now());
  }
}
