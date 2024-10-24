// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.rewinding;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.TruthJUnit.assume;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.includescanning.IncludeScanningModule;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.testutil.ActionEventRecorder;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Integration tests for action rewinding.
 *
 * <p>Uses {@link TestParameter}s to run tests with all combinations of {@code
 * --track_incremental_state}, {@code --keep_going}, and {@code
 * --experimental_merged_skyframe_analysis_execution}.
 */
// TODO(b/228090759): Consider asserting on graph structure to improve coverage for incrementality.
// TODO(b/228090759): Add back actionFromPreviousBuildReevaluated.
@RunWith(TestParameterInjector.class)
public final class RewindingTest extends BuildIntegrationTestCase {

  @TestParameter private boolean trackIncrementalState;
  @TestParameter private boolean keepGoing;
  @TestParameter private boolean skymeld;

  private final ActionEventRecorder actionEventRecorder = new ActionEventRecorder();
  private final RewindingTestsHelper helper = new RewindingTestsHelper(this, actionEventRecorder);

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new IncludeScanningModule())
        .addBlazeModule(helper.makeControllableActionStrategyModule("standalone"))
        .addBlazeModule(helper.getLostOutputsModule())
        .addBlazeModule(
            new BlazeModule() {
              @Override
              public void workspaceInit(
                  BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
                // Null out RepositoryHelpersHolder so that we don't trigger
                // RepoMappingManifestAction. This preserves action graph structure between blaze
                // and bazel, which is important for this test's assertions.
                builder.setSkyframeExecutorRepositoryHelpersHolder(null);
              }
            });
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions(
        "--spawn_strategy=standalone",
        "--noexperimental_merged_skyframe_analysis_execution",
        "--rewind_lost_inputs",
        "--features=cc_include_scanning",
        "--experimental_remote_include_extraction_size_threshold=0",
        "--track_incremental_state=" + trackIncrementalState,
        "--keep_going=" + keepGoing,
        "--experimental_merged_skyframe_analysis_execution=" + skymeld);
    runtimeWrapper.registerSubscriber(actionEventRecorder);
  }

  /**
   * Skips test cases that cannot run with bazel.
   *
   * <p>{@link BuildIntegrationTestCase} currently does not support CPP compilation on bazel.
   */
  // TODO(b/195425240): Remove once CPP compilation on bazel is supported. Assumptions that
  // generated headers are always under k8-opt will need to be relaxed to support other platforms.
  private static void skipIfBazel() {
    assume().that(AnalysisMock.get().isThisBazel()).isFalse();
  }

  @Test
  public void noLossSmokeTest() throws Exception {
    helper.runNoLossSmokeTest();
  }

  @Test
  public void lostInputWithRewindingDisabled() throws Exception {
    helper.runLostInputWithRewindingDisabled();
  }

  @Test
  public void buildingParentFoundUndoneChildNotToleratedWithoutRewinding() throws Exception {
    helper.runBuildingParentFoundUndoneChildNotToleratedWithoutRewinding();
  }

  @Test
  public void dependentActionsReevaluated() throws Exception {
    helper.runDependentActionsReevaluated_spawnFailed();
  }

  @Test
  public void multipleLostInputsForRewindPlan() throws Exception {
    helper.runMultipleLostInputsForRewindPlan();
  }

  @Test
  public void ineffectiveRewindingResultsInLostInputTooManyTimes() throws Exception {
    helper.runIneffectiveRewindingResultsInLostInputTooManyTimes();
    assertOutputForRule2NotCreated();
  }

  @Test
  public void interruptedDuringRewindStopsNormally() throws Exception {
    helper.runInterruptedDuringRewindStopsNormally();
    assertOutputForRule2NotCreated();
  }

  @Test
  public void failureDuringRewindStopsNormally() throws Exception {
    helper.runFailureDuringRewindStopsNormally();
    assertOutputForRule2NotCreated();
  }

  /**
   * Because this test infrastructure allows builds to write outputs to the filesystem, these
   * "fail"/"stops normally" tests can assert that the build's output file was not written.
   */
  private void assertOutputForRule2NotCreated() throws Exception {
    Artifact output =
        Iterables.getOnlyElement(
            getFilesToBuild(getExistingConfiguredTarget("//test:rule2")).toList());
    assertThat(output.getPath().exists()).isFalse();
  }

  @Test
  public void intermediateActionRewound() throws Exception {
    helper.runIntermediateActionRewound();
  }

  @Test
  public void chainOfActionsRewound() throws Exception {
    helper.runChainOfActionsRewound();
  }

  @Test
  public void nondeterministicActionRewound() throws Exception {
    helper.runNondeterministicActionRewound();
  }

  @Test
  public void parallelTrackSharedActionsRewound() throws Exception {
    helper.runParallelTrackSharedActionsRewound();
  }

  @Test
  public void treeFileArtifactRewound() throws Exception {
    skipIfBazel();
    helper.runTreeFileArtifactRewound_spawnFailed();
  }

  @Test
  public void treeArtifactRewound_allFilesLost() throws Exception {
    skipIfBazel();
    helper.runTreeArtifactRewound_allFilesLost_spawnFailed();
  }

  @Test
  public void treeArtifactRewound_oneFileLost() throws Exception {
    skipIfBazel();
    helper.runTreeArtifactRewound_oneFileLost_spawnFailed();
  }

  @Test
  public void generatedRunfilesRewound_allFilesLost() throws Exception {
    helper.runGeneratedRunfilesRewound_allFilesLost_spawnFailed();
  }

  @Test
  public void generatedRunfilesRewound_oneFileLost() throws Exception {
    helper.runGeneratedRunfilesRewound_oneFileLost_spawnFailed();
  }

  @Test
  public void dupeDirectAndRunfilesDependencyRewound() throws Exception {
    helper.runDupeDirectAndRunfilesDependencyRewound_spawnFailed();
  }

  @Test
  public void treeInRunfilesRewound() throws Exception {
    helper.runTreeInRunfilesRewound_spawnFailed();
  }

  @Test
  public void inputsFromSameGeneratingActionSplitAmongNestedSetChildren() throws Exception {
    helper.runInputsFromSameGeneratingActionSplitAmongNestedSetChildren();
  }

  @Test
  public void generatedHeaderRewound_lostInInputDiscovery() throws Exception {
    skipIfBazel();
    helper.runGeneratedHeaderRewound_lostInInputDiscovery_spawnFailed();
  }

  @Test
  public void generatedHeaderRewound_lostInActionExecution() throws Exception {
    skipIfBazel();
    helper.runGeneratedHeaderRewound_lostInActionExecution_spawnFailed();
  }

  @Test
  public void generatedTransitiveHeaderRewound_lostInInputDiscovery() throws Exception {
    skipIfBazel();
    helper.runGeneratedTransitiveHeaderRewound_lostInInputDiscovery_spawnFailed();
  }

  @Test
  public void generatedTransitiveHeaderRewound_lostInActionExecution() throws Exception {
    skipIfBazel();
    helper.runGeneratedTransitiveHeaderRewound_lostInActionExecution_spawnFailed();
  }

  @Test
  public void doneToDirtyDepForNodeInError() throws Exception {
    helper.runDoneToDirtyDepForNodeInError();
  }

  @Test
  public void flakyActionFailsAfterRewind_raceWithIndirectConsumer_undoneDuringInputChecking()
      throws Exception {
    helper.runFlakyActionFailsAfterRewind_raceWithIndirectConsumer_undoneDuringInputChecking();
  }

  @Test
  public void flakyActionFailsAfterRewind_raceWithIndirectConsumer_undoneDuringLostInputHandling()
      throws Exception {
    helper.runFlakyActionFailsAfterRewind_raceWithIndirectConsumer_undoneDuringLostInputHandling();
  }

  @Test
  public void discoveredCppModuleLost() throws Exception {
    skipIfBazel();
    helper.runDiscoveredCppModuleLost();
  }

  @Test
  public void lostTopLevelOutputWithRewindingDisabled() throws Exception {
    helper.runLostTopLevelOutputWithRewindingDisabled();
  }

  @Test
  public void topLevelOutputRewound_regularFile() throws Exception {
    helper.runTopLevelOutputRewound_regularFile();
  }

  @Test
  public void topLevelOutputRewound_aspectOwned() throws Exception {
    helper.runTopLevelOutputRewound_aspectOwned();
  }

  @Test
  public void topLevelOutputRewound_fileInTreeArtifact() throws Exception {
    helper.runTopLevelOutputRewound_fileInTreeArtifact();
  }

  @Test
  public void topLevelOutputRewound_partiallyBuiltTarget_regularFile() throws Exception {
    helper.runTopLevelOutputRewound_partiallyBuiltTarget_regularFile();
  }

  @Test
  public void topLevelOutputRewound_partiallyBuiltTarget_fileInTreeArtifact() throws Exception {
    helper.runTopLevelOutputRewound_partiallyBuiltTarget_fileInTreeArtifact();
  }

  @Test
  public void topLevelOutputRewound_ineffectiveRewinding() throws Exception {
    helper.runTopLevelOutputRewound_ineffectiveRewinding();
  }
}
