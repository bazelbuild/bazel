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
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.includescanning.IncludeScanningModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.testutil.ActionEventRecorder;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for action rewinding on non-incremental builds. */
// TODO(b/228090759): Add back actionFromPreviousBuildReevaluated when incrementality is supported.
@RunWith(TestParameterInjector.class)
public final class RewindingTest extends BuildIntegrationTestCase {

  @TestParameter private boolean keepGoing;

  private final ActionEventRecorder actionEventRecorder = new ActionEventRecorder();
  private final RewindingTestsHelper helper = new RewindingTestsHelper(this, actionEventRecorder);

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new IncludeScanningModule())
        .addBlazeModule(helper.makeControllableActionStrategyModule("standalone"));
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions(
        "--spawn_strategy=standalone",
        "--notrack_incremental_state",
        "--nouse_action_cache",
        "--rewind_lost_inputs",
        "--features=cc_include_scanning",
        "--experimental_remote_include_extraction_size_threshold=0",
        "--keep_going=" + keepGoing);
    runtimeWrapper.registerSubscriber(actionEventRecorder);
    // Tell Skyframe to ignore RepositoryHelpersHolder so that we don't trigger
    // RepoMappingManifestAction to preserve the expected order of Actions.
    this.getSkyframeExecutor().ignoreRepositoryHelpersHolderForTesting();
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
  public void multiplyLosingInputsFails() throws Exception {
    helper.runMultiplyLosingInputsFails();
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
}
