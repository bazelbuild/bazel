// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.test.ProducerKeyedTestCacheEligibility;
import com.google.devtools.build.lib.analysis.test.ProducerKeyedTestCacheEligibility.Eligible;
import com.google.devtools.build.lib.analysis.test.ProducerKeyedTestCacheEligibility.IneligibilityReason;
import com.google.devtools.build.lib.analysis.test.ProducerKeyedTestCacheEligibility.Ineligible;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.test.TestTargetExecutionSettings;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ProducerKeyedTestCacheEligibilityTest {
  private static final Label OWNER = Label.parseCanonicalUnchecked("//pkg:test");

  private final TestRunnerAction testAction = mock(TestRunnerAction.class);
  private final TestTargetExecutionSettings executionSettings =
      mock(TestTargetExecutionSettings.class);
  private final DerivedArtifact executable = mock(DerivedArtifact.class);
  private final SpawnAction producer = mock(SpawnAction.class);

  @Before
  public void setUp() {
    ActionOwner owner = mock(ActionOwner.class);
    when(owner.getLabel()).thenReturn(OWNER);
    when(testAction.getOwner()).thenReturn(owner);
    when(testAction.getExecutionSettings()).thenReturn(executionSettings);
    when(executionSettings.getExecutable()).thenReturn(executable);
    when(producer.getOwner()).thenReturn(owner);
    when(producer.getMnemonic()).thenReturn("GoLink");
    when(producer.getOutputs()).thenReturn(ImmutableList.of(executable));
    when(producer.getExecutionInfo()).thenReturn(ImmutableMap.of());
  }

  @Test
  public void eligibleGoLink() {
    assertThat(check(producer)).isInstanceOf(Eligible.class);
  }

  @Test
  public void noRemoteCacheIsEligible() {
    when(producer.getExecutionInfo())
        .thenReturn(ImmutableMap.of(ExecutionRequirements.NO_REMOTE_CACHE, "1"));

    assertThat(check(producer)).isInstanceOf(Eligible.class);
  }

  @Test
  public void noCacheIsIneligible() {
    when(producer.getExecutionInfo())
        .thenReturn(ImmutableMap.of(ExecutionRequirements.NO_CACHE, "1"));

    assertReason(check(producer), IneligibilityReason.PRODUCER_NO_CACHE);
  }

  @Test
  public void mnemonicMustBeAllowlisted() {
    when(producer.getMnemonic()).thenReturn("CppLink");

    assertReason(check(producer), IneligibilityReason.PRODUCER_MNEMONIC_NOT_ALLOWLISTED);
  }

  @Test
  public void executableMustBeDerived() {
    when(executionSettings.getExecutable()).thenReturn(null);

    assertReason(check(producer), IneligibilityReason.EXECUTABLE_NOT_DERIVED);
  }

  @Test
  public void producerMustBeSpawnAction() {
    Action nonSpawnProducer = mock(Action.class);
    when(nonSpawnProducer.getMnemonic()).thenReturn("GoLink");

    assertReason(check(nonSpawnProducer), IneligibilityReason.UNSUPPORTED_PRODUCER_ACTION_TYPE);
  }

  @Test
  public void inputDiscoveringProducerIsIneligible() {
    when(producer.discoversInputs()).thenReturn(true);

    assertReason(check(producer), IneligibilityReason.PRODUCER_DISCOVERS_INPUTS);
  }

  @Test
  public void producerOwnerMustMatch() {
    ActionOwner otherOwner = mock(ActionOwner.class);
    when(otherOwner.getLabel()).thenReturn(Label.parseCanonicalUnchecked("//other:test"));
    when(producer.getOwner()).thenReturn(otherOwner);

    assertReason(check(producer), IneligibilityReason.PRODUCER_OWNER_MISMATCH);
  }

  @Test
  public void producerOutputMustContainTestExecutable() {
    when(producer.getOutputs()).thenReturn(ImmutableList.of());

    assertReason(check(producer), IneligibilityReason.PRODUCER_OUTPUT_MISMATCH);
  }

  @Test
  public void producerMustHaveOnlyTheTestExecutableOutput() {
    when(producer.getOutputs())
        .thenReturn(ImmutableList.of(executable, mock(DerivedArtifact.class)));

    assertReason(check(producer), IneligibilityReason.PRODUCER_OUTPUT_MISMATCH);
  }

  @Test
  public void volatileProducerIsIneligible() {
    when(producer.isVolatile()).thenReturn(true);

    assertReason(check(producer), IneligibilityReason.PRODUCER_VOLATILE);
  }

  @Test
  public void unconditionallyExecutedProducerIsIneligible() {
    when(producer.executeUnconditionally()).thenReturn(true);

    assertReason(check(producer), IneligibilityReason.PRODUCER_VOLATILE);
  }

  private ProducerKeyedTestCacheEligibility.Result check(Action producerAction) {
    return ProducerKeyedTestCacheEligibility.check(
        testAction, producerAction, ImmutableSet.of("GoLink"));
  }

  private static void assertReason(
      ProducerKeyedTestCacheEligibility.Result result, IneligibilityReason reason) {
    assertThat(result).isInstanceOf(Ineligible.class);
    assertThat(((Ineligible) result).reason()).isEqualTo(reason);
  }
}
