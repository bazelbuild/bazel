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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;

/** Fail-closed eligibility checks for producer-keyed test cache key computation. */
public final class ProducerKeyedTestCacheEligibility {
  public enum IneligibilityReason {
    EXECUTABLE_NOT_DERIVED,
    PRODUCER_MNEMONIC_NOT_ALLOWLISTED,
    UNSUPPORTED_PRODUCER_ACTION_TYPE,
    PRODUCER_DISCOVERS_INPUTS,
    PRODUCER_OWNER_MISMATCH,
    PRODUCER_OUTPUT_MISMATCH,
    PRODUCER_NO_CACHE,
    PRODUCER_VOLATILE
  }

  public sealed interface Result permits Eligible, Ineligible {}

  public record Eligible(SpawnAction producer) implements Result {}

  public record Ineligible(IneligibilityReason reason) implements Result {}

  private ProducerKeyedTestCacheEligibility() {}

  public static Result check(
      TestRunnerAction testAction, Action producer, ImmutableSet<String> allowlistedMnemonics) {
    Artifact executable = testAction.getExecutionSettings().getExecutable();
    if (!(executable instanceof DerivedArtifact)) {
      return new Ineligible(IneligibilityReason.EXECUTABLE_NOT_DERIVED);
    }
    if (!allowlistedMnemonics.contains(producer.getMnemonic())) {
      return new Ineligible(IneligibilityReason.PRODUCER_MNEMONIC_NOT_ALLOWLISTED);
    }
    if (!(producer instanceof SpawnAction spawnAction)) {
      return new Ineligible(IneligibilityReason.UNSUPPORTED_PRODUCER_ACTION_TYPE);
    }
    if (producer.discoversInputs()) {
      return new Ineligible(IneligibilityReason.PRODUCER_DISCOVERS_INPUTS);
    }
    if (testAction.getOwner().getLabel() == null
        || !testAction.getOwner().getLabel().equals(producer.getOwner().getLabel())) {
      return new Ineligible(IneligibilityReason.PRODUCER_OWNER_MISMATCH);
    }
    if (producer.getOutputs().size() != 1 || !producer.getOutputs().contains(executable)) {
      return new Ineligible(IneligibilityReason.PRODUCER_OUTPUT_MISMATCH);
    }
    if (spawnAction.getExecutionInfo().containsKey(ExecutionRequirements.NO_CACHE)) {
      return new Ineligible(IneligibilityReason.PRODUCER_NO_CACHE);
    }
    if (producer.isVolatile() || producer.executeUnconditionally()) {
      return new Ineligible(IneligibilityReason.PRODUCER_VOLATILE);
    }
    return new Eligible(spawnAction);
  }
}
