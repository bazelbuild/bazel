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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/** Tracks whether an artifact was "consumed" by any action in the build. */
public final class ConsumedArtifactsTracker implements EphemeralCheckIfOutputConsumed {

  private final Set<Artifact> consumed = ConcurrentHashMap.newKeySet(524288);
  private final Set<Artifact> middlemanArtifactSkippedRegistering =
      ConcurrentHashMap.newKeySet(2048);
  private final Set<Artifact> middlemanArtifactBackFilled = ConcurrentHashMap.newKeySet(2048);

  private final Supplier<MemoizingEvaluator> evaluatorSupplier;

  public ConsumedArtifactsTracker(Supplier<MemoizingEvaluator> evaluatorSupplier) {
    this.evaluatorSupplier = evaluatorSupplier;
  }

  /**
   * This method guarantees the best estimate before the execution of the action that would generate
   * this artifact. The return value is undefined afterwards.
   */
  @Override
  public boolean test(Artifact artifact) {
    return consumed.contains(artifact);
  }

  void unregisterOutputsAfterExecutionDone(Collection<Artifact> outputs) {
    consumed.removeAll(outputs);
  }

  /**
   * Register the provided artifact as "consumed".
   *
   * <p>If the provided artifact is a middleman artifact, expand it and register the underlying
   * artifacts.
   */
  void registerConsumedArtifact(Artifact artifact) throws InterruptedException {
    if (artifact.isMiddlemanArtifact()
        && wasRegistrationSkippedForArtifactsUnderMiddleman(artifact)) {
      // Special case: this means the action that generates this middleman was already evaluated as
      // a top-level middleman action and the registration of its underlying artifacts were skipped.
      // We therefore need to do it again here.
      backfillArtifactsUnderMiddleman((DerivedArtifact) artifact);
      return;
    }
    storeConsumedStatusIfRequired(artifact);
  }

  private void storeConsumedStatusIfRequired(Artifact artifact) {
    if (shouldStoreConsumedStatus(artifact)) {
      consumed.add(artifact);
    }
  }

  void skipRegisteringArtifactsUnderMiddleman(Artifact middlemanArtifact) {
    middlemanArtifactSkippedRegistering.add(middlemanArtifact);
  }

  boolean wasRegistrationSkippedForArtifactsUnderMiddleman(Artifact middlemanArtifact) {
    return middlemanArtifactSkippedRegistering.contains(middlemanArtifact);
  }

  // TODO(b/304440811) Remove this after we removed the concept of middleman.
  private void backfillArtifactsUnderMiddleman(DerivedArtifact middlemanArtifact)
      throws InterruptedException {
    Set<Artifact> toBackfill = new HashSet<>();

    recursivelyCollectArtifactsToBackfill(middlemanArtifact, toBackfill);

    for (Artifact expandedInput : toBackfill) {
      storeConsumedStatusIfRequired(expandedInput);
    }
  }

  // In case we're backfilling for a middleman artifact that includes other middleman artifacts.
  private void recursivelyCollectArtifactsToBackfill(
      DerivedArtifact middlemanArtifact, Set<Artifact> toBackfill) throws InterruptedException {
    // We only need to do the backfilling once for each middleman artifact.
    if (!middlemanArtifactBackFilled.add(middlemanArtifact)) {
      return;
    }

    var generatingActionKey = middlemanArtifact.getGeneratingActionKey();
    // Avoid establishing a skyframe dependency.
    ActionLookupValue actionLookupValue =
        (ActionLookupValue)
            Preconditions.checkNotNull(
                    evaluatorSupplier
                        .get()
                        .getExistingEntryAtCurrentlyEvaluatingVersion(
                            generatingActionKey.getActionLookupKey()))
                .getValue();
    Action middlemanAction = actionLookupValue.getAction(generatingActionKey.getActionIndex());

    for (Artifact expandedInput : middlemanAction.getInputs().toList()) {
      if (expandedInput.isMiddlemanArtifact()) {
        recursivelyCollectArtifactsToBackfill((DerivedArtifact) expandedInput, toBackfill);
      } else if (shouldStoreConsumedStatus(expandedInput)) {
        toBackfill.add(expandedInput);
      }
    }
  }

  /**
   * Return whether the consumed status of an artifact should be recorded at all.
   *
   * <p>We should only store the consumed status of artifacts that will later on be checked for
   * orphaned status directly. This is an optimization to keep the set smaller.
   */
  private static boolean shouldStoreConsumedStatus(Artifact artifact) {
    return !(artifact.isSourceArtifact() // Source artifacts won't be orphaned.
        || artifact.hasParent()); // Will be checked through the parent artifact.
  }
}
