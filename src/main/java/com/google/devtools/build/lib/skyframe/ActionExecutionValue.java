// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * A value representing an executed action.
 */
@Immutable
@ThreadSafe
public class ActionExecutionValue implements SkyValue {
  private final ImmutableMap<Artifact, FileValue> artifactData;
  private final ImmutableMap<Artifact, FileArtifactValue> additionalOutputData;

  /**
   * @param artifactData Map from Artifacts to corresponding FileValues.
   * @param additionalOutputData Map from Artifacts to values if the FileArtifactValue for this
   *     artifact cannot be derived from the corresponding FileValue (see {@link
   *     ActionMetadataHandler#getAdditionalOutputData} for when this is necessary).
   */
  ActionExecutionValue(Map<Artifact, FileValue> artifactData,
      Map<Artifact, FileArtifactValue> additionalOutputData) {
    this.artifactData = ImmutableMap.copyOf(artifactData);
    this.additionalOutputData = ImmutableMap.copyOf(additionalOutputData);
  }

  /**
   * Returns metadata for a given artifact, if that metadata cannot be inferred from the
   * corresponding {@link #getData} call for that Artifact. See {@link
   * ActionMetadataHandler#getAdditionalOutputData} for when that can happen.
   */
  @Nullable
  FileArtifactValue getArtifactValue(Artifact artifact) {
    return additionalOutputData.get(artifact);
  }

  /**
   * @return The data for each non-middleman output of this action, in the form of the {@link
   * FileValue} that would be created for the file if it were to be read from disk.
   */
  FileValue getData(Artifact artifact) {
    Preconditions.checkState(!additionalOutputData.containsKey(artifact),
        "Should not be requesting data for already-constructed FileArtifactValue: %s", artifact);
    return artifactData.get(artifact);
  }

  /**
   * @return The map from {@link Artifact} to the corresponding {@link FileValue} that would be
   * returned by {@link #getData}. Should only be needed by {@link FilesystemValueChecker}.
   */
  ImmutableMap<Artifact, FileValue> getAllOutputArtifactData() {
    return artifactData;
  }

  @ThreadSafe
  @VisibleForTesting
  public static SkyKey key(Action action) {
    return new SkyKey(SkyFunctions.ACTION_EXECUTION, action);
  }

  /**
   * Returns whether the key corresponds to a ActionExecutionValue worth reporting status about.
   *
   * <p>If an action can do real work, it's probably worth counting and reporting status about.
   * Actions that don't really do any work (typically middleman actions) should not be counted
   * towards enqueued and completed actions.
   */
  public static boolean isReportWorthyAction(SkyKey key) {
    return key.functionName().equals(SkyFunctions.ACTION_EXECUTION)
        && isReportWorthyAction((Action) key.argument());
  }

  /**
   * Returns whether the action is worth reporting status about.
   *
   * <p>If an action can do real work, it's probably worth counting and reporting status about.
   * Actions that don't really do any work (typically middleman actions) should not be counted
   * towards enqueued and completed actions.
   */
  public static boolean isReportWorthyAction(Action action) {
    return action.getActionType() == MiddlemanType.NORMAL;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("artifactData", artifactData)
        .add("additionalOutputData", additionalOutputData)
        .toString();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ActionExecutionValue)) {
      return false;
    }
    ActionExecutionValue o = (ActionExecutionValue) obj;
    return artifactData.equals(o.artifactData)
        && additionalOutputData.equals(o.additionalOutputData);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(artifactData, additionalOutputData);
  }
}
