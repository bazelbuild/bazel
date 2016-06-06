// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
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

  private static final Function<Action, SkyKey> TO_KEY =
      new Function<Action, SkyKey>() {
        @Override
        public SkyKey apply(Action action) {
          return key(action);
        }
      };

  /*
  Concerning the data in this class:

  We want to track all output data from an ActionExecutionValue. However, we want to separate
  quickly-accessible Filesystem data from other kinds of data. We use FileValues
  to represent data that may be quickly accessed, TreeArtifactValues to give us directory contents,
  and FileArtifactValues inside TreeArtifactValues or the additionalOutputData map
  to give us full mtime/digest information on all output files.

  The reason for this separation is so that FileSystemValueChecker remains fast. When it checks
  the validity of an ActionExecutionValue, it only checks the quickly-accessible data stored
  in FileValues and TreeArtifactValues.
   */

  /**
   * The FileValues of all files for this ActionExecutionValue. These FileValues can be
   * read and checked quickly from the filesystem, unlike FileArtifactValues.
   */
  private final ImmutableMap<Artifact, FileValue> artifactData;

  /** The TreeArtifactValue of all TreeArtifacts output by this Action. */
  private final ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData;

  /**
   * Contains all remaining data that weren't in the above maps. See
   * {@link ActionMetadataHandler#getAdditionalOutputData}.
   */
  private final ImmutableMap<Artifact, FileArtifactValue> additionalOutputData;

  /**
   * @param artifactData Map from Artifacts to corresponding FileValues.
   * @param treeArtifactData All tree artifact data.
   * @param additionalOutputData Map from Artifacts to values if the FileArtifactValue for this
   *     artifact cannot be derived from the corresponding FileValue (see {@link
   *     ActionMetadataHandler#getAdditionalOutputData} for when this is necessary).
   *     These output data are not used by the {@link FilesystemValueChecker}
   *     to invalidate ActionExecutionValues.
   */
  ActionExecutionValue(
      Map<Artifact, FileValue> artifactData,
      Map<Artifact, TreeArtifactValue> treeArtifactData,
      Map<Artifact, FileArtifactValue> additionalOutputData) {
    this.artifactData = ImmutableMap.<Artifact, FileValue>copyOf(artifactData);
    this.additionalOutputData = ImmutableMap.copyOf(additionalOutputData);
    this.treeArtifactData = ImmutableMap.copyOf(treeArtifactData);
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

  TreeArtifactValue getTreeArtifactValue(Artifact artifact) {
    Preconditions.checkArgument(artifact.isTreeArtifact());
    return treeArtifactData.get(artifact);
  }

  /**
   * @return The map from {@link Artifact}s to the corresponding {@link FileValue}s that would
   * be returned by {@link #getData}. Should only be needed by {@link FilesystemValueChecker}.
   */
  ImmutableMap<Artifact, FileValue> getAllFileValues() {
    return artifactData;
  }

  /**
   * @return The map from {@link Artifact}s to the corresponding {@link TreeArtifactValue}s that
   * would be returned by {@link #getTreeArtifactValue}. Should only be needed by
   * {@link FilesystemValueChecker}.
   */
  ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactValues() {
    return treeArtifactData;
  }

  @ThreadSafe
  @VisibleForTesting
  public static SkyKey key(Action action) {
    return SkyKey.create(SkyFunctions.ACTION_EXECUTION, action);
  }

  static Iterable<SkyKey> keys(Iterable<Action> actions) {
    return Iterables.transform(actions, TO_KEY);
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
        .add("treeArtifactData", treeArtifactData)
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
        && treeArtifactData.equals(o.treeArtifactData)
        && additionalOutputData.equals(o.additionalOutputData);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(artifactData, treeArtifactData, additionalOutputData);
  }
}
