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
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.ArtifactFileMetadata;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.UnshareableValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * A value representing an executed action.
 *
 * <p>Concerning the data in this class:
 *
 * <p>We want to track all output data from an ActionExecutionValue. See {@link OutputStore} for a
 * discussion of how its fields {@link OutputStore#artifactData} and {@link
 * OutputStore#additionalOutputData} relate. The relationship is the same between {@link
 * #artifactData} and {@link #additionalOutputData} here.
 */
@Immutable
@ThreadSafe
public class ActionExecutionValue implements SkyValue {

  /**
   * The metadata of all files for this ActionExecutionValue whose filesystem metadata differs from
   * the metadata in {@link #additionalOutputData} or is not present there. This metadata can be
   * checked quickly against the actual filesystem on incremental builds.
   *
   * <p>If additional metadata is not needed here over what is in {@link #additionalOutputData}, the
   * value will be {@link ArtifactFileMetadata#PLACEHOLDER}.
   */
  private final ImmutableMap<Artifact, ArtifactFileMetadata> artifactData;

  /** The TreeArtifactValue of all TreeArtifacts output by this Action. */
  private final ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData;

  /**
   * Contains all remaining data that weren't in the above maps. See {@link
   * OutputStore#getAllAdditionalOutputData}.
   */
  private final ImmutableMap<Artifact, FileArtifactValue> additionalOutputData;

  @Nullable private final ImmutableList<FilesetOutputSymlink> outputSymlinks;

  @Nullable private final NestedSet<Artifact> discoveredModules;

  /**
   * @param artifactData Map from Artifacts to corresponding {@link ArtifactFileMetadata}.
   * @param treeArtifactData All tree artifact data.
   * @param additionalOutputData Map from Artifacts to values if the FileArtifactValue for this
   *     artifact cannot be derived from the corresponding FileValue (see {@link
   *     OutputStore#getAllAdditionalOutputData} for when this is necessary). These output data are
   *     not used by the {@link FilesystemValueChecker} to invalidate ActionExecutionValues.
   * @param outputSymlinks This represents the SymlinkTree which is the output of a fileset action.
   * @param discoveredModules cpp modules discovered
   */
  private ActionExecutionValue(
      Map<Artifact, ArtifactFileMetadata> artifactData,
      Map<Artifact, TreeArtifactValue> treeArtifactData,
      Map<Artifact, FileArtifactValue> additionalOutputData,
      @Nullable ImmutableList<FilesetOutputSymlink> outputSymlinks,
      @Nullable NestedSet<Artifact> discoveredModules) {
    this.artifactData = ImmutableMap.copyOf(artifactData);
    this.additionalOutputData = ImmutableMap.copyOf(additionalOutputData);
    this.treeArtifactData = ImmutableMap.copyOf(treeArtifactData);
    this.outputSymlinks = outputSymlinks;
    this.discoveredModules = discoveredModules;
  }

  static ActionExecutionValue createFromOutputStore(
      OutputStore outputStore,
      @Nullable ImmutableList<FilesetOutputSymlink> outputSymlinks,
      @Nullable NestedSet<Artifact> discoveredModules,
      boolean actionDependsOnBuildId) {
    return create(
        outputStore.getAllArtifactData(),
        outputStore.getAllTreeArtifactData(),
        outputStore.getAllAdditionalOutputData(),
        outputSymlinks,
        discoveredModules,
        actionDependsOnBuildId);
  }

  static ActionExecutionValue create(
      Map<Artifact, ArtifactFileMetadata> artifactData,
      Map<Artifact, TreeArtifactValue> treeArtifactData,
      Map<Artifact, FileArtifactValue> additionalOutputData,
      @Nullable ImmutableList<FilesetOutputSymlink> outputSymlinks,
      @Nullable NestedSet<Artifact> discoveredModules,
      boolean actionDependsOnBuildId) {
    return actionDependsOnBuildId
        ? new CrossServerUnshareableActionExecutionValue(
            artifactData, treeArtifactData, additionalOutputData, outputSymlinks, discoveredModules)
        : new ActionExecutionValue(
            artifactData,
            treeArtifactData,
            additionalOutputData,
            outputSymlinks,
            discoveredModules);
  }

  /**
   * Returns metadata for a given artifact, if that metadata cannot be inferred from the
   * corresponding {@link #getData} call for that Artifact. See {@link
   * OutputStore#getAllAdditionalOutputData} for when that can happen.
   */
  @Nullable
  public FileArtifactValue getArtifactValue(Artifact artifact) {
    return additionalOutputData.get(artifact);
  }

  /**
   * @return The data for each non-middleman output of this action, in the form of the {@link
   *     FileValue} that would be created for the file if it were to be read from disk.
   */
  ArtifactFileMetadata getData(Artifact artifact) {
    Preconditions.checkState(!additionalOutputData.containsKey(artifact),
        "Should not be requesting data for already-constructed FileArtifactValue: %s", artifact);
    return artifactData.get(artifact);
  }

  TreeArtifactValue getTreeArtifactValue(Artifact artifact) {
    Preconditions.checkArgument(artifact.isTreeArtifact());
    return treeArtifactData.get(artifact);
  }

  /**
   * @return The map from {@link Artifact}s to the corresponding {@link FileValue}s that would be
   *     returned by {@link #getData}. Primarily needed by {@link FilesystemValueChecker}, also
   *     called by {@link ArtifactFunction} when aggregating a {@link TreeArtifactValue}.
   */
  Map<Artifact, ArtifactFileMetadata> getAllFileValues() {
    return Maps.transformEntries(artifactData, this::transformIfPlaceholder);
  }

  /**
   * @return The map from {@link Artifact}s to the corresponding {@link TreeArtifactValue}s that
   *     would be returned by {@link #getTreeArtifactValue}. Should only be needed by {@link
   *     FilesystemValueChecker}.
   */
  ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactValues() {
    return treeArtifactData;
  }

  @Nullable
  ImmutableList<FilesetOutputSymlink> getOutputSymlinks() {
    return outputSymlinks;
  }

  @Nullable
  public NestedSet<Artifact> getDiscoveredModules() {
    return discoveredModules;
  }

  /**
   * @param lookupKey A {@link SkyKey} whose argument is an {@code ActionLookupKey}, whose
   *     corresponding {@code ActionLookupValue} contains the action to be executed.
   * @param index the index of the action to be executed in the {@code ActionLookupValue}, to be
   *     passed to {@code ActionLookupValue#getAction}.
   */
  @ThreadSafe
  @VisibleForTesting
  public static ActionLookupData key(ActionLookupValue.ActionLookupKey lookupKey, int index) {
    return ActionLookupData.create(lookupKey, index);
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
    if (obj == null) {
      return false;
    }
    if (!obj.getClass().equals(getClass())) {
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

  /** Transforms PLACEHOLDER values into normal instances. */
  private ArtifactFileMetadata transformIfPlaceholder(
      Artifact artifact, ArtifactFileMetadata value) {
    if (value == ArtifactFileMetadata.PLACEHOLDER) {
      FileArtifactValue metadata =
          Preconditions.checkNotNull(
              additionalOutputData.get(artifact),
              "Placeholder without corresponding FileArtifactValue for: %s",
              artifact);
      FileStateValue.RegularFileStateValue fileStateValue =
          new FileStateValue.RegularFileStateValue(
              metadata.getSize(), metadata.getDigest(), /*contentsProxy=*/ null);
      PathFragment pathFragment = artifact.getPath().asFragment();
      return ArtifactFileMetadata.value(pathFragment, fileStateValue, pathFragment, fileStateValue);
    }
    return value;
  }

  /**
   * Marker subclass that indicates this value cannot be shared across servers. Note that this is
   * unrelated to the concept of shared actions.
   */
  private static class CrossServerUnshareableActionExecutionValue extends ActionExecutionValue
      implements UnshareableValue {
    CrossServerUnshareableActionExecutionValue(
        Map<Artifact, ArtifactFileMetadata> artifactData,
        Map<Artifact, TreeArtifactValue> treeArtifactData,
        Map<Artifact, FileArtifactValue> additionalOutputData,
        @Nullable ImmutableList<FilesetOutputSymlink> outputSymlinks,
        @Nullable NestedSet<Artifact> discoveredModules) {
      super(
          artifactData, treeArtifactData, additionalOutputData, outputSymlinks, discoveredModules);
    }
  }

  private static <V> ImmutableMap<Artifact, V> transformKeys(
      ImmutableMap<Artifact, V> data, Map<OwnerlessArtifactWrapper, Artifact> newArtifactMap) {
    if (data.isEmpty()) {
      return data;
    }
    ImmutableMap.Builder<Artifact, V> result = ImmutableMap.builderWithExpectedSize(data.size());
    for (Map.Entry<Artifact, V> entry : data.entrySet()) {
      Artifact artifact = entry.getKey();
      Artifact transformedArtifact =
          newArtifactMap.get(new OwnerlessArtifactWrapper(entry.getKey()));
      if (transformedArtifact == null) {
        // If this action generated a tree artifact, then the declared outputs of the action will
        // not include the contents of the directory corresponding to that artifact, but the
        // contents are present in this ActionExecutionValue as TreeFileArtifacts. We must create
        // corresponding artifacts in the shared action's ActionExecutionValue. We can do that since
        // a TreeFileArtifact is uniquely described by its parent, its owner, and its parent-
        // relative path. Since the child was not a declared output, the child and parent must be
        // generated by the same action, hence they have the same owner, and the parent was a
        // declared output, so it is present in the shared action. Then we can create the new
        // TreeFileArtifact to have the shared action's version of the parent artifact (instead of
        // the original parent artifact); the same parent-relative path; and the new parent's
        // ArtifactOwner.
        Preconditions.checkState(
            artifact.hasParent(),
            "Output artifact %s from one shared action not present in another's outputs (%s)",
            artifact,
            newArtifactMap);
        ArtifactOwner childOwner = artifact.getArtifactOwner();
        Artifact parent = Preconditions.checkNotNull(artifact.getParent(), artifact);
        ArtifactOwner parentOwner = parent.getArtifactOwner();
        Preconditions.checkState(
            parentOwner.equals(childOwner),
            "A parent tree artifact %s has a different ArtifactOwner (%s) than its child %s (owned "
                + "by %s), but both artifacts were generated by the same action",
            parent,
            parentOwner,
            artifact,
            childOwner);
        Artifact newParent =
            Preconditions.checkNotNull(
                newArtifactMap.get(new OwnerlessArtifactWrapper(parent)),
                "parent %s of %s was not present in shared action's data (%s)",
                parent,
                artifact,
                newArtifactMap);
        transformedArtifact =
            ActionInputHelper.treeFileArtifact(
                (Artifact.SpecialArtifact) newParent, artifact.getParentRelativePath());
      }
      result.put(transformedArtifact, entry.getValue());
    }
    return result.build();
  }

  ActionExecutionValue transformForSharedAction(ImmutableSet<Artifact> outputs) {
    Map<OwnerlessArtifactWrapper, Artifact> newArtifactMap =
        outputs
            .stream()
            .collect(Collectors.toMap(OwnerlessArtifactWrapper::new, Function.identity()));
    // This is only called for shared actions, so we'll almost certainly have to transform all keys
    // in all sets.
    // Discovered modules come from the action's inputs, and so don't need to be transformed.
    return create(
        transformKeys(artifactData, newArtifactMap),
        transformKeys(treeArtifactData, newArtifactMap),
        transformKeys(additionalOutputData, newArtifactMap),
        outputSymlinks,
        discoveredModules,
        this instanceof CrossServerUnshareableActionExecutionValue);
  }
}
