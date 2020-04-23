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

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A value representing an executed action. */
@Immutable
@ThreadSafe
public class ActionExecutionValue implements SkyValue {

  /** A map from each output artifact of this action to their {@link FileArtifactValue}s. */
  private final ImmutableMap<Artifact, FileArtifactValue> artifactData;

  /** The TreeArtifactValue of all TreeArtifacts output by this Action. */
  private final ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData;

  @Nullable private final ImmutableList<FilesetOutputSymlink> outputSymlinks;

  @Nullable private final NestedSet<Artifact> discoveredModules;

  /**
   * @param artifactData Map from Artifacts to corresponding {@link FileArtifactValue}.
   * @param treeArtifactData All tree artifact data.
   * @param outputSymlinks This represents the SymlinkTree which is the output of a fileset action.
   * @param discoveredModules cpp modules discovered
   */
  private ActionExecutionValue(
      Map<Artifact, FileArtifactValue> artifactData,
      Map<Artifact, TreeArtifactValue> treeArtifactData,
      @Nullable ImmutableList<FilesetOutputSymlink> outputSymlinks,
      @Nullable NestedSet<Artifact> discoveredModules) {
    for (Map.Entry<Artifact, FileArtifactValue> entry : artifactData.entrySet()) {
      if (entry.getValue().getType() == FileStateType.REGULAR_FILE) {
        Preconditions.checkArgument(
            entry.getValue().getDigest() != null, "missing digest for %s", entry.getKey());
      }
    }

    for (Map.Entry<Artifact, TreeArtifactValue> tree : treeArtifactData.entrySet()) {
      for (Map.Entry<TreeFileArtifact, FileArtifactValue> file :
          tree.getValue().getChildValues().entrySet()) {
        // We should only have RegularFileValue instances in here, but apparently tree artifacts
        // sometimes store their own root directory in here. Sad.
        // https://github.com/bazelbuild/bazel/issues/9058
        if (file.getValue().getType() == FileStateType.REGULAR_FILE) {
          Preconditions.checkArgument(
              file.getValue().getDigest() != null,
              "missing digest for file %s in tree artifact %s",
              file.getKey(),
              tree.getKey());
        }
      }
    }

    this.artifactData = ImmutableMap.copyOf(artifactData);
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
        outputSymlinks,
        discoveredModules,
        actionDependsOnBuildId);
  }

  static ActionExecutionValue create(
      Map<Artifact, FileArtifactValue> artifactData,
      Map<Artifact, TreeArtifactValue> treeArtifactData,
      @Nullable ImmutableList<FilesetOutputSymlink> outputSymlinks,
      @Nullable NestedSet<Artifact> discoveredModules,
      boolean actionDependsOnBuildId) {
    return actionDependsOnBuildId
        ? new CrossServerUnshareableActionExecutionValue(
            artifactData, treeArtifactData, outputSymlinks, discoveredModules)
        : new ActionExecutionValue(
            artifactData, treeArtifactData, outputSymlinks, discoveredModules);
  }

  /**
   * Create {@link FileArtifactValue} for artifact that must be non-middleman non-tree derived
   * artifact.
   */
  static FileArtifactValue createSimpleFileArtifactValue(
      Artifact.DerivedArtifact artifact, ActionExecutionValue actionValue) {
    Preconditions.checkState(!artifact.isMiddlemanArtifact(), "%s %s", artifact, actionValue);
    Preconditions.checkState(!artifact.isTreeArtifact(), "%s %s", artifact, actionValue);
    return Preconditions.checkNotNull(
        actionValue.getArtifactValue(artifact),
        "%s %s %s",
        artifact,
        artifact.getGeneratingActionKey(),
        actionValue);
  }

  /**
   * @return The data for each non-middleman output of this action, in the form of the {@link
   *     com.google.devtools.build.lib.actions.FileValue} that would be created for the file if it
   *     were to be read from disk.
   */
  @Nullable
  public FileArtifactValue getArtifactValue(Artifact artifact) {
    FileArtifactValue result = artifactData.get(artifact);
    if (result != null || !artifact.hasParent()) {
      return result;
    }
    // In some cases, TreeFileArtifact metadata may not have been injected directly, and is only
    // available via the parent. However, if this ActionExecutionValue corresponds to a templated
    // action, as opposed to an action that created a tree artifact itself, the TreeFileArtifact
    // metadata will be in artifactData, since this value will have no treeArtifactData.
    TreeArtifactValue treeArtifactValue = treeArtifactData.get(artifact.getParent());
    return treeArtifactValue == null ? null : treeArtifactValue.getChildValues().get(artifact);
  }

  TreeArtifactValue getTreeArtifactValue(Artifact artifact) {
    Preconditions.checkArgument(artifact.isTreeArtifact());
    return treeArtifactData.get(artifact);
  }

  /**
   * @return The map from {@link Artifact}s to the corresponding {@link
   *     com.google.devtools.build.lib.actions.FileValue}s that would be returned by {@link
   *     #getArtifactValue}. Primarily needed by {@link FilesystemValueChecker}, also called by
   *     {@link ArtifactFunction} when aggregating a {@link TreeArtifactValue}.
   */
  Map<Artifact, FileArtifactValue> getAllFileValues() {
    return artifactData;
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

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("artifactData", artifactData)
        .add("treeArtifactData", treeArtifactData)
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
        && (outputSymlinks == null || outputSymlinks.equals(o.outputSymlinks));
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(artifactData, treeArtifactData);
  }

  /**
   * Subclass that reports this value cannot be shared across servers. Note that this is unrelated
   * to the concept of shared actions.
   */
  private static final class CrossServerUnshareableActionExecutionValue
      extends ActionExecutionValue {
    CrossServerUnshareableActionExecutionValue(
        Map<Artifact, FileArtifactValue> artifactData,
        Map<Artifact, TreeArtifactValue> treeArtifactData,
        @Nullable ImmutableList<FilesetOutputSymlink> outputSymlinks,
        @Nullable NestedSet<Artifact> discoveredModules) {
      super(artifactData, treeArtifactData, outputSymlinks, discoveredModules);
    }

    @Override
    public boolean dataIsShareable() {
      return false;
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
        outputs.stream()
            .collect(Collectors.toMap(OwnerlessArtifactWrapper::new, Function.identity()));
    // This is only called for shared actions, so we'll almost certainly have to transform all keys
    // in all sets.
    // Discovered modules come from the action's inputs, and so don't need to be transformed.
    return create(
        transformKeys(artifactData, newArtifactMap),
        transformKeys(treeArtifactData, newArtifactMap),
        outputSymlinks,
        discoveredModules,
        this instanceof CrossServerUnshareableActionExecutionValue);
  }
}
