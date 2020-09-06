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
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue.ArchivedRepresentation;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.BiFunction;
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
      Preconditions.checkArgument(
          !entry.getKey().isChildOfDeclaredDirectory(),
          "%s should only be stored in a TreeArtifactValue",
          entry.getKey());
      if (entry.getValue().getType().isFile()) {
        Preconditions.checkNotNull(
            entry.getValue().getDigest(), "missing digest for %s", entry.getKey());
      }
    }

    for (Map.Entry<Artifact, TreeArtifactValue> tree : treeArtifactData.entrySet()) {
      TreeArtifactValue treeArtifact = tree.getValue();
      if (TreeArtifactValue.OMITTED_TREE_MARKER.equals(treeArtifact)) {
        continue;
      }
      for (Map.Entry<TreeFileArtifact, FileArtifactValue> file :
          treeArtifact.getChildValues().entrySet()) {
        // Tree artifacts can contain symlinks to directories, which don't have a digest.
        if (file.getValue().getType().isFile()) {
          Preconditions.checkNotNull(
              file.getValue().getDigest(),
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
   * Retrieves a {@link FileArtifactValue} for a regular (non-tree) derived artifact.
   *
   * <p>The value for the given artifact must be present, or else {@link NullPointerException} will
   * be thrown.
   */
  public FileArtifactValue getExistingFileArtifactValue(Artifact artifact) {
    Preconditions.checkArgument(
        artifact instanceof DerivedArtifact && !artifact.isTreeArtifact(),
        "Cannot request %s from %s",
        artifact,
        this);

    FileArtifactValue result;
    if (artifact.isChildOfDeclaredDirectory()) {
      TreeArtifactValue tree = treeArtifactData.get(artifact.getParent());
      result = tree == null ? null : tree.getChildValues().get(artifact);
    } else if (artifact instanceof ArchivedTreeArtifact) {
      TreeArtifactValue tree = treeArtifactData.get(artifact.getParent());
      ArchivedRepresentation archivedRepresentation =
          tree.getArchivedRepresentation()
              .orElseThrow(
                  () -> new NoSuchElementException("Missing archived representation in: " + tree));
      Preconditions.checkArgument(
          archivedRepresentation.archivedTreeFileArtifact().equals(artifact),
          "Multiple archived tree artifacts for: %s",
          artifact.getParent());
      result = archivedRepresentation.archivedFileValue();
    } else {
      result = artifactData.get(artifact);
    }

    return Preconditions.checkNotNull(
        result,
        "Missing artifact %s (generating action key %s) in %s",
        artifact,
        ((DerivedArtifact) artifact).getGeneratingActionKey(),
        this);
  }

  TreeArtifactValue getTreeArtifactValue(Artifact artifact) {
    Preconditions.checkArgument(artifact.isTreeArtifact());
    return treeArtifactData.get(artifact);
  }

  /**
   * Returns a map containing all artifacts output by the action, except for tree artifacts which
   * are accesible via {@link #getAllTreeArtifactValues}.
   *
   * <p>Primarily needed by {@link FilesystemValueChecker}. Also called by {@link ArtifactFunction}
   * when aggregating a {@link TreeArtifactValue} out of action template expansion outputs.
   */
  // Visible only for testing: should be package-private.
  public ImmutableMap<Artifact, FileArtifactValue> getAllFileValues() {
    return artifactData;
  }

  /**
   * Returns a map containing all tree artifacts output by the action.
   *
   * <p>Should only be needed by {@link FilesystemValueChecker}.
   */
  // Visible only for testing: should be package-private.
  public ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactValues() {
    return treeArtifactData;
  }

  @Nullable
  public ImmutableList<FilesetOutputSymlink> getOutputSymlinks() {
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

  private static <V> ImmutableMap<Artifact, V> transformMap(
      ImmutableMap<Artifact, V> data,
      Map<OwnerlessArtifactWrapper, Artifact> newArtifactMap,
      BiFunction<Artifact, V, V> transform) {
    if (data.isEmpty()) {
      return data;
    }

    ImmutableMap.Builder<Artifact, V> result = ImmutableMap.builderWithExpectedSize(data.size());
    for (Map.Entry<Artifact, V> entry : data.entrySet()) {
      Artifact artifact = entry.getKey();
      Artifact newArtifact =
          Preconditions.checkNotNull(
              newArtifactMap.get(new OwnerlessArtifactWrapper(artifact)),
              "Output artifact %s from one shared action not present in another's outputs (%s)",
              artifact,
              newArtifactMap);
      result.put(newArtifact, transform.apply(newArtifact, entry.getValue()));
    }
    return result.build();
  }

  /** Transforms the children of a {@link TreeArtifactValue} so that owners are consistent. */
  private static TreeArtifactValue transformSharedTree(
      Artifact newArtifact, TreeArtifactValue tree) {
    Preconditions.checkState(
        newArtifact.isTreeArtifact(), "Expected tree artifact, got %s", newArtifact);

    if (TreeArtifactValue.OMITTED_TREE_MARKER.equals(tree)) {
      return TreeArtifactValue.OMITTED_TREE_MARKER;
    }

    SpecialArtifact newParent = (SpecialArtifact) newArtifact;
    TreeArtifactValue.Builder newTree = TreeArtifactValue.newBuilder(newParent);

    for (Map.Entry<TreeFileArtifact, FileArtifactValue> child : tree.getChildValues().entrySet()) {
      newTree.putChild(
          TreeFileArtifact.createTreeOutput(newParent, child.getKey().getParentRelativePath()),
          child.getValue());
    }

    tree.getArchivedRepresentation()
        .ifPresent(
            archivedRepresentation -> {
              newTree.setArchivedRepresentation(
                  new ArchivedTreeArtifact(
                      newParent,
                      archivedRepresentation.archivedTreeFileArtifact().getRoot(),
                      archivedRepresentation.archivedTreeFileArtifact().getExecPath()),
                  archivedRepresentation.archivedFileValue());
            });

    return newTree.build();
  }

  ActionExecutionValue transformForSharedAction(ImmutableSet<Artifact> outputs) {
    Map<OwnerlessArtifactWrapper, Artifact> newArtifactMap =
        Maps.uniqueIndex(outputs, OwnerlessArtifactWrapper::new);
    // This is only called for shared actions, so we'll almost certainly have to transform all keys
    // in all sets.
    // Discovered modules come from the action's inputs, and so don't need to be transformed.
    return create(
        transformMap(artifactData, newArtifactMap, (newArtifact, value) -> value),
        transformMap(treeArtifactData, newArtifactMap, ActionExecutionValue::transformSharedTree),
        outputSymlinks,
        discoveredModules,
        this instanceof CrossServerUnshareableActionExecutionValue);
  }
}
