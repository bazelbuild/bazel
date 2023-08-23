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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.rules.cpp.IncludeScannable;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue.ArchivedRepresentation;
import com.google.devtools.build.lib.util.ClassName;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import java.util.Collection;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.BiFunction;
import javax.annotation.Nullable;

/** A value representing an executed action. */
@Immutable
@ThreadSafe
public abstract class ActionExecutionValue implements SkyValue {

  private ActionExecutionValue() {}

  @VisibleForTesting // All non-test usage should go through createFromOutputStore.
  public static ActionExecutionValue create(
      ImmutableMap<Artifact, FileArtifactValue> artifactData,
      ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData,
      ImmutableList<FilesetOutputSymlink> outputSymlinks,
      NestedSet<Artifact> discoveredModules) {
    // Use forEach instead of entrySet to avoid instantiating an EntrySet in ImmutableMap.
    artifactData.forEach(
        (artifact, value) -> {
          checkArgument(
              !artifact.isChildOfDeclaredDirectory(),
              "%s should only be stored in a TreeArtifactValue",
              artifact);
          checkArgument(
              !value.getType().isFile() || value.getDigest() != null,
              "Missing digest for %s",
              artifact);
        });
    treeArtifactData.forEach(
        (tree, treeValue) -> {
          if (TreeArtifactValue.OMITTED_TREE_MARKER.equals(treeValue)) {
            return;
          }
          treeValue
              .getChildValues()
              .forEach(
                  (child, childValue) ->
                      // Ignore symlinks to directories, which don't have a digest.
                      checkArgument(
                          !childValue.getType().isFile() || childValue.getDigest() != null,
                          "Missing digest for file %s in tree artifact %s",
                          child,
                          tree));
        });

    if (!outputSymlinks.isEmpty()) {
      checkArgument(
          artifactData.size() == 1,
          "Fileset actions should have a single output file (the manifest): %s",
          artifactData);
      checkArgument(
          treeArtifactData.isEmpty(),
          "Fileset actions do not output tree artifacts: %s",
          treeArtifactData);
      checkArgument(
          discoveredModules.isEmpty(),
          "Fileset actions do not discover modules: %s",
          discoveredModules);
      return new Fileset(artifactData, outputSymlinks);
    }

    if (!discoveredModules.isEmpty()) {
      checkArgument(
          artifactData.size() == 1,
          "Module-discovering actions should have a single output file (the .pcm file): %s",
          artifactData);
      checkArgument(
          treeArtifactData.isEmpty(),
          "Module-discovering actions do not output tree artifacts: %s",
          treeArtifactData);
      return new ModuleDiscovering(artifactData, discoveredModules);
    }

    if (!treeArtifactData.isEmpty()) {
      return treeArtifactData.size() == 1 && artifactData.isEmpty()
          ? new SingleTree(treeArtifactData)
          : new MultiTree(artifactData, treeArtifactData);
    }

    checkArgument(!artifactData.isEmpty(), "No outputs");
    return artifactData.size() == 1
        ? new SingleOutputFile(artifactData)
        : new MultiOutputFile(artifactData);
  }

  static ActionExecutionValue createFromOutputStore(
      OutputStore outputStore, ImmutableList<FilesetOutputSymlink> outputSymlinks, Action action) {
    return create(
        outputStore.getAllArtifactData(),
        outputStore.getAllTreeArtifactData(),
        outputSymlinks,
        action instanceof IncludeScannable
            ? ((IncludeScannable) action).getDiscoveredModules()
            : NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  /**
   * Retrieves a {@link FileArtifactValue} for a regular (non-tree) derived artifact.
   *
   * <p>The value for the given artifact must be present, or else {@link NullPointerException} will
   * be thrown.
   */
  public FileArtifactValue getExistingFileArtifactValue(Artifact artifact) {
    checkArgument(
        artifact instanceof DerivedArtifact && !artifact.isTreeArtifact(),
        "Cannot request %s from %s",
        artifact,
        this);

    FileArtifactValue result;
    if (artifact.isChildOfDeclaredDirectory()) {
      TreeArtifactValue tree = getTreeArtifactValue(artifact.getParent());
      result = tree == null ? null : tree.getChildValues().get(artifact);
    } else if (artifact instanceof ArchivedTreeArtifact) {
      TreeArtifactValue tree = getTreeArtifactValue(artifact.getParent());
      ArchivedRepresentation archivedRepresentation =
          tree.getArchivedRepresentation()
              .orElseThrow(
                  () -> new NoSuchElementException("Missing archived representation in: " + tree));
      checkArgument(
          archivedRepresentation.archivedTreeFileArtifact().equals(artifact),
          "Multiple archived tree artifacts for: %s",
          artifact.getParent());
      result = archivedRepresentation.archivedFileValue();
    } else {
      result = getAllFileValues().get(artifact);
    }

    return checkNotNull(
        result,
        "Missing artifact %s (generating action key %s) in %s",
        artifact,
        ((DerivedArtifact) artifact).getGeneratingActionKey(),
        this);
  }

  @Nullable
  TreeArtifactValue getTreeArtifactValue(Artifact artifact) {
    checkArgument(artifact.isTreeArtifact(), artifact);
    return null;
  }

  /**
   * Returns a map containing all artifacts output by the action, except for tree artifacts which
   * are accessible via {@link #getAllTreeArtifactValues}.
   */
  public abstract ImmutableMap<Artifact, FileArtifactValue> getAllFileValues();

  /** Returns a map containing all tree artifacts output by the action. */
  public ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactValues() {
    return ImmutableMap.of();
  }

  /**
   * Returns whether all artifacts output by the action are {@linkplain FileArtifactValue#isRemote
   * remote}.
   */
  public abstract boolean isEntirelyRemote();

  public ImmutableList<FilesetOutputSymlink> getOutputSymlinks() {
    return ImmutableList.of();
  }

  public NestedSet<Artifact> getDiscoveredModules() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public final String toString() {
    return MoreObjects.toStringHelper(ClassName.getSimpleNameWithOuter(getClass()))
        .add("files", getAllFileValues())
        .add("trees", getAllTreeArtifactValues())
        .toString();
  }

  @Override
  public final boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ActionExecutionValue)) {
      return false;
    }
    ActionExecutionValue o = (ActionExecutionValue) obj;
    return getAllFileValues().equals(o.getAllFileValues())
        && getAllTreeArtifactValues().equals(o.getAllTreeArtifactValues())
        && Objects.equals(getOutputSymlinks(), o.getOutputSymlinks())
        // We use shallowEquals to avoid materializing the nested sets just for change-pruning. This
        // makes change-pruning potentially less effective, but never incorrect.
        && getDiscoveredModules().shallowEquals(o.getDiscoveredModules());
  }

  @Override
  public final int hashCode() {
    return 31
            * HashCodes.hashObjects(
                getAllFileValues(), getAllTreeArtifactValues(), getOutputSymlinks())
        + getDiscoveredModules().shallowHashCode();
  }

  private static <V> ImmutableMap<Artifact, V> transformMap(
      ImmutableMap<Artifact, V> data,
      Map<OwnerlessArtifactWrapper, Artifact> newArtifactMap,
      Action action,
      BiFunction<Artifact, V, V> transform)
      throws ActionTransformException {
    if (data.isEmpty()) {
      return data;
    }

    ImmutableMap.Builder<Artifact, V> result = ImmutableMap.builderWithExpectedSize(data.size());
    for (Map.Entry<Artifact, V> entry : data.entrySet()) {
      Artifact artifact = entry.getKey();
      Artifact newArtifact = newArtifactMap.get(new OwnerlessArtifactWrapper(artifact));
      if (newArtifact == null) {
        throw new ActionTransformException(
            "No output matching %s, cannot share with %s", artifact, action);
      }
      result.put(newArtifact, transform.apply(newArtifact, entry.getValue()));
    }
    return result.buildOrThrow();
  }

  /** Transforms the children of a {@link TreeArtifactValue} so that owners are consistent. */
  private static TreeArtifactValue transformSharedTree(
      Artifact newArtifact, TreeArtifactValue tree) {
    checkState(newArtifact.isTreeArtifact(), "Expected tree artifact, got %s", newArtifact);

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
            archivedRepresentation ->
                newTree.setArchivedRepresentation(
                    ArchivedTreeArtifact.createForTree(newParent),
                    archivedRepresentation.archivedFileValue()));

    return newTree.build();
  }

  /**
   * Creates a new {@code ActionExecutionValue} by transforming this one's outputs so that artifact
   * owners match the given action's outputs.
   *
   * <p>The given action must be {@linkplain
   * com.google.devtools.build.lib.actions.Actions#canBeShared shareable} with the action that
   * originally produced this {@code ActionExecutionValue}.
   */
  public final ActionExecutionValue transformForSharedAction(Action action)
      throws ActionTransformException {
    ImmutableMap<Artifact, FileArtifactValue> artifactData = getAllFileValues();
    ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData = getAllTreeArtifactValues();
    Collection<Artifact> outputs = action.getOutputs();
    if (outputs.size() != artifactData.size() + treeArtifactData.size()) {
      throw new ActionTransformException("Cannot share %s with %s", this, action);
    }
    ImmutableMap<OwnerlessArtifactWrapper, Artifact> newArtifactMap =
        Maps.uniqueIndex(outputs, OwnerlessArtifactWrapper::new);
    return create(
        transformMap(artifactData, newArtifactMap, action, (newArtifact, value) -> value),
        transformMap(
            treeArtifactData, newArtifactMap, action, ActionExecutionValue::transformSharedTree),
        getOutputSymlinks(),
        // Discovered modules come from the action's inputs, and so don't need to be transformed.
        getDiscoveredModules());
  }

  /**
   * Exception thrown when {@link #transformForSharedAction} is called with an action that does not
   * have the same outputs.
   */
  public static final class ActionTransformException extends Exception {
    @FormatMethod
    private ActionTransformException(@FormatString String format, Object... args) {
      super(String.format(format, args));
    }
  }

  /**
   * The result of an action that outputs a single file (the common case). Optimizes for space by
   * storing the single artifact and value without the {@link ImmutableMap} wrapper.
   */
  private static class SingleOutputFile extends ActionExecutionValue {
    private final Artifact artifact;
    private final FileArtifactValue value;

    SingleOutputFile(ImmutableMap<Artifact, FileArtifactValue> artifactData) {
      this.artifact = Iterables.getOnlyElement(artifactData.keySet());
      this.value = artifactData.get(artifact);
    }

    // Override to avoid creating an ImmutableMap in the common case that the requested artifact is
    // correct. This bypasses the preconditions checks in super, but if the artifact is correct,
    // those would all pass anyway.
    @Override
    public final FileArtifactValue getExistingFileArtifactValue(Artifact artifact) {
      if (artifact.equals(this.artifact)) {
        return value;
      }
      // This will throw an exception. Call super to make failure modes consistent.
      return super.getExistingFileArtifactValue(artifact);
    }

    @Override
    public final ImmutableMap<Artifact, FileArtifactValue> getAllFileValues() {
      return ImmutableMap.of(artifact, value);
    }

    @Override
    public final boolean isEntirelyRemote() {
      return value.isRemote();
    }
  }

  /**
   * The result of a {@link
   * com.google.devtools.build.lib.view.fileset.SkyframeFilesetManifestAction}.
   */
  private static final class Fileset extends SingleOutputFile {
    private final ImmutableList<FilesetOutputSymlink> outputSymlinks;

    Fileset(
        ImmutableMap<Artifact, FileArtifactValue> artifactData,
        ImmutableList<FilesetOutputSymlink> outputSymlinks) {
      super(artifactData);
      this.outputSymlinks = outputSymlinks;
    }

    @Override
    public ImmutableList<FilesetOutputSymlink> getOutputSymlinks() {
      return outputSymlinks;
    }
  }

  /**
   * The result of a {@link com.google.devtools.build.lib.rules.cpp.CppCompileAction} that
   * {@linkplain IncludeScannable#getDiscoveredModules discovers modules}.
   */
  private static final class ModuleDiscovering extends SingleOutputFile {
    private final NestedSet<Artifact> discoveredModules;

    ModuleDiscovering(
        ImmutableMap<Artifact, FileArtifactValue> artifactData,
        NestedSet<Artifact> discoveredModules) {
      super(artifactData);
      this.discoveredModules = discoveredModules;
    }

    @Override
    public NestedSet<Artifact> getDiscoveredModules() {
      return discoveredModules;
    }
  }

  /** The result of an action that outputs an arbitrary number of files. */
  private static class MultiOutputFile extends ActionExecutionValue {
    private final ImmutableMap<Artifact, FileArtifactValue> artifactData;

    MultiOutputFile(ImmutableMap<Artifact, FileArtifactValue> artifactData) {
      this.artifactData = artifactData;
    }

    @Override
    public final ImmutableMap<Artifact, FileArtifactValue> getAllFileValues() {
      return artifactData;
    }

    @Override
    public boolean isEntirelyRemote() {
      for (FileArtifactValue fileArtifactValue : artifactData.values()) {
        if (!fileArtifactValue.isRemote()) {
          return false;
        }
      }
      return true;
    }
  }

  /** The result of an action that outputs a single tree artifact and no other files. */
  private static final class SingleTree extends ActionExecutionValue {
    private final Artifact treeArtifact;
    private final TreeArtifactValue treeValue;

    SingleTree(ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData) {
      this.treeArtifact = Iterables.getOnlyElement(treeArtifactData.keySet());
      this.treeValue = treeArtifactData.get(treeArtifact);
    }

    @Override
    @Nullable
    TreeArtifactValue getTreeArtifactValue(Artifact artifact) {
      checkArgument(artifact.isTreeArtifact(), artifact);
      return artifact.equals(treeArtifact) ? treeValue : null;
    }

    @Override
    public ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactValues() {
      return ImmutableMap.of(treeArtifact, treeValue);
    }

    @Override
    public ImmutableMap<Artifact, FileArtifactValue> getAllFileValues() {
      return ImmutableMap.of();
    }

    @Override
    public boolean isEntirelyRemote() {
      return treeValue.isEntirelyRemote();
    }
  }

  /**
   * The result of an action that outputs multiple tree artifacts or a combination of tree artifacts
   * and files.
   */
  private static final class MultiTree extends MultiOutputFile {
    private final ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData;

    MultiTree(
        ImmutableMap<Artifact, FileArtifactValue> artifactData,
        ImmutableMap<Artifact, TreeArtifactValue> treeArtifactData) {
      super(artifactData);
      this.treeArtifactData = treeArtifactData;
    }

    @Override
    @Nullable
    TreeArtifactValue getTreeArtifactValue(Artifact artifact) {
      checkArgument(artifact.isTreeArtifact(), artifact);
      return treeArtifactData.get(artifact);
    }

    @Override
    public ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactValues() {
      return treeArtifactData;
    }

    @Override
    public boolean isEntirelyRemote() {
      for (TreeArtifactValue treeArtifactValue : treeArtifactData.values()) {
        if (!treeArtifactValue.isEntirelyRemote()) {
          return false;
        }
      }
      return super.isEntirelyRemote();
    }
  }
}
