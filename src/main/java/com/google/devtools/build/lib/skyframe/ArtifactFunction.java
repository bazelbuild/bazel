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

import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactFileMetadata;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactSkyKey;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversalRoot;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalFunction.RecursiveFilesystemTraversalException;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.ResolvedFile;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.TraversalRequest;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** A builder of values for {@link ArtifactSkyKey} keys. */
class ArtifactFunction implements SkyFunction {

  private final Supplier<Boolean> mkdirForTreeArtifacts;

  public ArtifactFunction(Supplier<Boolean> mkdirForTreeArtifacts) {
    this.mkdirForTreeArtifacts = mkdirForTreeArtifacts;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ArtifactFunctionException, InterruptedException {
    Artifact artifact = ArtifactSkyKey.artifact(skyKey);
    if (artifact.isSourceArtifact()) {
      try {
        return createSourceValue(artifact, ArtifactSkyKey.isMandatory(skyKey), env);
      } catch (MissingInputFileException e) {
        // The error is not necessarily truly transient, but we mark it as such because we have
        // the above side effect of posting an event to the EventBus. Importantly, that event
        // is potentially used to report root causes.
        throw new ArtifactFunctionException(e, Transience.TRANSIENT);
      } catch (IOException e) {
        throw new ArtifactFunctionException(e, Transience.TRANSIENT);
      }
    }

    ArtifactDependencies artifactDependencies =
        ArtifactDependencies.discoverDependencies(artifact, env);
    if (artifactDependencies == null) {
      return null;
    }

    // If the action is an ActionTemplate, we need to expand the ActionTemplate into concrete
    // actions, execute those actions in parallel and then aggregate the action execution results.
    if (artifactDependencies.isTemplateActionForTreeArtifact()) {
      if (mkdirForTreeArtifacts.get()) {
        mkdirForTreeArtifact(artifact, env);
      }
      return createTreeArtifactValueFromActionKey(artifactDependencies, env);
    }

    ActionExecutionValue actionValue =
        (ActionExecutionValue)
            env.getValue(artifactDependencies.getNontemplateActionExecutionKey());
    if (actionValue == null) {
      return null;
    }

    if (artifact.isTreeArtifact()) {
      // We get a request for the whole tree artifact. We can just return the associated
      // TreeArtifactValue.
      return Preconditions.checkNotNull(actionValue.getTreeArtifactValue(artifact), artifact);
    }

    if (artifact.isMiddlemanArtifact()) {
      Action action =
          Preconditions.checkNotNull(
              artifactDependencies.getAction(), "Null middleman action? %s", artifactDependencies);
      if (isAggregatingValue(action)) {
        return createAggregatingValue(
            artifact, action, actionValue.getArtifactValue(artifact), env);
      }
    }

    return createSimpleFileArtifactValue(artifact, actionValue);
  }

  private static void mkdirForTreeArtifact(Artifact artifact, Environment env)
      throws ArtifactFunctionException {
    try {
      artifact.getPath().createDirectoryAndParents();
    } catch (IOException e) {
      env.getListener()
          .handle(
              Event.error(
                  String.format(
                      "Failed to create output directory for TreeArtifact %s: %s",
                      artifact, e.getMessage())));
      throw new ArtifactFunctionException(e, Transience.TRANSIENT);
    }
  }

  private static TreeArtifactValue createTreeArtifactValueFromActionKey(
      ArtifactDependencies artifactDependencies, Environment env) throws InterruptedException {
    ActionLookupKey actionLookupKey = artifactDependencies.getActionLookupKey();
    int actionIndex = artifactDependencies.getActionIndex();
    Artifact treeArtifact = artifactDependencies.getArtifact();

    // Request the list of expanded actions from the ActionTemplate.
    ActionTemplateExpansionValue.ActionTemplateExpansionKey templateKey =
        ActionTemplateExpansionValue.key(actionLookupKey, actionIndex);
    ActionTemplateExpansionValue expansionValue =
        (ActionTemplateExpansionValue) env.getValue(templateKey);

    // The expanded actions are not yet available.
    if (env.valuesMissing()) {
      return null;
    }

    List<SkyKey> expandedActionExecutionKeys = new ArrayList<>(expansionValue.getNumActions());
    for (int i = 0; i < expansionValue.getNumActions(); i++) {
      expandedActionExecutionKeys.add(ActionExecutionValue.key(templateKey, i));
    }
    Map<SkyKey, SkyValue> expandedActionValueMap = env.getValues(expandedActionExecutionKeys);

    // The execution values of the expanded actions are not yet all available.
    if (env.valuesMissing()) {
      return null;
    }

    // Aggregate the ArtifactValues for individual TreeFileArtifacts into a TreeArtifactValue for
    // the parent TreeArtifact.
    ImmutableMap.Builder<TreeFileArtifact, FileArtifactValue> map = ImmutableMap.builder();
    for (int i = 0; i < expansionValue.getNumActions(); i++) {
      final ActionExecutionValue actionExecutionValue =
          (ActionExecutionValue)
              Preconditions.checkNotNull(
                  expandedActionValueMap.get(expandedActionExecutionKeys.get(i)),
                  "Missing tree value: %s %s %s %s %s",
                  treeArtifact,
                  actionLookupKey,
                  actionIndex,
                  expansionValue,
                  expandedActionValueMap);
      Iterable<TreeFileArtifact> treeFileArtifacts =
          Iterables.transform(
              Iterables.filter(
                  actionExecutionValue.getAllFileValues().keySet(),
                  new Predicate<Artifact>() {
                    @Override
                    public boolean apply(Artifact artifact) {
                      Preconditions.checkState(
                          artifact.hasParent(),
                          "No parent: %s %s %s %s %s",
                          artifact,
                          treeArtifact,
                          actionExecutionValue,
                          actionLookupKey,
                          actionIndex);
                      return artifact.getParent().equals(treeArtifact);
                    }
                  }),
              new Function<Artifact, TreeFileArtifact>() {
                @Override
                public TreeFileArtifact apply(Artifact artifact) {
                  return (TreeFileArtifact) artifact;
                }
              });

      Preconditions.checkState(
          !Iterables.isEmpty(treeFileArtifacts),
          "Action denoted by %s does not output TreeFileArtifact under %s",
          expandedActionExecutionKeys.get(i),
          treeArtifact);

      for (TreeFileArtifact treeFileArtifact : treeFileArtifacts) {
        FileArtifactValue value =
            createSimpleFileArtifactValue(treeFileArtifact, actionExecutionValue);
        map.put(treeFileArtifact, value);
      }
    }

    // Return the aggregated TreeArtifactValue.
    return TreeArtifactValue.create(map.build());
  }

  private FileArtifactValue createSourceValue(Artifact artifact, boolean mandatory, Environment env)
      throws MissingInputFileException, IOException, InterruptedException {
    RootedPath path = RootedPath.toRootedPath(artifact.getRoot().getRoot(), artifact.getPath());
    SkyKey fileSkyKey = FileValue.key(path);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
    } catch (IOException e) {
      throw makeMissingInputFileException(artifact, mandatory, e, env.getListener());
    }
    if (fileValue == null) {
      return null;
    }
    if (!fileValue.exists()) {
      if (!mandatory) {
        return FileArtifactValue.MISSING_FILE_MARKER;
      } else {
        throw makeMissingInputFileException(artifact, mandatory, null, env.getListener());
      }
    }
    // For directory artifacts that are not Filesets, we initiate a directory traversal here, and
    // compute a hash from the directory structure.
    if (fileValue.isDirectory() && TrackSourceDirectoriesFlag.trackSourceDirectories()) {
      // We rely on the guarantees of RecursiveFilesystemTraversalFunction for correctness.
      //
      // This approach may have unexpected interactions with --package_path. In particular, the exec
      // root is setup from the loading / analysis phase, and it is now too late to change it;
      // therefore, this may traverse a different set of files depending on which targets are built
      // at the same time and what the package-path layout is (this may be moot if there is only one
      // entry). Or this may return a set of files that's inconsistent with those actually available
      // to the action (for local execution).
      //
      // In the future, we need to make this result the source of truth for the files available to
      // the action so that we at least have consistency.
      TraversalRequest request = TraversalRequest.create(
          DirectTraversalRoot.forRootedPath(path),
          /*isRootGenerated=*/ false,
          PackageBoundaryMode.CROSS,
          /*strictOutputFiles=*/ true,
          /*skipTestingForSubpackage=*/ true,
          /*errorInfo=*/ null);
      RecursiveFilesystemTraversalValue value;
      try {
        value =
            (RecursiveFilesystemTraversalValue) env.getValueOrThrow(
                request, RecursiveFilesystemTraversalException.class);
      } catch (RecursiveFilesystemTraversalException e) {
        throw new IOException(e);
      }
      if (value == null) {
        return null;
      }
      Fingerprint fp = new Fingerprint();
      for (ResolvedFile file : value.getTransitiveFiles()) {
        fp.addString(file.getNameInSymlinkTree().getPathString());
        fp.addInt(file.getMetadata().hashCode());
      }
      return FileArtifactValue.createDirectoryWithHash(fp.digestAndReset());
    }
    try {
      return FileArtifactValue.create(artifact, fileValue);
    } catch (IOException e) {
      throw makeMissingInputFileException(artifact, mandatory, e, env.getListener());
    }
  }

  private static MissingInputFileException makeMissingInputFileException(
      Artifact artifact, boolean mandatory, Exception failure, EventHandler reporter) {
    String extraMsg = (failure == null) ? "" : (":" + failure.getMessage());
    MissingInputFileException ex =
        new MissingInputFileException(constructErrorMessage(artifact) + extraMsg, null);
    if (mandatory) {
      reporter.handle(Event.error(ex.getLocation(), ex.getMessage()));
    }
    return ex;
  }

  // Non-aggregating artifact -- should contain at most one piece of artifact data.
  // data may be null if and only if artifact is a middleman artifact.
  private static FileArtifactValue createSimpleFileArtifactValue(
      Artifact artifact, ActionExecutionValue actionValue) {
    FileArtifactValue value = actionValue.getArtifactValue(artifact);
    if (value != null) {
      return value;
    }
    // Middleman artifacts have no corresponding files, so their ArtifactValues should have already
    // been constructed during execution of the action.
    Preconditions.checkState(!artifact.isMiddlemanArtifact(), artifact);
    ArtifactFileMetadata data =
        Preconditions.checkNotNull(actionValue.getData(artifact), "%s %s", artifact, actionValue);
    Preconditions.checkNotNull(
        data.getDigest(), "Digest should already have been calculated for %s (%s)", artifact, data);
    // Directories are special-cased because their mtimes are used, so should have been constructed
    // during execution of the action (in ActionMetadataHandler#maybeStoreAdditionalData).
    Preconditions.checkState(data.isFile(), "Unexpected not file %s (%s)", artifact, data);
    return FileArtifactValue.createNormalFile(data);
  }

  @Nullable
  private static AggregatingArtifactValue createAggregatingValue(
      Artifact artifact,
      ActionAnalysisMetadata action,
      FileArtifactValue value,
      SkyFunction.Environment env)
      throws InterruptedException {
    ImmutableList.Builder<Pair<Artifact, FileArtifactValue>> fileInputsBuilder =
        ImmutableList.builder();
    ImmutableList.Builder<Pair<Artifact, TreeArtifactValue>> directoryInputsBuilder =
        ImmutableList.builder();
    for (Map.Entry<SkyKey, SkyValue> entry : env.getValues(action.getInputs()).entrySet()) {
      Artifact input = ArtifactSkyKey.artifact(entry.getKey());
      SkyValue inputValue = entry.getValue();
      if (inputValue == null) {
        return null;
      }
      if (inputValue instanceof FileArtifactValue) {
        fileInputsBuilder.add(Pair.of(input, (FileArtifactValue) inputValue));
      } else if (inputValue instanceof TreeArtifactValue) {
        directoryInputsBuilder.add(Pair.of(input, (TreeArtifactValue) inputValue));
      } else {
        // We do not recurse in aggregating middleman artifacts.
        Preconditions.checkState(
            !(inputValue instanceof AggregatingArtifactValue),
            "%s %s %s",
            artifact,
            action,
            inputValue);
      }
    }

    ImmutableList<Pair<Artifact, FileArtifactValue>> fileInputs =
        ImmutableList.sortedCopyOf(
            Comparator.comparing(pair -> pair.getFirst().getExecPathString()),
            fileInputsBuilder.build());
    ImmutableList<Pair<Artifact, TreeArtifactValue>> directoryInputs =
        ImmutableList.sortedCopyOf(
            Comparator.comparing(pair -> pair.getFirst().getExecPathString()),
            directoryInputsBuilder.build());

    return (action.getActionType() == MiddlemanType.AGGREGATING_MIDDLEMAN)
        ? new AggregatingArtifactValue(fileInputs, directoryInputs, value)
        : new RunfilesArtifactValue(fileInputs, directoryInputs, value);
  }

  /**
   * Returns whether this value needs to contain the data of all its inputs. Currently only tests to
   * see if the action is an aggregating or runfiles middleman action. However, may include Fileset
   * artifacts in the future.
   */
  private static boolean isAggregatingValue(ActionAnalysisMetadata action) {
    switch (action.getActionType()) {
      case AGGREGATING_MIDDLEMAN:
      case RUNFILES_MIDDLEMAN:
        return true;
      default:
        return false;
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(ArtifactSkyKey.artifact(skyKey).getOwner());
  }

  static ActionLookupKey getActionLookupKey(Artifact artifact) {
    ArtifactOwner artifactOwner = artifact.getArtifactOwner();
    Preconditions.checkState(
        artifactOwner instanceof ActionLookupKey, "%s %s", artifact, artifactOwner);
    return (ActionLookupKey) artifactOwner;
  }

  @Nullable
  static ActionLookupValue getActionLookupValue(
      SkyKey actionLookupKey, SkyFunction.Environment env, Artifact artifact)
      throws InterruptedException {
    ActionLookupValue value = (ActionLookupValue) env.getValue(actionLookupKey);
    if (value == null) {
      ArtifactOwner artifactOwner = artifact.getArtifactOwner();
      Preconditions.checkState(
          artifactOwner == CoverageReportValue.COVERAGE_REPORT_KEY,
          "Not-yet-present artifact owner: %s (%s %s)",
          artifactOwner,
          artifact,
          actionLookupKey);
      return null;
    }
    return value;
  }

  static final class ArtifactFunctionException extends SkyFunctionException {
    ArtifactFunctionException(MissingInputFileException e, Transience transience) {
      super(e, transience);
    }

    ArtifactFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }
  }

  private static String constructErrorMessage(Artifact artifact) {
    if (artifact.getOwner() == null) {
      return String.format("missing input file '%s'", artifact.getPath().getPathString());
    } else {
      return String.format("missing input file '%s'", artifact.getOwner());
    }
  }

  /** Describes dependencies of derived artifacts. */
  // TODO(mschaller): extend this to comprehensively support template actions and middleman
  // artifacts.
  static class ArtifactDependencies {

    private final Artifact artifact;
    private final ActionLookupKey actionLookupKey;
    private final ActionLookupValue actionLookupValue;
    private final int actionIndex;

    private ArtifactDependencies(
        Artifact artifact,
        ActionLookupKey actionLookupKey,
        ActionLookupValue actionLookupValue,
        int actionIndex) {
      this.artifact = artifact;
      this.actionLookupKey = actionLookupKey;
      this.actionLookupValue = actionLookupValue;
      this.actionIndex = actionIndex;
    }

    /**
     * Constructs an {@link ArtifactDependencies} for the provided {@code derivedArtifact}, which
     * must not be a source artifact. Returns {@code null} if any dependencies are not yet ready.
     */
    @Nullable
    static ArtifactDependencies discoverDependencies(
        Artifact derivedArtifact, SkyFunction.Environment env) throws InterruptedException {
      Preconditions.checkArgument(
          !derivedArtifact.isSourceArtifact(),
          "derivedArtifact is not derived: %s",
          derivedArtifact);

      ActionLookupKey actionLookupKey = ArtifactFunction.getActionLookupKey(derivedArtifact);
      ActionLookupValue actionLookupValue =
          ArtifactFunction.getActionLookupValue(actionLookupKey, env, derivedArtifact);
      if (actionLookupValue == null) {
        return null;
      }
      Integer actionIndex = actionLookupValue.getGeneratingActionIndex(derivedArtifact);
      if (derivedArtifact.hasParent() && actionIndex == null) {
        // If a TreeFileArtifact is created by a templated action, then it should have the proper
        // reference to its owner. However, if it was created as part of a directory, by the first
        // TreeArtifact-generating action in a chain, then its parent's generating action also
        // generated it. This catches that case.
        actionIndex = actionLookupValue.getGeneratingActionIndex(derivedArtifact.getParent());
      }
      Preconditions.checkNotNull(
          actionIndex, "%s %s %s", derivedArtifact, actionLookupKey, actionLookupValue);

      return new ArtifactDependencies(
          derivedArtifact, actionLookupKey, actionLookupValue, actionIndex);
    }

    Artifact getArtifact() {
      return artifact;
    }

    ActionLookupKey getActionLookupKey() {
      return actionLookupKey;
    }

    int getActionIndex() {
      return actionIndex;
    }

    boolean isTemplateActionForTreeArtifact() {
      return artifact.isTreeArtifact() && actionLookupValue.isActionTemplate(actionIndex);
    }

    ActionLookupData getNontemplateActionExecutionKey() {
      Preconditions.checkState(
          !actionLookupValue.isActionTemplate(actionIndex),
          "Action is unexpectedly template: %s",
          this);
      return ActionExecutionValue.key(actionLookupKey, actionIndex);
    }

    Action getAction() {
      return actionLookupValue.getAction(actionIndex);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("artifact", artifact)
          .add("actionLookupKey", actionLookupKey)
          .add("actionLookupValue", actionLookupValue)
          .add("actionIndex", actionIndex)
          .toString();
    }
  }
}
