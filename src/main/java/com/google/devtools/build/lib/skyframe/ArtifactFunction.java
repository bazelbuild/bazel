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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversalRoot;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalFunction.RecursiveFilesystemTraversalException;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.ResolvedFile;
import com.google.devtools.build.lib.skyframe.RecursiveFilesystemTraversalValue.TraversalRequest;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Comparator;
import java.util.Map;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A builder of values for {@link Artifact} keys when the key is not a simple generated artifact. To
 * save memory, ordinary generated artifacts (non-middleman, non-tree) have their metadata accessed
 * directly from the corresponding {@link ActionExecutionValue}. This SkyFunction is therefore only
 * usable for source, middleman, and tree artifacts.
 */
class ArtifactFunction implements SkyFunction {
  private final Supplier<Boolean> mkdirForTreeArtifacts;

  public static final class MissingFileArtifactValue implements SkyValue {
    private final MissingInputFileException exception;

    private MissingFileArtifactValue(MissingInputFileException e) {
      this.exception = e;
    }

    public MissingInputFileException getException() {
      return exception;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("exception", exception).toString();
    }
  }

  public ArtifactFunction(Supplier<Boolean> mkdirForTreeArtifacts) {
    this.mkdirForTreeArtifacts = mkdirForTreeArtifacts;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ArtifactFunctionException, InterruptedException {
    Artifact artifact = (Artifact) skyKey;
    if (!artifact.hasKnownGeneratingAction()) {
      // If the artifact has no known generating action, it is either a source artifact, or a
      // NinjaMysteryArtifact, which undergoes the same handling here.
      try {
        return createSourceValue(artifact, env);
      } catch (IOException e) {
        throw new ArtifactFunctionException(e);
      }
    }
    Artifact.DerivedArtifact derivedArtifact = (DerivedArtifact) artifact;

    ArtifactDependencies artifactDependencies =
        ArtifactDependencies.discoverDependencies(derivedArtifact, env);
    if (artifactDependencies == null) {
      return null;
    }

    // If the action is an ActionTemplate, we need to expand the ActionTemplate into concrete
    // actions, execute those actions in parallel and then aggregate the action execution results.
    ActionTemplate<?> actionTemplate = artifactDependencies.maybeGetTemplateActionForTreeArtifact();
    if (actionTemplate != null) {
      if (mkdirForTreeArtifacts.get()) {
        mkdirForTreeArtifact(artifact, env, actionTemplate);
      }
      return createTreeArtifactValueFromActionKey(artifactDependencies, env);
    }

    ActionLookupData generatingActionKey = derivedArtifact.getGeneratingActionKey();
    ActionExecutionValue actionValue = (ActionExecutionValue) env.getValue(generatingActionKey);
    if (actionValue == null) {
      return null;
    }

    if (artifact.isTreeArtifact()) {
      // We got a request for the whole tree artifact. We can just return the associated
      // TreeArtifactValue.
      return Preconditions.checkNotNull(actionValue.getTreeArtifactValue(artifact), artifact);
    }

    Preconditions.checkState(artifact.isMiddlemanArtifact(), artifact);
    Action action =
        Preconditions.checkNotNull(
            artifactDependencies.actionLookupValue.getAction(generatingActionKey.getActionIndex()),
            "Null middleman action? %s",
            artifactDependencies);
    FileArtifactValue individualMetadata =
        Preconditions.checkNotNull(
            actionValue.getArtifactValue(artifact), "%s %s", artifact, actionValue);
    if (isAggregatingValue(action)) {
      return createAggregatingValue(artifact, action, individualMetadata, env);
    }
    return individualMetadata;
  }

  private static void mkdirForTreeArtifact(
      Artifact artifact, Environment env, ActionTemplate<?> actionForFailure)
      throws ArtifactFunctionException {
    try {
      artifact.getPath().createDirectoryAndParents();
    } catch (IOException e) {
      String errorMessage =
          String.format(
              "Failed to create output directory for TreeArtifact %s: %s",
              artifact.getExecPath(), e.getMessage());
      env.getListener()
          .handle(Event.error(actionForFailure.getOwner().getLocation(), errorMessage));
      // We could throw this as an IOException and expect our callers to catch and reprocess it,
      // but we know the action at fault, so we should be in charge.
      throw new ArtifactFunctionException(
          new ActionExecutionException(errorMessage, e, actionForFailure, false));
    }
  }

  private static SkyValue createTreeArtifactValueFromActionKey(
      ArtifactDependencies artifactDependencies, Environment env) throws InterruptedException {
    // Request the list of expanded actions from the ActionTemplate.
    ActionTemplateExpansion actionTemplateExpansion =
        artifactDependencies.getActionTemplateExpansion(env);
    if (actionTemplateExpansion == null) {
      // The expanded actions are not yet available.
      return null;
    }
    ActionTemplateExpansionValue expansionValue = actionTemplateExpansion.getValue();
    ImmutableList<ActionLookupData> expandedActionExecutionKeys =
        actionTemplateExpansion.getExpandedActionExecutionKeys();

    Map<SkyKey, SkyValue> expandedActionValueMap = env.getValues(expandedActionExecutionKeys);
    if (env.valuesMissing()) {
      // The execution values of the expanded actions are not yet all available.
      return null;
    }

    // Aggregate the ArtifactValues for individual TreeFileArtifacts into a TreeArtifactValue for
    // the parent TreeArtifact.
    ImmutableMap.Builder<TreeFileArtifact, FileArtifactValue> map = ImmutableMap.builder();
    boolean omitted = false;
    for (ActionLookupData actionKey : expandedActionExecutionKeys) {
      ActionExecutionValue actionExecutionValue =
          (ActionExecutionValue)
              Preconditions.checkNotNull(
                  expandedActionValueMap.get(actionKey),
                  "Missing tree value: %s %s %s",
                  artifactDependencies,
                  expansionValue,
                  expandedActionValueMap);
      Iterable<TreeFileArtifact> treeFileArtifacts =
          Iterables.transform(
              Iterables.filter(
                  actionExecutionValue.getAllFileValues().keySet(),
                  artifact -> {
                    Preconditions.checkState(
                        artifact.hasParent(),
                        "No parent: %s %s %s",
                        artifact,
                        actionExecutionValue,
                        artifactDependencies);
                    return artifact.getParent().equals(artifactDependencies.artifact);
                  }),
              artifact -> (TreeFileArtifact) artifact);

      Preconditions.checkState(
          !Iterables.isEmpty(treeFileArtifacts),
          "Action denoted by %s does not output TreeFileArtifact from %s",
          actionKey,
          artifactDependencies);

      for (TreeFileArtifact treeFileArtifact : treeFileArtifacts) {
        FileArtifactValue value =
            ActionExecutionValue.createSimpleFileArtifactValue(
                treeFileArtifact, actionExecutionValue);
        if (FileArtifactValue.OMITTED_FILE_MARKER.equals(value)) {
          omitted = true;
        } else {
          map.put(treeFileArtifact, value);
        }
      }
    }

    ImmutableMap<TreeFileArtifact, FileArtifactValue> children = map.build();

    if (omitted) {
      Preconditions.checkState(
          children.isEmpty(),
          "Action template expansion has some but not all outputs omitted, present outputs: %s",
          children);
      return FileArtifactValue.OMITTED_FILE_MARKER;
    }
    return TreeArtifactValue.create(children);
  }

  private static SkyValue createSourceValue(Artifact artifact, Environment env)
      throws IOException, InterruptedException {
    RootedPath path = RootedPath.toRootedPath(artifact.getRoot().getRoot(), artifact.getPath());
    SkyKey fileSkyKey = FileValue.key(path);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
    } catch (IOException e) {
      return makeMissingInputFileValue(artifact, e);
    }
    if (fileValue == null) {
      return null;
    }
    if (!fileValue.exists()) {
      return makeMissingInputFileValue(artifact, null);
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
      for (ResolvedFile file : value.getTransitiveFiles().toList()) {
        fp.addString(file.getNameInSymlinkTree().getPathString());
        fp.addBytes(file.getMetadata().getDigest());
      }
      return FileArtifactValue.createForDirectoryWithHash(fp.digestAndReset());
    }
    try {
      return FileArtifactValue.createForSourceArtifact(artifact, fileValue);
    } catch (IOException e) {
      return makeMissingInputFileValue(artifact, e);
    }
  }

  static MissingFileArtifactValue makeMissingInputFileValue(Artifact artifact, Exception failure) {
    String extraMsg = (failure == null) ? "" : (": " + failure.getMessage());
    MissingInputFileException ex =
        new MissingInputFileException(constructErrorMessage(artifact) + extraMsg, null);
    return new MissingFileArtifactValue(ex);
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
    // Avoid iterating over nested set twice.
    Iterable<Artifact> inputs = action.getInputs().toList();
    Map<SkyKey, SkyValue> values = env.getValues(Artifact.keys(inputs));
    if (env.valuesMissing()) {
      return null;
    }
    for (Artifact input : inputs) {
      SkyValue inputValue = Preconditions.checkNotNull(values.get(Artifact.key(input)), input);
      if (inputValue instanceof FileArtifactValue) {
        fileInputsBuilder.add(Pair.of(input, (FileArtifactValue) inputValue));
      } else if (inputValue instanceof ActionExecutionValue) {
        fileInputsBuilder.add(
            Pair.of(
                input,
                ActionExecutionValue.createSimpleFileArtifactValue(
                    (DerivedArtifact) input, (ActionExecutionValue) inputValue)));
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
    return Label.print(((Artifact) skyKey).getOwner());
  }

  static ActionLookupKey getActionLookupKey(Artifact artifact) {
    ArtifactOwner artifactOwner = artifact.getArtifactOwner();
    Preconditions.checkState(
        artifactOwner instanceof ActionLookupKey, "%s %s", artifact, artifactOwner);
    return (ActionLookupKey) artifactOwner;
  }

  @Nullable
  static ActionLookupValue getActionLookupValue(
      ActionLookupKey actionLookupKey, SkyFunction.Environment env) throws InterruptedException {
    ActionLookupValue value = (ActionLookupValue) env.getValue(actionLookupKey);
    if (value == null) {
      Preconditions.checkState(
          actionLookupKey == CoverageReportValue.COVERAGE_REPORT_KEY,
          "Not-yet-present artifact owner: %s",
          actionLookupKey);
      return null;
    }
    return value;
  }

  static final class ArtifactFunctionException extends SkyFunctionException {
    ArtifactFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }

    ArtifactFunctionException(ActionExecutionException e) {
      super(e, Transience.TRANSIENT);
    }
  }

  private static String constructErrorMessage(Artifact artifact) {
    Label ownerLabel = artifact.getOwner();
    if (ownerLabel == null) {
      // Discovered inputs may not have an owner.
      return String.format("missing input file '%s'", artifact.getExecPathString());
    } else if (ownerLabel.toPathFragment().equals(artifact.getExecPath())) {
      // No additional useful information from path.
      return String.format("missing input file '%s'", ownerLabel);
    } else {
      // TODO(janakr): when is this hit?
      BugReport.sendBugReport(
          new IllegalStateException("Unexpected special owner? " + artifact + ", " + ownerLabel));
      return String.format(
          "missing input file '%s', owner: '%s'", artifact.getExecPathString(), ownerLabel);
    }
  }

  /** Describes dependencies of derived artifacts. */
  // TODO(b/19539699): extend this to comprehensively support all special artifact types (e.g.
  // middleman, etc).
  static class ArtifactDependencies {
    private final DerivedArtifact artifact;
    private final ActionLookupValue actionLookupValue;

    private ArtifactDependencies(DerivedArtifact artifact, ActionLookupValue actionLookupValue) {
      this.artifact = artifact;
      this.actionLookupValue = actionLookupValue;
    }

    /**
     * Constructs an {@link ArtifactDependencies} for the provided {@code derivedArtifact}. Returns
     * {@code null} if any dependencies are not yet ready.
     */
    @Nullable
    static ArtifactDependencies discoverDependencies(
        Artifact.DerivedArtifact derivedArtifact, SkyFunction.Environment env)
        throws InterruptedException {

      ActionLookupData generatingActionKey = derivedArtifact.getGeneratingActionKey();
      ActionLookupValue actionLookupValue =
          ArtifactFunction.getActionLookupValue(generatingActionKey.getActionLookupKey(), env);
      if (actionLookupValue == null) {
        return null;
      }

      return new ArtifactDependencies(derivedArtifact, actionLookupValue);
    }

    boolean isTemplateActionForTreeArtifact() {
      return maybeGetTemplateActionForTreeArtifact() != null;
    }

    ActionTemplate<?> maybeGetTemplateActionForTreeArtifact() {
      if (!artifact.isTreeArtifact()) {
        return null;
      }
      ActionAnalysisMetadata result =
          actionLookupValue.getActions().get(artifact.getGeneratingActionKey().getActionIndex());
      return result instanceof ActionTemplate ? (ActionTemplate<?>) result : null;
    }

    /**
     * Returns action template expansion information or {@code null} if that information is
     * unavailable.
     *
     * <p>Must not be called if {@code !isTemplateActionForTreeArtifact()}.
     */
    @Nullable
    ActionTemplateExpansion getActionTemplateExpansion(SkyFunction.Environment env)
        throws InterruptedException {
      Preconditions.checkState(
          isTemplateActionForTreeArtifact(), "Action is unexpectedly non-template: %s", this);
      ActionTemplateExpansionValue.ActionTemplateExpansionKey key =
          ActionTemplateExpansionValue.key(
              artifact.getArtifactOwner(), artifact.getGeneratingActionKey().getActionIndex());
      ActionTemplateExpansionValue value = (ActionTemplateExpansionValue) env.getValue(key);
      if (value == null) {
        return null;
      }
      return new ActionTemplateExpansion(key, value);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("artifact", artifact)
          .add("generatingActionKey", artifact.getGeneratingActionKey())
          .add("actionLookupValue", actionLookupValue)
          .toString();
    }
  }

  static class ActionTemplateExpansion {
    private final ActionTemplateExpansionValue.ActionTemplateExpansionKey key;
    private final ActionTemplateExpansionValue value;

    private ActionTemplateExpansion(
        ActionTemplateExpansionValue.ActionTemplateExpansionKey key,
        ActionTemplateExpansionValue value) {
      this.key = key;
      this.value = value;
    }

    ActionTemplateExpansionValue.ActionTemplateExpansionKey getKey() {
      return key;
    }

    ActionTemplateExpansionValue getValue() {
      return value;
    }

    ImmutableList<ActionLookupData> getExpandedActionExecutionKeys() {
      int numActions = value.getNumActions();
      ImmutableList.Builder<ActionLookupData> expandedActionExecutionKeys =
          ImmutableList.builderWithExpectedSize(numActions);
      for (ActionAnalysisMetadata action : value.getActions()) {
        expandedActionExecutionKeys.add(
            ((DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey());
      }
      return expandedActionExecutionKeys.build();
    }
  }
}
