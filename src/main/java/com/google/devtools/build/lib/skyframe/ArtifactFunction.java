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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.actions.ActionTemplate;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.skyframe.ArtifactSkyKey.OwnedArtifact;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;

/** A builder of values for {@link ArtifactSkyKey} keys. */
class ArtifactFunction implements SkyFunction {

  private final Predicate<PathFragment> allowedMissingInputs;

  ArtifactFunction(Predicate<PathFragment> allowedMissingInputs) {
    this.allowedMissingInputs = allowedMissingInputs;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ArtifactFunctionException, InterruptedException {
    OwnedArtifact ownedArtifact = (OwnedArtifact) skyKey.argument();
    Artifact artifact = ownedArtifact.getArtifact();
    if (artifact.isSourceArtifact()) {
      try {
        return createSourceValue(artifact, ownedArtifact.isMandatory(), env);
      } catch (MissingInputFileException e) {
        // The error is not necessarily truly transient, but we mark it as such because we have
        // the above side effect of posting an event to the EventBus. Importantly, that event
        // is potentially used to report root causes.
        throw new ArtifactFunctionException(e, Transience.TRANSIENT);
      }
    }

    ActionAnalysisMetadata actionMetadata = extractActionFromArtifact(artifact, env);
    if (actionMetadata == null) {
      return null;
    }

    // If the action is an ActionTemplate, we need to expand the ActionTemplate into concrete
    // actions, execute those actions in parallel and then aggregate the action execution results.
    if (artifact.isTreeArtifact() && actionMetadata instanceof ActionTemplate) {
      // Create the directory structures for the output TreeArtifact first.
      try {
        FileSystemUtils.createDirectoryAndParents(artifact.getPath());
      } catch (IOException e) {
        env.getListener().handle(
            Event.error(
                String.format(
                    "Failed to create output directory for TreeArtifact %s: %s",
                    artifact,
                    e.getMessage())));
        throw new ArtifactFunctionException(e, Transience.TRANSIENT);
      }

      return createTreeArtifactValueFromActionTemplate(
          (ActionTemplate) actionMetadata, artifact, env);
    } else {
      Preconditions.checkState(
          actionMetadata instanceof Action,
          "%s is not a proper Action object and therefore cannot be executed",
          actionMetadata);
      Action action = (Action) actionMetadata;
      ActionExecutionValue actionValue =
        (ActionExecutionValue) env.getValue(ActionExecutionValue.key(action));
      if (actionValue == null) {
        return null;
      }

      if (artifact.isTreeArtifact()) {
        // We get a request for the whole tree artifact. We can just return the associated
        // TreeArtifactValue.
        return Preconditions.checkNotNull(actionValue.getTreeArtifactValue(artifact), artifact);
      } else if (isAggregatingValue(action)) {
        return createAggregatingValue(artifact, action,
            actionValue.getArtifactValue(artifact), env);
      } else {
        return createSimpleFileArtifactValue(artifact, action, actionValue, env);
      }
    }
  }

  private static TreeArtifactValue createTreeArtifactValueFromActionTemplate(
      ActionTemplate actionTemplate, Artifact treeArtifact, Environment env)
      throws ArtifactFunctionException, InterruptedException {
    // Request the list of expanded actions from the ActionTemplate.
    ActionTemplateExpansionValue expansionValue = (ActionTemplateExpansionValue) env.getValue(
        ActionTemplateExpansionValue.key(actionTemplate));

    // The expanded actions are not yet available.
    if (env.valuesMissing()) {
      return null;
    }

    // Execute the expanded actions in parallel.
    Iterable<SkyKey> expandedActionExecutionKeys = ActionExecutionValue.keys(
        expansionValue.getExpandedActions());
    Map<SkyKey, SkyValue> expandedActionValueMap = env.getValues(expandedActionExecutionKeys);

    // The execution values of the expanded actions are not yet all available.
    if (env.valuesMissing()) {
      return null;
    }

    // Aggregate the ArtifactValues for individual TreeFileArtifacts into a TreeArtifactValue for
    // the parent TreeArtifact.
    ImmutableMap.Builder<TreeFileArtifact, FileArtifactValue> map = ImmutableMap.builder();
    for (Map.Entry<SkyKey, SkyValue> entry : expandedActionValueMap.entrySet()) {
      SkyKey expandedActionExecutionKey = entry.getKey();
      ActionExecutionValue actionExecutionValue = (ActionExecutionValue) entry.getValue();
      Action expandedAction = (Action) expandedActionExecutionKey.argument();

      Iterable<TreeFileArtifact> treeFileArtifacts = findActionOutputsWithMatchingParent(
          expandedAction, treeArtifact);

      Preconditions.checkState(
          !Iterables.isEmpty(treeFileArtifacts),
          "Action %s does not output TreeFileArtifact under %s",
          expandedAction,
          treeArtifact);

      for (TreeFileArtifact treeFileArtifact : treeFileArtifacts) {
        FileArtifactValue value  = createSimpleFileArtifactValue(
            treeFileArtifact, expandedAction, actionExecutionValue, env);
        map.put(treeFileArtifact, value);
      }
    }

    // Return the aggregated TreeArtifactValue.
    return TreeArtifactValue.create(map.build());
  }

  private FileArtifactValue createSourceValue(Artifact artifact, boolean mandatory, Environment env)
      throws MissingInputFileException, InterruptedException {
    SkyKey fileSkyKey = FileValue.key(RootedPath.toRootedPath(artifact.getRoot().getPath(),
        artifact.getPath()));
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
          InconsistentFilesystemException.class, FileSymlinkException.class);
    } catch (IOException | InconsistentFilesystemException | FileSymlinkException e) {
      throw makeMissingInputFileExn(artifact, mandatory, e, env.getListener());
    }
    if (fileValue == null) {
      return null;
    }
    if (!fileValue.exists()) {
      if (isAllowedMissingInput(fileSkyKey)) {
        return FileArtifactValue.MISSING_FILE_MARKER;
      } else {
        return missingInputFile(artifact, mandatory, null, env.getListener());
      }
    }
    try {
      return FileArtifactValue.create(artifact, fileValue);
    } catch (IOException e) {
      if (isAllowedMissingInput(fileSkyKey)) {
        return FileArtifactValue.MISSING_FILE_MARKER;
      }
      throw makeMissingInputFileExn(artifact, mandatory, e, env.getListener());
    }
  }

  private boolean isAllowedMissingInput(SkyKey fileSkyKey) {
    return allowedMissingInputs.apply(((RootedPath) fileSkyKey.argument()).getRelativePath());
  }

  private static FileArtifactValue missingInputFile(
      Artifact artifact, boolean mandatory, Exception failure, EventHandler reporter)
      throws MissingInputFileException {
    if (!mandatory) {
      return FileArtifactValue.MISSING_FILE_MARKER;
    }
    throw makeMissingInputFileExn(artifact, mandatory, failure, reporter);
  }

  private static MissingInputFileException makeMissingInputFileExn(Artifact artifact,
      boolean mandatory, Exception failure, EventHandler reporter) {
    String extraMsg = (failure == null) ? "" : (":" + failure.getMessage());
    MissingInputFileException ex = new MissingInputFileException(
        constructErrorMessage(artifact) + extraMsg, null);
    if (mandatory) {
      reporter.handle(Event.error(ex.getLocation(), ex.getMessage()));
    }
    return ex;
  }

  // Non-aggregating artifact -- should contain at most one piece of artifact data.
  // data may be null if and only if artifact is a middleman artifact.
  private static FileArtifactValue createSimpleFileArtifactValue(Artifact artifact,
      Action generatingAction, ActionExecutionValue actionValue, Environment env)
      throws ArtifactFunctionException {
    FileArtifactValue value = actionValue.getArtifactValue(artifact);
    if (value != null) {
      return value;
    }
    // Middleman artifacts have no corresponding files, so their ArtifactValues should have already
    // been constructed during execution of the action.
    Preconditions.checkState(!artifact.isMiddlemanArtifact(), artifact);
    FileValue data = Preconditions.checkNotNull(actionValue.getData(artifact),
        "%s %s", artifact, actionValue);
    Preconditions.checkNotNull(data.getDigest(),
          "Digest should already have been calculated for %s (%s)", artifact, data);

    try {
      return FileArtifactValue.create(artifact, data);
    } catch (IOException e) {
      ActionExecutionException ex = new ActionExecutionException(e, generatingAction,
          /*catastrophe=*/false);
      env.getListener().handle(Event.error(ex.getLocation(), ex.getMessage()));
      // This is a transient error since we did the work that led to the IOException.
      throw new ArtifactFunctionException(ex, Transience.TRANSIENT);
    }
  }

  private static AggregatingArtifactValue createAggregatingValue(
      Artifact artifact,
      ActionAnalysisMetadata action,
      FileArtifactValue value,
      SkyFunction.Environment env)
      throws InterruptedException {
    // This artifact aggregates other artifacts. Keep track of them so callers can find them.
    ImmutableList.Builder<Pair<Artifact, FileArtifactValue>> inputs = ImmutableList.builder();
    for (Map.Entry<SkyKey, SkyValue> entry :
        env.getValues(ArtifactSkyKey.mandatoryKeys(action.getInputs())).entrySet()) {
      Artifact input = ArtifactSkyKey.artifact(entry.getKey());
      SkyValue inputValue = entry.getValue();
      Preconditions.checkNotNull(inputValue, "%s has null dep %s", artifact, input);
      if (!(inputValue instanceof FileArtifactValue)) {
        // We do not recurse in aggregating middleman artifacts.
        Preconditions.checkState(!(inputValue instanceof AggregatingArtifactValue),
            "%s %s %s", artifact, action, inputValue);
        continue;
      }
      inputs.add(Pair.of(input, (FileArtifactValue) inputValue));
    }
    return new AggregatingArtifactValue(inputs.build(), value);
  }

  /**
   * Returns whether this value needs to contain the data of all its inputs. Currently only tests to
   * see if the action is an aggregating middleman action. However, may include runfiles middleman
   * actions and Fileset artifacts in the future.
   */
  private static boolean isAggregatingValue(ActionAnalysisMetadata action) {
    return action.getActionType() == MiddlemanType.AGGREGATING_MIDDLEMAN;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((OwnedArtifact) skyKey.argument()).getArtifact().getOwner());
  }

  private static ActionAnalysisMetadata extractActionFromArtifact(
      Artifact artifact, SkyFunction.Environment env) throws InterruptedException {
    ArtifactOwner artifactOwner = artifact.getArtifactOwner();

    Preconditions.checkState(artifactOwner instanceof ActionLookupKey, "", artifact, artifactOwner);
    SkyKey actionLookupKey = ActionLookupValue.key((ActionLookupKey) artifactOwner);
    ActionLookupValue value = (ActionLookupValue) env.getValue(actionLookupKey);
    if (value == null) {
      Preconditions.checkState(artifactOwner == CoverageReportValue.ARTIFACT_OWNER,
          "Not-yet-present artifact owner: %s", artifactOwner);
      return null;
    }
    // The value should already exist (except for the coverage report action output artifacts):
    // ConfiguredTargetValues were created during the analysis phase, and BuildInfo*Values
    // were created during the first analysis of a configured target.
    Preconditions.checkNotNull(value,
        "Owner %s of %s not in graph %s", artifactOwner, artifact, actionLookupKey);

    ActionAnalysisMetadata action = value.getGeneratingAction(artifact);
    if (artifact.hasParent()) {
      // We are trying to resolve the generating action for a TreeFileArtifact. It may not have
      // a generating action in the action graph at analysis time. In that case, we get the
      // generating action for its parent TreeArtifact, which contains this TreeFileArtifact.
      if (action == null) {
        action = value.getGeneratingAction(artifact.getParent());
      }
    }

    return Preconditions.checkNotNull(action,
        "Value %s does not contain generating action of %s", value, artifact);
  }

  private static Iterable<TreeFileArtifact> findActionOutputsWithMatchingParent(
      Action action, Artifact treeArtifact) {
    ImmutableList.Builder<TreeFileArtifact> matchingOutputs = ImmutableList.builder();
    for (Artifact output : action.getOutputs()) {
      Preconditions.checkState(output.hasParent(), "%s must be a TreeFileArtifact", output);
      if (output.getParent().equals(treeArtifact)) {
        matchingOutputs.add((TreeFileArtifact) output);
      }
    }
    return matchingOutputs.build();
  }

  private static final class ArtifactFunctionException extends SkyFunctionException {
    ArtifactFunctionException(MissingInputFileException e, Transience transience) {
      super(e, transience);
    }

    ArtifactFunctionException(ActionExecutionException e, Transience transience) {
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
}
