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
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.actions.ActionTemplate;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.ArtifactSkyKey.OwnedArtifact;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** A builder of values for {@link ArtifactSkyKey} keys. */
class ArtifactFunction implements SkyFunction {

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

    SkyKey actionLookupKey = getActionLookupKey(artifact);
    ActionLookupValue actionLookupValue = getActionLookupValue(actionLookupKey, env, artifact);
    if (actionLookupValue == null) {
      return null;
    }
    Integer actionIndex = actionLookupValue.getGeneratingActionIndex(artifact);
    if (artifact.hasParent() && actionIndex == null) {
      // If a TreeFileArtifact is created by a templated action, then it should have the proper
      // reference to its owner. However, if it was created as part of a directory, by the first
      // TreeArtifact-generating action in a chain, then its parent's generating action also
      // generated it. This catches that case.
      actionIndex = actionLookupValue.getGeneratingActionIndex(artifact.getParent());
    }
    Preconditions.checkNotNull(
        actionIndex, "%s %s %s", artifact, actionLookupKey, actionLookupValue);

    // If the action is an ActionTemplate, we need to expand the ActionTemplate into concrete
    // actions, execute those actions in parallel and then aggregate the action execution results.
    if (artifact.isTreeArtifact()) {
      ActionAnalysisMetadata actionMetadata =
          actionLookupValue.getIfPresentAndNotAction(actionIndex);
      if (actionMetadata instanceof ActionTemplate) {
        // Create the directory structures for the output TreeArtifact first.
        try {
          FileSystemUtils.createDirectoryAndParents(artifact.getPath());
        } catch (IOException e) {
          env.getListener()
              .handle(
                  Event.error(
                      String.format(
                          "Failed to create output directory for TreeArtifact %s: %s",
                          artifact, e.getMessage())));
          throw new ArtifactFunctionException(e, Transience.TRANSIENT);
        }

        return createTreeArtifactValueFromActionTemplate(
            (ActionTemplate<?>) actionMetadata, artifact, env);
      }
    }
    ActionExecutionValue actionValue =
        (ActionExecutionValue) env.getValue(ActionExecutionValue.key(actionLookupKey, actionIndex));
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
              actionLookupValue.getAction(actionIndex),
              "Null middleman action? %s %s %s %s",
              artifact,
              actionLookupKey,
              actionLookupValue,
              actionIndex);
      if (isAggregatingValue(action)) {
        return createAggregatingValue(artifact, action,
            actionValue.getArtifactValue(artifact), env);
      }
    }
    return createSimpleFileArtifactValue(artifact, actionValue);
  }

  private static TreeArtifactValue createTreeArtifactValueFromActionTemplate(
      final ActionTemplate<?> actionTemplate, final Artifact treeArtifact, Environment env)
          throws InterruptedException {
    // Request the list of expanded actions from the ActionTemplate.
    SkyKey templateKey = ActionTemplateExpansionValue.key(actionTemplate);
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
                  "Missing tree value: %s %s %s %s",
                  treeArtifact,
                  actionTemplate,
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
                          "No parent: %s %s %s %s",
                          artifact,
                          treeArtifact,
                          actionExecutionValue,
                          actionTemplate);
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
      throws MissingInputFileException, InterruptedException {
    SkyKey fileSkyKey = FileValue.key(RootedPath.toRootedPath(artifact.getRoot().getPath(),
        artifact.getPath()));
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
          InconsistentFilesystemException.class, FileSymlinkException.class);
    } catch (IOException | InconsistentFilesystemException | FileSymlinkException e) {
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
    try {
      return FileArtifactValue.create(artifact, fileValue);
    } catch (IOException e) {
      throw makeMissingInputFileException(artifact, mandatory, e, env.getListener());
    }
  }

  private static MissingInputFileException makeMissingInputFileException(
      Artifact artifact, boolean mandatory, Exception failure, EventHandler reporter) {
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
  private static FileArtifactValue createSimpleFileArtifactValue(
      Artifact artifact, ActionExecutionValue actionValue) {
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
    // Directories are special-cased because their mtimes are used, so should have been constructed
    // during execution of the action (in ActionMetadataHandler#maybeStoreAdditionalData).
    Preconditions.checkState(data.isFile(), "Unexpected not file %s (%s)", artifact, data);
    return FileArtifactValue.createNormalFile(data.getDigest(), data.getSize());
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

  @VisibleForTesting
  static SkyKey getActionLookupKey(Artifact artifact) {
    ArtifactOwner artifactOwner = artifact.getArtifactOwner();

    Preconditions.checkState(artifactOwner instanceof ActionLookupKey, "", artifact, artifactOwner);
    return ActionLookupValue.key((ActionLookupKey) artifactOwner);
  }

  @Nullable
  private static ActionLookupValue getActionLookupValue(
      SkyKey actionLookupKey, SkyFunction.Environment env, Artifact artifact)
      throws InterruptedException {
    ActionLookupValue value = (ActionLookupValue) env.getValue(actionLookupKey);
    if (value == null) {
      ArtifactOwner artifactOwner = artifact.getArtifactOwner();
      Preconditions.checkState(
          artifactOwner == CoverageReportValue.ARTIFACT_OWNER,
          "Not-yet-present artifact owner: %s (%s %s)",
          artifactOwner,
          artifact,
          actionLookupKey);
      return null;
    }
    return value;
  }

  private static final class ArtifactFunctionException extends SkyFunctionException {
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
}
