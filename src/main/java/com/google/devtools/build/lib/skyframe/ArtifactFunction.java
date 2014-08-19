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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.MissingArtifactEvent;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.skyframe.ArtifactValue.OwnedArtifact;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A builder for {@link ArtifactValue}s.
 */
class ArtifactFunction implements SkyFunction {

  private final AtomicReference<EventBus> eventBus;
  private final Predicate<PathFragment> allowedMissingInputs;

  ArtifactFunction(AtomicReference<EventBus> eventBus,
      Predicate<PathFragment> allowedMissingInputs) {
    this.eventBus = eventBus;
    this.allowedMissingInputs = allowedMissingInputs;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws ArtifactFunctionException {
    OwnedArtifact ownedArtifact = (OwnedArtifact) skyKey.argument();
    Artifact artifact = ownedArtifact.getArtifact();
    if (artifact.isSourceArtifact()) {
      try {
        return createSourceValue(artifact, ownedArtifact.isMandatory(), env);
      } catch (MissingInputFileException e) {
        if (eventBus.get() != null) {
          eventBus.get().post(new MissingArtifactEvent(artifact.getOwner()));
        }
        throw new ArtifactFunctionException(skyKey, e);
      }
    }

    Action action = extractActionFromArtifact(artifact, env);
    if (action == null) {
      return null;
    }

    ActionExecutionValue actionValue =
        (ActionExecutionValue) env.getValue(ActionExecutionValue.key(action));
    if (actionValue == null) {
      return null;
    }

    if (!isAggregatingValue(action)) {
      try {
        return createSimpleValue(artifact, actionValue);
      } catch (IOException e) {
        ActionExecutionException ex = new ActionExecutionException(e, action,
            /*catastrophe=*/false);
        env.getListener().handle(Event.error(ex.getLocation(), ex.getMessage()));
        throw new ArtifactFunctionException(skyKey, ex);
      }
    } else {
      return createAggregatingValue(artifact, action, actionValue.getArtifactValue(artifact), env);
    }
  }

  private ArtifactValue createSourceValue(Artifact artifact, boolean mandatory, Environment env)
      throws MissingInputFileException {
    SkyKey fileSkyKey = FileValue.key(artifact);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, Exception.class);
    } catch (IOException | InconsistentFilesystemException | FileSymlinkCycleException e) {
      return missingInputFile(artifact, mandatory, e, env.getListener());
    } catch (Exception e) {
      // Can't get here.
      throw new IllegalStateException(e);
    }
    if (fileValue == null) {
      return null;
    }
    if (!fileValue.exists()) {
      if (allowedMissingInputs.apply(((RootedPath) fileSkyKey.argument()).getRelativePath())) {
        return FileArtifactValue.MISSING_FILE_MARKER;
      } else {
        return missingInputFile(artifact, mandatory, null, env.getListener());
      }
    }
    try {
      return FileArtifactValue.create(artifact, fileValue);
    } catch (IOException e) {
      return missingInputFile(artifact, mandatory, e, env.getListener());
    }
  }

  private static ArtifactValue missingInputFile(Artifact artifact, boolean mandatory,
                                               Exception failure,
                                               EventHandler reporter)
      throws MissingInputFileException {
    if (!mandatory) {
      return FileArtifactValue.MISSING_FILE_MARKER;
    }
    String extraMsg = (failure == null) ? "" : (":" + failure.getMessage());
    MissingInputFileException ex = new MissingInputFileException(
        constructErrorMessage(artifact) + extraMsg, null);
    reporter.handle(Event.error(ex.getLocation(), ex.getMessage()));
    throw ex;
  }

  // Non-aggregating artifact -- should contain at most one piece of artifact data.
  // data may be null if and only if artifact is a middleman artifact.
  private ArtifactValue createSimpleValue(Artifact artifact, ActionExecutionValue actionValue)
      throws IOException {
    ArtifactValue value = actionValue.getArtifactValue(artifact);
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
    return FileArtifactValue.create(artifact, data);
  }

  private AggregatingArtifactValue createAggregatingValue(Artifact artifact, Action action,
      FileArtifactValue value, SkyFunction.Environment env) {
    // This artifact aggregates other artifacts. Keep track of them so callers can find them.
    ImmutableList.Builder<Pair<Artifact, FileArtifactValue>> inputs = ImmutableList.builder();
    for (Map.Entry<SkyKey, SkyValue> entry :
        env.getValues(ArtifactValue.mandatoryKeys(action.getInputs())).entrySet()) {
      Artifact input = ArtifactValue.artifact(entry.getKey());
      ArtifactValue inputValue = (ArtifactValue) entry.getValue();
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
  private static boolean isAggregatingValue(Action action) {
    return action.getActionType() == MiddlemanType.AGGREGATING_MIDDLEMAN;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((OwnedArtifact) skyKey.argument()).getArtifact().getOwner());
  }

  private Action extractActionFromArtifact(Artifact artifact, SkyFunction.Environment env) {
    ArtifactOwner artifactOwner = artifact.getArtifactOwner();

    Preconditions.checkState(artifactOwner instanceof ActionLookupKey, "", artifact, artifactOwner);
    SkyKey actionLookupKey = ActionLookupValue.key((ActionLookupKey) artifactOwner);
    ActionLookupValue value = (ActionLookupValue) env.getValue(actionLookupKey);
    if (value == null) {
      // TargetCompletionActionValues are created on demand. All others should already exist --
      // ConfiguredTargetValues were created during the analysis phase, and BuildInfo*Values were
      // created during the first analysis of a configured target.
      Preconditions.checkState(artifactOwner instanceof TargetCompletionKey,
          "Owner %s of %s not in graph %s", artifactOwner, artifact, actionLookupKey);
      return null;
    }
    return Preconditions.checkNotNull(value.getGeneratingAction(artifact),
          "Value %s does not contain generating action of %s", value, artifact);
  }

  private static final class ArtifactFunctionException extends SkyFunctionException {
    ArtifactFunctionException(SkyKey key, MissingInputFileException e) {
      super(key, e);
    }

    ArtifactFunctionException(SkyKey key, ActionExecutionException e) {
      super(key, e);
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
