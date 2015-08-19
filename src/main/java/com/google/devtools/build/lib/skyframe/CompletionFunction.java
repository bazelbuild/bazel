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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * CompletionFunction builds the artifactsToBuild collection of a {@link ConfiguredTarget}.
 */
public final class CompletionFunction<TValue extends SkyValue, TResult extends SkyValue>
    implements SkyFunction {

  /**
   * A strategy for completing the build.
   */
  public interface Completor<TValue, TResult extends SkyValue> {

    /**
     * Obtains an analysis result value from environment.
     */
    TValue getValueFromSkyKey(SkyKey skyKey, Environment env);

    /**
     * Returns all artefacts that need to be built to complete the {@code value}
     */
    ArtifactsToBuild getAllArtifactsToBuild(TValue value, TopLevelArtifactContext context);

    /**
     * Creates an event reporting an absent input artifact.
     */
    Event getRootCauseError(TValue value, Label rootCause);

    /**
     * Creates an error message reporting {@code missingCount} missing input files.
     */
    MissingInputFileException getMissingFilesException(TValue value, int missingCount);

    /**
     * Creates a successful completion value.
     */
    TResult createResult(TValue value);

    /**
     * Creates a failed completion value.
     */
    SkyValue createFailed(TValue value, NestedSet<Label> rootCauses);
  }

  private static class TargetCompletor
      implements Completor<ConfiguredTargetValue, TargetCompletionValue> {
    @Override
    public ConfiguredTargetValue getValueFromSkyKey(SkyKey skyKey, Environment env) {
      LabelAndConfiguration lac = (LabelAndConfiguration) skyKey.argument();
      return (ConfiguredTargetValue)
          env.getValue(ConfiguredTargetValue.key(lac.getLabel(), lac.getConfiguration()));
    }

    @Override
    public ArtifactsToBuild getAllArtifactsToBuild(
        ConfiguredTargetValue value, TopLevelArtifactContext topLevelContext) {
      return TopLevelArtifactHelper.getAllArtifactsToBuild(
          value.getConfiguredTarget(), topLevelContext);
    }

    @Override
    public Event getRootCauseError(ConfiguredTargetValue ctValue, Label rootCause) {
      return Event.error(
          ctValue.getConfiguredTarget().getTarget().getLocation(),
          String.format(
              "%s: missing input file '%s'", ctValue.getConfiguredTarget().getLabel(), rootCause));
    }

    @Override
    public MissingInputFileException getMissingFilesException(
        ConfiguredTargetValue value, int missingCount) {
      return new MissingInputFileException(
          value.getConfiguredTarget().getTarget().getLocation()
              + " "
              + missingCount
              + " input file(s) do not exist",
          value.getConfiguredTarget().getTarget().getLocation());
    }

    @Override
    public TargetCompletionValue createResult(ConfiguredTargetValue value) {
      return new TargetCompletionValue(value.getConfiguredTarget());
    }

    @Override
    public SkyValue createFailed(ConfiguredTargetValue value, NestedSet<Label> rootCauses) {
      return TargetCompleteEvent.createFailed(value.getConfiguredTarget(), rootCauses);
    }
  }

  private static class AspectCompletor implements Completor<AspectValue, AspectCompletionValue> {
    @Override
    public AspectValue getValueFromSkyKey(SkyKey skyKey, Environment env) {
      AspectKey aspectKey = (AspectKey) skyKey.argument();
      return (AspectValue) env.getValue(AspectValue.key(aspectKey));
    }

    @Override
    public ArtifactsToBuild getAllArtifactsToBuild(
        AspectValue value, TopLevelArtifactContext topLevelArtifactContext) {
      return TopLevelArtifactHelper.getAllArtifactsToBuild(value, topLevelArtifactContext);
    }

    @Override
    public Event getRootCauseError(AspectValue value, Label rootCause) {
      return Event.error(
          value.getLocation(),
          String.format(
              "%s, aspect %s: missing input file '%s'",
              value.getLabel(),
              value.getAspect().getName(),
              rootCause));
    }

    @Override
    public MissingInputFileException getMissingFilesException(AspectValue value, int missingCount) {
      return new MissingInputFileException(
          value.getLabel()
              + ", aspect "
              + value.getAspect().getName()
              + missingCount
              + " input file(s) do not exist",
          value.getLocation());
    }

    @Override
    public AspectCompletionValue createResult(AspectValue value) {
      return new AspectCompletionValue(value);
    }

    @Override
    public SkyValue createFailed(AspectValue value, NestedSet<Label> rootCauses) {
      return AspectCompleteEvent.createFailed(value, rootCauses);
    }
  }

  public static SkyFunction targetCompletionFunction(AtomicReference<EventBus> eventBusRef) {
    return new CompletionFunction<>(eventBusRef, new TargetCompletor());
  }

  public static SkyFunction aspectCompletionFunction(AtomicReference<EventBus> eventBusRef) {
    return new CompletionFunction<>(eventBusRef, new AspectCompletor());
  }

  private final AtomicReference<EventBus> eventBusRef;
  private final Completor<TValue, TResult> completor;

  private CompletionFunction(
      AtomicReference<EventBus> eventBusRef, Completor<TValue, TResult> completor) {
    this.eventBusRef = eventBusRef;
    this.completor = completor;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws CompletionFunctionException {
    TValue value = completor.getValueFromSkyKey(skyKey, env);
    TopLevelArtifactContext topLevelContext = PrecomputedValue.TOP_LEVEL_CONTEXT.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps =
        env.getValuesOrThrow(
            ArtifactValue.mandatoryKeys(
                completor.getAllArtifactsToBuild(value, topLevelContext).getAllArtifacts()),
            MissingInputFileException.class,
            ActionExecutionException.class);

    int missingCount = 0;
    ActionExecutionException firstActionExecutionException = null;
    MissingInputFileException missingInputException = null;
    NestedSetBuilder<Label> rootCausesBuilder = NestedSetBuilder.stableOrder();
    for (Map.Entry<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>>
        depsEntry : inputDeps.entrySet()) {
      Artifact input = ArtifactValue.artifact(depsEntry.getKey());
      try {
        depsEntry.getValue().get();
      } catch (MissingInputFileException e) {
        missingCount++;
        final Label inputOwner = input.getOwner();
        if (inputOwner != null) {
          rootCausesBuilder.add(inputOwner);
          env.getListener().handle(completor.getRootCauseError(value, inputOwner));
        }
      } catch (ActionExecutionException e) {
        rootCausesBuilder.addTransitive(e.getRootCauses());
        if (firstActionExecutionException == null) {
          firstActionExecutionException = e;
        }
      }
    }

    if (missingCount > 0) {
      missingInputException = completor.getMissingFilesException(value, missingCount);
    }

    NestedSet<Label> rootCauses = rootCausesBuilder.build();
    if (!rootCauses.isEmpty()) {
      eventBusRef.get().post(completor.createFailed(value, rootCauses));
      if (firstActionExecutionException != null) {
        throw new CompletionFunctionException(firstActionExecutionException);
      } else {
        throw new CompletionFunctionException(missingInputException);
      }
    }

    return env.valuesMissing() ? null : completor.createResult(value);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((LabelAndConfiguration) skyKey.argument()).getLabel());
  }

  private static final class CompletionFunctionException extends SkyFunctionException {

    private final ActionExecutionException actionException;

    public CompletionFunctionException(ActionExecutionException e) {
      super(e, Transience.PERSISTENT);
      this.actionException = e;
    }

    public CompletionFunctionException(MissingInputFileException e) {
      super(e, Transience.TRANSIENT);
      this.actionException = null;
    }

    @Override
    public boolean isCatastrophic() {
      return actionException != null && actionException.isCatastrophe();
    }
  }
}
