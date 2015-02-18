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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
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
 * TargetCompletionFunction builds the artifactsToBuild collection of a {@link ConfiguredTarget}.
 */
public final class TargetCompletionFunction implements SkyFunction {

  private final AtomicReference<EventBus> eventBusRef;

  public TargetCompletionFunction(AtomicReference<EventBus> eventBusRef) {
    this.eventBusRef = eventBusRef;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws TargetCompletionFunctionException {
    LabelAndConfiguration lac = (LabelAndConfiguration) skyKey.argument();
    ConfiguredTargetValue ctValue = (ConfiguredTargetValue)
        env.getValue(ConfiguredTargetValue.key(lac.getLabel(), lac.getConfiguration()));
    TopLevelArtifactContext topLevelContext = PrecomputedValue.TOP_LEVEL_CONTEXT.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps =
        env.getValuesOrThrow(ArtifactValue.mandatoryKeys(
            TopLevelArtifactHelper.getAllArtifactsToBuild(
                ctValue.getConfiguredTarget(), topLevelContext).getAllArtifacts()),
            MissingInputFileException.class, ActionExecutionException.class);

    int missingCount = 0;
    ActionExecutionException firstActionExecutionException = null;
    MissingInputFileException missingInputException = null;
    NestedSetBuilder<Label> rootCausesBuilder = NestedSetBuilder.stableOrder();
    for (Map.Entry<SkyKey, ValueOrException2<MissingInputFileException,
        ActionExecutionException>> depsEntry : inputDeps.entrySet()) {
      Artifact input = ArtifactValue.artifact(depsEntry.getKey());
      try {
        depsEntry.getValue().get();
      } catch (MissingInputFileException e) {
        missingCount++;
        if (input.getOwner() != null) {
          rootCausesBuilder.add(input.getOwner());
          env.getListener().handle(Event.error(
              ctValue.getConfiguredTarget().getTarget().getLocation(),
              String.format("%s: missing input file '%s'",
                  lac.getLabel(), input.getOwner())));
        }
      } catch (ActionExecutionException e) {
        rootCausesBuilder.addTransitive(e.getRootCauses());
        if (firstActionExecutionException == null) {
          firstActionExecutionException = e;
        }
      }
    }

    if (missingCount > 0) {
      missingInputException = new MissingInputFileException(
          ctValue.getConfiguredTarget().getTarget().getLocation() + " " + missingCount
          + " input file(s) do not exist", ctValue.getConfiguredTarget().getTarget().getLocation());
    }

    NestedSet<Label> rootCauses = rootCausesBuilder.build();
    if (!rootCauses.isEmpty()) {
      eventBusRef.get().post(
          TargetCompleteEvent.createFailed(ctValue.getConfiguredTarget(), rootCauses));
      if (firstActionExecutionException != null) {
        throw new TargetCompletionFunctionException(firstActionExecutionException);
      } else {
        throw new TargetCompletionFunctionException(missingInputException);
      }
    }

    return env.valuesMissing() ? null : new TargetCompletionValue(ctValue.getConfiguredTarget());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((LabelAndConfiguration) skyKey.argument()).getLabel());
  }

  private static final class TargetCompletionFunctionException extends SkyFunctionException {

    private final ActionExecutionException actionException;

    public TargetCompletionFunctionException(ActionExecutionException e) {
      super(e, Transience.PERSISTENT);
      this.actionException = e;
    }

    public TargetCompletionFunctionException(MissingInputFileException e) {
      super(e, Transience.TRANSIENT);
      this.actionException = null;
    }

    @Override
    public boolean isCatastrophic() {
      return actionException != null && actionException.isCatastrophe();
    }
  }
}
