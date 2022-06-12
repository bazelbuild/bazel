// Copyright 2021 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.ExtraActionArtifactsProvider;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue.AspectCompletionKey;
import com.google.devtools.build.lib.skyframe.ToplevelStarlarkAspectFunction.TopLevelAspectsValue;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Drives the analysis & execution of an ActionLookupKey, which is wrapped inside a BuildDriverKey.
 */
public class BuildDriverFunction implements SkyFunction {

  /**
   * From the ConfiguredTarget/Aspect keys, get the top-level artifacts. Then evaluate them together
   * with the appropriate CompletionFunctions. This is the bridge between the conceptual analysis &
   * execution phases.
   *
   * <p>TODO(b/199053098): implement build-info, build-changelist, coverage & exception handling.
   */
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    ActionLookupKey actionLookupKey = ((BuildDriverKey) skyKey).getActionLookupKey();
    TopLevelArtifactContext topLevelArtifactContext =
        ((BuildDriverKey) skyKey).getTopLevelArtifactContext();

    // Why SkyValue and not ActionLookupValue? The evaluation of some ActionLookupKey can result in
    // classes that don't implement
    // ActionLookupValue (e.g. ConfiguredTargetKey -> NonRuleConfiguredTargetValue).
    SkyValue topLevelSkyValue = env.getValue(actionLookupKey);

    if (env.valuesMissing()) {
      return null;
    }
    ImmutableSet.Builder<Artifact> artifactsToBuild = ImmutableSet.builder();

    Preconditions.checkState(
        topLevelSkyValue instanceof ConfiguredTargetValue
            || topLevelSkyValue instanceof TopLevelAspectsValue);
    if (topLevelSkyValue instanceof ConfiguredTargetValue) {
      ConfiguredTarget configuredTarget =
          ((ConfiguredTargetValue) topLevelSkyValue).getConfiguredTarget();
      addExtraActionsIfRequested(
          configuredTarget.getProvider(ExtraActionArtifactsProvider.class), artifactsToBuild);
      env.getValues(
          Iterables.concat(
              artifactsToBuild.build(),
              Collections.singletonList(
                  TargetCompletionValue.key(
                      (ConfiguredTargetKey) actionLookupKey, topLevelArtifactContext, false))));
    } else if (topLevelSkyValue instanceof TopLevelAspectsValue) {
      List<SkyKey> aspectCompletionKeys = new ArrayList<>();
      for (SkyValue aspectValue :
          ((TopLevelAspectsValue) topLevelSkyValue).getTopLevelAspectsValues()) {
        addExtraActionsIfRequested(
            ((AspectValue) aspectValue)
                .getConfiguredAspect()
                .getProvider(ExtraActionArtifactsProvider.class),
            artifactsToBuild);
        aspectCompletionKeys.add(
            AspectCompletionKey.create(
                ((AspectValue) aspectValue).getKey(), topLevelArtifactContext));
      }
      env.getValues(Iterables.concat(artifactsToBuild.build(), aspectCompletionKeys));
    }

    if (env.valuesMissing()) {
      return null;
    }

    return new BuildDriverValue(topLevelSkyValue);
  }

  private void addExtraActionsIfRequested(
      ExtraActionArtifactsProvider provider, ImmutableSet.Builder<Artifact> artifactsToBuild) {
    if (provider != null) {
      addArtifactsToBuilder(
          provider.getTransitiveExtraActionArtifacts().toList(), artifactsToBuild, null);
    }
  }

  private static void addArtifactsToBuilder(
      List<? extends Artifact> artifacts,
      ImmutableSet.Builder<Artifact> builder,
      RegexFilter filter) {
    for (Artifact artifact : artifacts) {
      if (filter.isIncluded(artifact.getOwnerLabel().toString())) {
        builder.add(artifact);
      }
    }
  }
}
