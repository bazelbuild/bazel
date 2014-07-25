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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.PostInitializationActionOwner;
import com.google.devtools.build.lib.view.TopLevelArtifactContext;
import com.google.devtools.build.lib.view.TopLevelArtifactHelper;
import com.google.devtools.build.lib.view.actions.TargetCompletionMiddlemanAction;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import javax.annotation.Nullable;

/**
 * TargetCompletionFunction constructs a TargetCompletionActionValue. This works around two
 * problems with an alternative design which would create the completion action in the
 * ConfiguredTargetFunction directly:
 * - creating the actions eagerly could be a memory concern
 * - the set of artifacts that determines whether a top-level target is complete depends on
 * attributes that are not otherwise part of the BuildConfiguration (eg, --compile_only).
 *
 * TODO(bazel-team): Make target-completion not based on Artifacts and Actions.
 */

final class TargetCompletionFunction implements SkyFunction {

  private final SkyframeExecutor.BuildViewProvider buildViewProvider;

  TargetCompletionFunction(SkyframeExecutor.BuildViewProvider buildViewProvider) {
    this.buildViewProvider = buildViewProvider;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
    LabelAndConfiguration lac = (LabelAndConfiguration) skyKey.argument();
    ConfiguredTargetValue ctValue = (ConfiguredTargetValue) env.getValue(
        ConfiguredTargetValue.key(lac.getLabel(), lac.getConfiguration()));
    if (ctValue == null) {
      return null;
    }

    TopLevelArtifactContext topLevelContext = BuildVariableValue.TOP_LEVEL_CONTEXT.get(env);
    if (topLevelContext == null) {
      return null;
    }

    ConfiguredTarget configuredTarget = ctValue.getConfiguredTarget();
    ArtifactFactory factory = buildViewProvider.getSkyframeBuildView().getArtifactFactory();
    Artifact middleman = factory.getDerivedArtifact(
        TopLevelArtifactHelper.getMiddlemanRelativePath(lac.getLabel()),
        lac.getConfiguration().getMiddlemanDirectory(), lac);
    return new TargetCompletionActionValue(new TargetCompletionMiddlemanAction(configuredTarget,
        new PostInitializationActionOwner(configuredTarget),
        TopLevelArtifactHelper.getAllArtifactsToBuild(configuredTarget, topLevelContext),
        middleman));
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((LabelAndConfiguration) skyKey.argument()).getLabel());
  }
}
