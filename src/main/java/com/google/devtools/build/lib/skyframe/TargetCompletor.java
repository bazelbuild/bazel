// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.CompletionFunction.Completor;
import com.google.devtools.build.lib.skyframe.TargetCompletionValue.TargetCompletionKey;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Manages completing builds for configured targets. */
class TargetCompletor implements Completor<ConfiguredTargetValue, TargetCompletionValue> {

  public static SkyFunction targetCompletionFunction(
      PathResolverFactory pathResolverFactory, Supplier<Path> execRootSupplier) {
    return new CompletionFunction<>(pathResolverFactory, new TargetCompletor(), execRootSupplier);
  }

  @Override
  public Event getRootCauseError(ConfiguredTargetValue ctValue, Cause rootCause, Environment env)
      throws InterruptedException {
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(ctValue.getConfiguredTarget(), env);
    return Event.error(
        configuredTargetAndData == null ? null : configuredTargetAndData.getTarget().getLocation(),
        String.format(
            "%s: missing input file '%s'", ctValue.getConfiguredTarget().getLabel(), rootCause));
  }

  @Override
  @Nullable
  public MissingInputFileException getMissingFilesException(
      ConfiguredTargetValue value, int missingCount, Environment env) throws InterruptedException {
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(value.getConfiguredTarget(), env);
    if (configuredTargetAndData == null) {
      return null;
    }
    return new MissingInputFileException(
        configuredTargetAndData.getTarget().getLocation()
            + " "
            + missingCount
            + " input file(s) do not exist",
        configuredTargetAndData.getTarget().getLocation());
  }

  @Override
  public TargetCompletionValue getResult() {
    return TargetCompletionValue.INSTANCE;
  }

  @Override
  @Nullable
  public ExtendedEventHandler.Postable createFailed(
      ConfiguredTargetValue value,
      NestedSet<Cause> rootCauses,
      NestedSet<ArtifactsInOutputGroup> outputs,
      Environment env,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    ConfiguredTarget target = value.getConfiguredTarget();
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(target, env);
    if (configuredTargetAndData == null) {
      return null;
    }
    return TargetCompleteEvent.createFailed(configuredTargetAndData, rootCauses, outputs);
  }

  @Override
  @Nullable
  public ExtendedEventHandler.Postable createSucceeded(
      SkyKey skyKey,
      ConfiguredTargetValue value,
      CompletionContext completionContext,
      ArtifactsToBuild artifactsToBuild,
      Environment env)
      throws InterruptedException {
    ConfiguredTarget target = value.getConfiguredTarget();
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(target, env);
    if (configuredTargetAndData == null) {
      return null;
    }
    if (((TargetCompletionKey) skyKey.argument()).willTest()) {
      return TargetCompleteEvent.successfulBuildSchedulingTest(
          configuredTargetAndData,
          completionContext,
          artifactsToBuild.getAllArtifactsByOutputGroup());
    } else {
      return TargetCompleteEvent.successfulBuild(
          configuredTargetAndData,
          completionContext,
          artifactsToBuild.getAllArtifactsByOutputGroup());
    }
  }
}
