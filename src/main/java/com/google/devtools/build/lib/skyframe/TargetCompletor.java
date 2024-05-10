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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.CompletionContext.PathResolverFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.CompletionFunction.Completor;
import com.google.devtools.build.lib.skyframe.TargetCompletionValue.TargetCompletionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** Manages completing builds for configured targets. */
class TargetCompletor
    implements Completor<
        ConfiguredTargetValue,
        TargetCompletionValue,
        TargetCompletionKey,
        ConfiguredTargetAndData> {

  private final SkyframeActionExecutor skyframeActionExecutor;

  static SkyFunction targetCompletionFunction(
      PathResolverFactory pathResolverFactory,
      SkyframeActionExecutor skyframeActionExecutor,
      MetadataConsumerForMetrics.FilesMetricConsumer topLevelArtifactsMetric,
      BugReporter bugReporter,
      Supplier<Boolean> isSkymeld) {
    return new CompletionFunction<>(
        pathResolverFactory,
        new TargetCompletor(skyframeActionExecutor),
        skyframeActionExecutor,
        topLevelArtifactsMetric,
        bugReporter,
        isSkymeld);
  }

  private TargetCompletor(SkyframeActionExecutor announceTargetSummaries) {
    // SkyframeActionExecutor.options not populated yet, so store and query lazily later
    this.skyframeActionExecutor = announceTargetSummaries;
  }

  @Override
  public Event getRootCauseError(
      ConfiguredTargetValue ctValue, TargetCompletionKey key, LabelCause rootCause, Environment env)
      throws InterruptedException {
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(ctValue.getConfiguredTarget(), env);
    return Event.error(
        configuredTargetAndData == null ? null : configuredTargetAndData.getLocation(),
        String.format("%s: %s", key.actionLookupKey().getLabel(), rootCause.getMessage()));
  }

  @Nullable
  @Override
  public Location getLocationIdentifier(
      ConfiguredTargetValue value, TargetCompletionKey key, Environment env)
      throws InterruptedException {
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(value.getConfiguredTarget(), env);
    return configuredTargetAndData == null ? null : configuredTargetAndData.getLocation();
  }

  @Override
  public TargetCompletionValue getResult() {
    return TargetCompletionValue.INSTANCE;
  }

  @Override
  @Nullable
  public ConfiguredTargetAndData getFailureData(
      TargetCompletionKey key, ConfiguredTargetValue value, Environment env)
      throws InterruptedException {
    ConfiguredTarget target = value.getConfiguredTarget();
    return ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(target, env);
  }

  @Override
  @Nullable
  public ExtendedEventHandler.Postable createFailed(
      TargetCompletionKey skyKey,
      NestedSet<Cause> rootCauses,
      CompletionContext ctx,
      ImmutableMap<String, ArtifactsInOutputGroup> outputs,
      ConfiguredTargetAndData configuredTargetAndData) {
    return TargetCompleteEvent.createFailed(
        configuredTargetAndData,
        ctx,
        rootCauses,
        outputs,
        skyframeActionExecutor.publishTargetSummaries());
  }

  @Override
  @Nullable
  public TargetCompleteEvent createSucceeded(
      TargetCompletionKey skyKey,
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
    if (skyKey.willTest()) {
      return TargetCompleteEvent.successfulBuildSchedulingTest(
          configuredTargetAndData,
          completionContext,
          artifactsToBuild.getAllArtifactsByOutputGroup(),
          skyframeActionExecutor.publishTargetSummaries());
    } else {
      if (target instanceof InputFileConfiguredTarget) {
        env.getListener()
            .handle(
                Event.warn(
                    configuredTargetAndData.getLocation(),
                    target.getLabel()
                        + " is a source file, nothing will be built for it. If you want to build a"
                        + " target that consumes this file, try --compile_one_dependency"));
      }
      return TargetCompleteEvent.successfulBuild(
          configuredTargetAndData,
          completionContext,
          artifactsToBuild.getAllArtifactsByOutputGroup(),
          skyframeActionExecutor.publishTargetSummaries());
    }
  }
}
