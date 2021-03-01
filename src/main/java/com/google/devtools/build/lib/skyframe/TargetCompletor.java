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
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.CompletionFunction.Completor;
import com.google.devtools.build.lib.skyframe.TargetCompletionValue.TargetCompletionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import javax.annotation.Nullable;

/** Manages completing builds for configured targets. */
class TargetCompletor
    implements Completor<
        ConfiguredTargetValue,
        TargetCompletionValue,
        TargetCompletionKey,
        ConfiguredTargetAndData> {
  static SkyFunction targetCompletionFunction(
      PathResolverFactory pathResolverFactory,
      SkyframeActionExecutor skyframeActionExecutor,
      MetadataConsumerForMetrics.FilesMetricConsumer topLevelArtifactsMetric) {
    return new CompletionFunction<>(
        pathResolverFactory,
        new TargetCompletor(),
        skyframeActionExecutor,
        topLevelArtifactsMetric);
  }

  @Override
  public Event getRootCauseError(
      ConfiguredTargetValue ctValue, TargetCompletionKey key, LabelCause rootCause, Environment env)
      throws InterruptedException {
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(ctValue.getConfiguredTarget(), env);
    return Event.error(
        configuredTargetAndData == null ? null : configuredTargetAndData.getTarget().getLocation(),
        String.format("%s: %s", key.actionLookupKey().getLabel(), rootCause.getMessage()));
  }

  @Override
  @Nullable
  public MissingInputFileException getMissingFilesException(
      ConfiguredTargetValue value, TargetCompletionKey key, int missingCount, Environment env)
      throws InterruptedException {
    ConfiguredTargetAndData configuredTargetAndData =
        ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(value.getConfiguredTarget(), env);
    if (configuredTargetAndData == null) {
      return null;
    }
    return new MissingInputFileException(
        FailureDetail.newBuilder()
            .setMessage(
                String.format(
                    "%s %d input file(s) do not exist",
                    configuredTargetAndData.getTarget().getLocation(), missingCount))
            .setExecution(Execution.newBuilder().setCode(Code.SOURCE_INPUT_MISSING))
            .build(),
        configuredTargetAndData.getTarget().getLocation());
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
      ConfiguredTargetValue value,
      NestedSet<Cause> rootCauses,
      CompletionContext ctx,
      ImmutableMap<String, ArtifactsInOutputGroup> outputs,
      ConfiguredTargetAndData configuredTargetAndData) {
    return TargetCompleteEvent.createFailed(configuredTargetAndData, ctx, rootCauses, outputs);
  }

  @Override
  @Nullable
  public ExtendedEventHandler.Postable createSucceeded(
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
          artifactsToBuild.getAllArtifactsByOutputGroup());
    } else {
      return TargetCompleteEvent.successfulBuild(
          configuredTargetAndData,
          completionContext,
          artifactsToBuild.getAllArtifactsByOutputGroup());
    }
  }
}
