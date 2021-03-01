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
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue.AspectCompletionKey;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.CompletionFunction.Completor;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import javax.annotation.Nullable;

/** Manages completing builds for aspects. */
class AspectCompletor
    implements Completor<AspectValue, AspectCompletionValue, AspectCompletionKey, BuildEventId> {

  static SkyFunction aspectCompletionFunction(
      PathResolverFactory pathResolverFactory,
      SkyframeActionExecutor skyframeActionExecutor,
      MetadataConsumerForMetrics.FilesMetricConsumer topLevelArtifactsMetric) {
    return new CompletionFunction<>(
        pathResolverFactory,
        new AspectCompletor(),
        skyframeActionExecutor,
        topLevelArtifactsMetric);
  }

  @Override
  public Event getRootCauseError(
      AspectValue value, AspectCompletionKey key, LabelCause rootCause, Environment env) {
    AspectKey aspectKey = key.actionLookupKey();
    return Event.error(
        value.getLocation(),
        String.format(
            "%s, aspect %s: %s",
            aspectKey.getLabel(), aspectKey.getAspectClass().getName(), rootCause.getMessage()));
  }

  @Override
  public MissingInputFileException getMissingFilesException(
      AspectValue value, AspectCompletionKey key, int missingCount, Environment env) {
    AspectKey aspectKey = key.actionLookupKey();
    String message =
        String.format(
            "%s, aspect %s %d input file(s) do not exist",
            aspectKey.getLabel(), aspectKey.getAspectClass().getName(), missingCount);
    return new MissingInputFileException(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(Code.SOURCE_INPUT_MISSING))
            .build(),
        value.getLocation());
  }

  @Override
  public AspectCompletionValue getResult() {
    return AspectCompletionValue.INSTANCE;
  }

  @Override
  @Nullable
  public BuildEventId getFailureData(AspectCompletionKey key, AspectValue value, Environment env)
      throws InterruptedException {
    return getConfigurationEventIdFromAspectKey(key.actionLookupKey(), env);
  }

  @Override
  public ExtendedEventHandler.Postable createFailed(
      AspectValue value,
      NestedSet<Cause> rootCauses,
      CompletionContext ctx,
      ImmutableMap<String, ArtifactsInOutputGroup> outputs,
      BuildEventId configurationEventId) {
    return AspectCompleteEvent.createFailed(value, ctx, rootCauses, configurationEventId, outputs);
  }

  @Nullable
  private BuildEventId getConfigurationEventIdFromAspectKey(AspectKey aspectKey, Environment env)
      throws InterruptedException {
    if (aspectKey.getBaseConfiguredTargetKey().getConfigurationKey() == null) {
      return BuildEventIdUtil.nullConfigurationId();
    } else {
      BuildConfigurationValue buildConfigurationValue =
          (BuildConfigurationValue)
              env.getValue(aspectKey.getBaseConfiguredTargetKey().getConfigurationKey());
      if (buildConfigurationValue == null) {
        return null;
      }
      return buildConfigurationValue.getConfiguration().getEventId();
    }
  }

  @Override
  public ExtendedEventHandler.Postable createSucceeded(
      AspectCompletionKey skyKey,
      AspectValue value,
      CompletionContext completionContext,
      ArtifactsToBuild artifactsToBuild,
      Environment env)
      throws InterruptedException {
    AspectKey aspectKey = skyKey.actionLookupKey();
    BuildEventId configurationEventId = getConfigurationEventIdFromAspectKey(aspectKey, env);
    if (configurationEventId == null) {
      return null;
    }

    return AspectCompleteEvent.createSuccessful(
        value,
        completionContext,
        artifactsToBuild.getAllArtifactsByOutputGroup(),
        configurationEventId);
  }
}
