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
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.CompletionFunction.Completor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Manages completing builds for aspects. */
class AspectCompletor implements Completor<AspectValue, AspectCompletionValue> {

  public static SkyFunction aspectCompletionFunction(
      PathResolverFactory pathResolverFactory, Supplier<Path> execRootSupplier) {
    return new CompletionFunction<>(pathResolverFactory, new AspectCompletor(), execRootSupplier);
  }

  @Override
  public Event getRootCauseError(AspectValue value, Cause rootCause, Environment env) {
    return Event.error(
        value.getLocation(),
        String.format(
            "%s, aspect %s: missing input file '%s'",
            value.getLabel(), value.getConfiguredAspect().getName(), rootCause));
  }

  @Override
  public MissingInputFileException getMissingFilesException(
      AspectValue value, int missingCount, Environment env) {
    return new MissingInputFileException(
        value.getLabel()
            + ", aspect "
            + value.getConfiguredAspect().getName()
            + missingCount
            + " input file(s) do not exist",
        value.getLocation());
  }

  @Override
  public AspectCompletionValue getResult() {
    return AspectCompletionValue.INSTANCE;
  }

  @Override
  public ExtendedEventHandler.Postable createFailed(
      AspectValue value,
      NestedSet<Cause> rootCauses,
      NestedSet<ArtifactsInOutputGroup> outputs,
      Environment env,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    BuildEventId configurationEventId = getConfigurationEventIdFromAspectValue(value, env);
    if (configurationEventId == null) {
      return null;
    }
    return AspectCompleteEvent.createFailed(value, rootCauses, configurationEventId, outputs);
  }

  @Nullable
  private BuildEventId getConfigurationEventIdFromAspectValue(AspectValue value, Environment env)
      throws InterruptedException {
    if (value.getKey().getBaseConfiguredTargetKey().getConfigurationKey() == null) {
      return BuildEventIdUtil.nullConfigurationId();
    } else {
      BuildConfigurationValue buildConfigurationValue =
          (BuildConfigurationValue)
              env.getValue(value.getKey().getBaseConfiguredTargetKey().getConfigurationKey());
      if (buildConfigurationValue == null) {
        return null;
      }
      return buildConfigurationValue.getConfiguration().getEventId();
    }
  }

  @Override
  public ExtendedEventHandler.Postable createSucceeded(
      SkyKey skyKey,
      AspectValue value,
      CompletionContext completionContext,
      ArtifactsToBuild artifactsToBuild,
      Environment env)
      throws InterruptedException {
    BuildEventId configurationEventId = getConfigurationEventIdFromAspectValue(value, env);
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
