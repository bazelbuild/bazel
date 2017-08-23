// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * The value of a TargetCompletion. Currently this just stores a ConfiguredTarget.
 */
public class TargetCompletionValue implements SkyValue {
  private final ConfiguredTarget ct;

  TargetCompletionValue(ConfiguredTarget ct) {
    this.ct = ct;
  }

  public ConfiguredTarget getConfiguredTarget() {
    return ct;
  }

  public static SkyKey key(
      LabelAndConfiguration labelAndConfiguration,
      TopLevelArtifactContext topLevelArtifactContext,
      SkyKey testExecutionSkyKey) {
    return TargetCompletionKey.create(
        labelAndConfiguration, topLevelArtifactContext, testExecutionSkyKey);
  }

  public static Iterable<SkyKey> keys(Collection<ConfiguredTarget> targets,
      final TopLevelArtifactContext ctx) {
    return Iterables.transform(
        targets, ct -> TargetCompletionKey.create(LabelAndConfiguration.of(ct), ctx, null));
  }

  @AutoValue
  abstract static class TargetCompletionKey implements SkyKey {
    public static TargetCompletionKey create(
        LabelAndConfiguration labelAndConfiguration,
        TopLevelArtifactContext topLevelArtifactContext,
        @Nullable SkyKey testExecutionSkyKey) {
      return new AutoValue_TargetCompletionValue_TargetCompletionKey(
          labelAndConfiguration, topLevelArtifactContext, testExecutionSkyKey);
    }

    abstract LabelAndConfiguration labelAndConfiguration();

    public abstract TopLevelArtifactContext topLevelArtifactContext();

    @Nullable
    abstract SkyKey testExecutionSkyKey();

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_COMPLETION;
    }
  }
}
