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
import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.Set;

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
      ConfiguredTargetKey configuredTargetKey,
      TopLevelArtifactContext topLevelArtifactContext,
      boolean willTest) {
    return TargetCompletionKey.create(configuredTargetKey, topLevelArtifactContext, willTest);
  }

  public static Iterable<SkyKey> keys(
      Collection<ConfiguredTarget> targets,
      final TopLevelArtifactContext ctx,
      final Set<ConfiguredTarget> targetsToTest) {
    return Iterables.transform(
        targets,
        new Function<ConfiguredTarget, SkyKey>() {
          @Override
          public SkyKey apply(ConfiguredTarget ct) {
            return TargetCompletionKey.create(
                ConfiguredTargetKey.of(ct), ctx, targetsToTest.contains(ct));
          }
        });
  }

  @AutoValue
  abstract static class TargetCompletionKey implements SkyKey {
    public static TargetCompletionKey create(
        ConfiguredTargetKey configuredTargetKey,
        TopLevelArtifactContext topLevelArtifactContext,
        boolean willTest) {
      return new AutoValue_TargetCompletionValue_TargetCompletionKey(
          configuredTargetKey, topLevelArtifactContext, willTest);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_COMPLETION;
    }

    abstract ConfiguredTargetKey configuredTargetKey();

    public abstract TopLevelArtifactContext topLevelArtifactContext();
    public abstract boolean willTest();
  }
}
