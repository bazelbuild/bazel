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
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.StallableSkykey;
import java.util.Collection;
import java.util.Set;

/** The value of a TargetCompletion. Just a sentinel. */
public class TargetCompletionValue implements SkyValue {
  @SerializationConstant static final TargetCompletionValue INSTANCE = new TargetCompletionValue();

  private TargetCompletionValue() {}

  public static TargetCompletionKey key(
      ConfiguredTargetKey configuredTargetKey,
      TopLevelArtifactContext topLevelArtifactContext,
      boolean willTest) {
    return TargetCompletionKey.create(configuredTargetKey, topLevelArtifactContext, willTest);
  }

  public static Iterable<TargetCompletionKey> keys(
      Collection<ConfiguredTarget> targets,
      final TopLevelArtifactContext ctx,
      final Set<ConfiguredTarget> targetsToTest) {
    return Iterables.transform(
        targets,
        ct ->
            TargetCompletionKey.create(
                ConfiguredTargetKey.fromConfiguredTarget(ct), ctx, targetsToTest.contains(ct)));
  }

  /** {@link com.google.devtools.build.skyframe.SkyKey} for {@link TargetCompletionValue}. */
  @AutoValue
  public abstract static class TargetCompletionKey
      implements TopLevelActionLookupKeyWrapper, StallableSkykey {
    static TargetCompletionKey create(
        ConfiguredTargetKey actionLookupKey,
        TopLevelArtifactContext topLevelArtifactContext,
        boolean willTest) {
      return new AutoValue_TargetCompletionValue_TargetCompletionKey(
          topLevelArtifactContext, actionLookupKey, willTest);
    }

    @Override
    public abstract ConfiguredTargetKey actionLookupKey();

    @Override
    public final SkyFunctionName functionName() {
      return SkyFunctions.TARGET_COMPLETION;
    }

    @Override
    public final boolean valueIsShareable() {
      return false;
    }

    abstract boolean willTest();
  }
}
