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
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/**
 * A test completion value represents the completion of a test target. This includes the execution
 * of all test shards and repeated runs, if applicable.
 */
public class TestCompletionValue implements SkyValue {
  static final TestCompletionValue TEST_COMPLETION_MARKER = new TestCompletionValue();

  private TestCompletionValue() { }

  @Override
  public boolean dataIsShareable() {
    return false;
  }

  public static SkyKey key(
      ConfiguredTargetKey lac,
      final TopLevelArtifactContext topLevelArtifactContext,
      final boolean exclusiveTesting) {
    return TestCompletionKey.create(lac, topLevelArtifactContext, exclusiveTesting);
  }

  public static Iterable<SkyKey> keys(Collection<ConfiguredTarget> targets,
                                      final TopLevelArtifactContext topLevelArtifactContext,
                                      final boolean exclusiveTesting) {
    return Iterables.transform(
        targets,
        ct ->
            TestCompletionKey.create(
                ConfiguredTargetKey.builder()
                    .setConfiguredTarget(ct)
                    .setConfigurationKey(ct.getConfigurationKey())
                    .build(),
                topLevelArtifactContext,
                exclusiveTesting));
  }

  /** Key for {@link TestCompletionValue} nodes. */
  @AutoCodec
  @AutoValue
  public abstract static class TestCompletionKey implements SkyKey {
    private static final Interner<TestCompletionKey> interner = BlazeInterners.newWeakInterner();

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static TestCompletionKey create(
        ConfiguredTargetKey configuredTargetKey,
        TopLevelArtifactContext topLevelArtifactContext,
        boolean exclusiveTesting) {
      return interner.intern(
          new AutoValue_TestCompletionValue_TestCompletionKey(
              configuredTargetKey, topLevelArtifactContext, exclusiveTesting));
    }

    abstract ConfiguredTargetKey configuredTargetKey();

    public abstract TopLevelArtifactContext topLevelArtifactContext();
    public abstract boolean exclusiveTesting();

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TEST_COMPLETION;
    }
  }
}
