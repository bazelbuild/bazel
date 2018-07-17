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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactSkyKey;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;

/**
 * TestCompletionFunction builds all relevant test artifacts of a {@link
 * com.google.devtools.build.lib.analysis.ConfiguredTarget}. This includes test shards and repeated
 * runs.
 */
public final class TestCompletionFunction implements SkyFunction {
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    TestCompletionValue.TestCompletionKey key =
        (TestCompletionValue.TestCompletionKey) skyKey.argument();
    ConfiguredTargetKey ctKey = key.configuredTargetKey();
    TopLevelArtifactContext ctx = key.topLevelArtifactContext();
    if (env.getValue(TargetCompletionValue.key(ctKey, ctx, /*willTest=*/ true)) == null) {
      return null;
    }

    ConfiguredTargetValue ctValue = (ConfiguredTargetValue) env.getValue(ctKey);
    if (ctValue == null) {
      return null;
    }

    ConfiguredTarget ct = ctValue.getConfiguredTarget();
    if (key.exclusiveTesting()) {
      // Request test execution iteratively if testing exclusively.
      for (Artifact testArtifact : TestProvider.getTestStatusArtifacts(ct)) {
        ActionLookupValue.ActionLookupKey actionLookupKey =
            ArtifactFunction.getActionLookupKey(testArtifact);
        ActionLookupValue actionLookupValue =
            ArtifactFunction.getActionLookupValue(actionLookupKey, env, testArtifact);
        if (actionLookupValue == null) {
          return null;
        }
        env.getValue(getActionLookupData(testArtifact, actionLookupKey, actionLookupValue));
        if (env.valuesMissing()) {
          return null;
        }
      }
    } else {
      Multimap<ActionLookupValue.ActionLookupKey, ArtifactSkyKey> keyToArtifactMap =
          Multimaps.index(
              ArtifactSkyKey.mandatoryKeys(TestProvider.getTestStatusArtifacts(ct)),
              (val) -> ArtifactFunction.getActionLookupKey(val.getArtifact()));
      Map<SkyKey, SkyValue> actionLookupValues = env.getValues(keyToArtifactMap.keySet());
      if (env.valuesMissing()) {
        return null;
      }
      env.getValues(
          keyToArtifactMap
              .entries()
              .stream()
              .map(
                  entry ->
                      getActionLookupData(
                          entry.getValue().getArtifact(),
                          entry.getKey(),
                          (ActionLookupValue) actionLookupValues.get(entry.getKey())))
              .distinct()
              .collect(ImmutableSet.toImmutableSet()));
      if (env.valuesMissing()) {
        return null;
      }
    }
    return TestCompletionValue.TEST_COMPLETION_MARKER;
  }

  private static ActionLookupData getActionLookupData(
      Artifact artifact,
      ActionLookupValue.ActionLookupKey actionLookupKey,
      ActionLookupValue actionLookupValue) {
    return ActionExecutionValue.key(
        actionLookupKey, actionLookupValue.getGeneratingActionIndex(artifact));
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((ConfiguredTargetKey) skyKey.argument()).getLabel());
  }
}
