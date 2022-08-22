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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.GraphTraversingHelper;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * TestCompletionFunction builds all relevant test artifacts of a {@link
 * com.google.devtools.build.lib.analysis.ConfiguredTarget}. This includes test shards and repeated
 * runs.
 */
public final class TestCompletionFunction implements SkyFunction {
  @Override
  @Nullable
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
      for (Artifact.DerivedArtifact testArtifact : TestProvider.getTestStatusArtifacts(ct)) {
        env.getValue(testArtifact.getGeneratingActionKey());
        if (env.valuesMissing()) {
          return null;
        }
      }
    } else {
      if (GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions(
          env,
          Iterables.transform(
              TestProvider.getTestStatusArtifacts(ct),
              Artifact.DerivedArtifact::getGeneratingActionKey))) {
        return null;
      }
    }
    return TestCompletionValue.TEST_COMPLETION_MARKER;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((ConfiguredTargetKey) skyKey.argument()).getLabel());
  }
}
