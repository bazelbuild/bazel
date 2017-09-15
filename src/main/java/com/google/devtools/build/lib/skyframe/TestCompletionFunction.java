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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

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
    LabelAndConfiguration lac = key.labelAndConfiguration();
    TopLevelArtifactContext ctx = key.topLevelArtifactContext();
    if (env.getValue(TargetCompletionValue.key(lac, ctx)) == null) {
      return null;
    }

    ConfiguredTargetValue ctValue = (ConfiguredTargetValue)
        env.getValue(ConfiguredTargetValue.key(lac.getLabel(), lac.getConfiguration()));
    if (ctValue == null) {
      return null;
    }

    ConfiguredTarget ct = ctValue.getConfiguredTarget();
    if (key.exclusiveTesting()) {
      // Request test artifacts iteratively if testing exclusively.
      for (Artifact testArtifact : TestProvider.getTestStatusArtifacts(ct)) {
        if (env.getValue(ArtifactSkyKey.key(testArtifact, /*isMandatory=*/ true)) == null) {
          return null;
        }
      }
    } else {
      env.getValues(ArtifactSkyKey.mandatoryKeys(TestProvider.getTestStatusArtifacts(ct)));
      if (env.valuesMissing()) {
        return null;
      }
    }
    return TestCompletionValue.TEST_COMPLETION_MARKER;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((LabelAndConfiguration) skyKey.argument()).getLabel());
  }
}
