// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class ConfiguredTargetKeyTest extends BuildViewTestCase {
  @Test
  public void testCodec() throws Exception {
    var nullConfigKey =
        createKey(
            /* useNullConfig= */ true,
            /* isToolchainKey= */ false,
            /* shouldApplyRuleTransition= */ true);
    var keyWithConfig =
        createKey(
            /* useNullConfig= */ false,
            /* isToolchainKey= */ false,
            /* shouldApplyRuleTransition= */ true);
    var keyWithFinalConfig =
        createKey(
            /* useNullConfig= */ false,
            /* isToolchainKey= */ false,
            /* shouldApplyRuleTransition= */ false);
    var toolchainKey =
        createKey(
            /* useNullConfig= */ false,
            /* isToolchainKey= */ true,
            /* shouldApplyRuleTransition= */ true);
    var toolchainKeyWithFinalConfig =
        createKey(
            /* useNullConfig= */ false,
            /* isToolchainKey= */ true,
            /* shouldApplyRuleTransition= */ false);

    new SerializationTester(
            nullConfigKey,
            keyWithConfig,
            keyWithFinalConfig,
            toolchainKey,
            toolchainKeyWithFinalConfig)
        .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
        .runTests();
  }

  private ConfiguredTargetKey createKey(
      boolean useNullConfig, boolean isToolchainKey, boolean shouldApplyRuleTransition) {
    var key = ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked("//p:key"));
    if (!useNullConfig) {
      key.setConfigurationKey(targetConfigKey);
    }
    if (isToolchainKey) {
      key.setExecutionPlatformLabel(Label.parseCanonicalUnchecked("//platforms:b"));
    }
    key.setShouldApplyRuleTransition(shouldApplyRuleTransition);
    return key.build();
  }
}
