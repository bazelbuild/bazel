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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class ConfiguredTargetKeyTest extends BuildViewTestCase {
  private static final AtomicInteger nextId = new AtomicInteger();

  @Test
  public void testDelegation(
      @TestParameter boolean useNullConfig, @TestParameter boolean isToolchainKey) {
    var baseKey = createKey(useNullConfig, isToolchainKey);

    assertThat(baseKey.isProxy()).isFalse();
    assertThat(baseKey.toKey()).isSameInstanceAs(baseKey);

    BuildConfigurationKey newConfigurationKey = getNewUniqueConfigurationKey();
    var delegatingKey =
        ConfiguredTargetKey.builder()
            .setDelegate(baseKey)
            .setConfigurationKey(newConfigurationKey)
            .build();
    assertThat(delegatingKey.isProxy()).isTrue();
    assertThat(delegatingKey.toKey()).isSameInstanceAs(baseKey);
    assertThat(delegatingKey.getLabel()).isSameInstanceAs(baseKey.getLabel());
    assertThat(delegatingKey.getConfigurationKey()).isSameInstanceAs(newConfigurationKey);
    assertThat(delegatingKey.getExecutionPlatformLabel())
        .isSameInstanceAs(baseKey.getExecutionPlatformLabel());

    // Building a key with the same parameters as the delegating key returns the delegating key.
    var similarKey =
        ConfiguredTargetKey.builder()
            .setLabel(delegatingKey.getLabel())
            .setConfigurationKey(delegatingKey.getConfigurationKey())
            .setExecutionPlatformLabel(delegatingKey.getExecutionPlatformLabel())
            .build();
    assertThat(similarKey).isSameInstanceAs(delegatingKey);
  }

  @Test
  public void existingKey_inhibitsDelegation(
      @TestParameter boolean useNullConfig, @TestParameter boolean isToolchainKey) {
    var baseKey = createKey(useNullConfig, isToolchainKey);

    BuildConfigurationKey newConfigurationKey = getNewUniqueConfigurationKey();

    var existingKey =
        ConfiguredTargetKey.builder()
            .setLabel(baseKey.getLabel())
            .setConfigurationKey(newConfigurationKey)
            .setExecutionPlatformLabel(baseKey.getExecutionPlatformLabel())
            .build();

    var delegatingKey =
        ConfiguredTargetKey.builder()
            .setDelegate(baseKey)
            .setConfigurationKey(newConfigurationKey)
            .build();

    assertThat(delegatingKey).isSameInstanceAs(existingKey);
  }

  @Test
  public void testCodec() throws Exception {
    var nullConfigKey = createKey(/* useNullConfig= */ true, /* isToolchainKey= */ false);
    var keyWithConfig = createKey(/* useNullConfig= */ false, /* isToolchainKey= */ false);
    var toolchainKey = createKey(/* useNullConfig= */ false, /* isToolchainKey= */ true);

    var delegatingToNullConfig =
        ConfiguredTargetKey.builder()
            .setDelegate(nullConfigKey)
            .setConfigurationKey(targetConfigKey)
            .build();
    var delegatingToKeyWithConfig =
        ConfiguredTargetKey.builder().setDelegate(keyWithConfig).build();
    var delegatingToToolchainKey =
        ConfiguredTargetKey.builder()
            .setDelegate(toolchainKey)
            .setConfigurationKey(getNewUniqueConfigurationKey())
            .build();

    new SerializationTester(
            nullConfigKey,
            keyWithConfig,
            toolchainKey,
            delegatingToNullConfig,
            delegatingToKeyWithConfig,
            delegatingToToolchainKey)
        .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
        .runTests();
  }

  private ConfiguredTargetKey createKey(boolean useNullConfig, boolean isToolchainKey) {
    var key = ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked("//p:key"));
    if (!useNullConfig) {
      key.setConfigurationKey(targetConfigKey);
    }
    if (isToolchainKey) {
      key.setExecutionPlatformLabel(Label.parseCanonicalUnchecked("//platforms:b"));
    }
    return key.build();
  }

  private BuildConfigurationKey getNewUniqueConfigurationKey() {
    BuildOptions newOptions = targetConfigKey.getOptions().clone();
    var coreOptions = newOptions.get(CoreOptions.class);
    coreOptions.affectedByStarlarkTransition =
        ImmutableList.of("//fake:id" + nextId.getAndIncrement());
    assertThat(newOptions.checksum()).isNotEqualTo(targetConfigKey.getOptions().checksum());
    return BuildConfigurationKey.withoutPlatformMapping(newOptions);
  }
}
