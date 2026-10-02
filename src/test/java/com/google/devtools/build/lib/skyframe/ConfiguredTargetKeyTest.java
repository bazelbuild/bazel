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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.OptionsDiff;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PlatformConfigurationProvider;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.devtools.build.lib.skyframe.serialization.analysis.DefaultPlatformConfigurationProvider;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class ConfiguredTargetKeyTest extends BuildViewTestCase {

  private static final CompressionService COMPRESSION_SERVICE = new CompressionServiceImpl();

  @Test
  public void testCodec(@TestParameter boolean useSharedValues) throws Exception {
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

    var tester =
        new SerializationTester(
                nullConfigKey,
                keyWithConfig,
                keyWithFinalConfig,
                toolchainKey,
                toolchainKeyWithFinalConfig)
            .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
            .addDependency(PlatformConfigurationProvider.class, getPlatformConfigurationProvider());

    if (useSharedValues) {
      tester
          .addCodec(ConfiguredTargetKey.valueSharingCodec())
          .addCodec(new OptionsDiff.OptionsDiffCodec())
          .makeMemoizingAndAllowFutureBlocking(true);
    }

    tester.runTests();
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

  @Test
  public void testValueSharingCodec_identicalBytesAcrossDifferentBaselineConfigurations()
      throws Exception {
    // Two invocations with different top-level platforms and different baseline options. Each key
    // stays on its own top-level platform and undergoes the same option transition, so both
    // invocations must serialize the ConfiguredTargetKey to identical bytes.
    BuildOptions baselineOne = createOptions("//platform:one", "opt");
    BuildOptions targetOne = createOptions("//platform:one", "fastbuild");
    BuildOptions baselineTwo = createOptions("//platform:two", "dbg");
    BuildOptions targetTwo = createOptions("//platform:two", "fastbuild");

    ObjectCodecs codecsOne = createCodecs(createProvider("//platform:one", baselineOne));
    ObjectCodecs codecsTwo = createCodecs(createProvider("//platform:two", baselineTwo));

    // Each invocation gets its own fingerprint value service, as they would in separate builds.
    FingerprintValueService serviceOne = FingerprintValueService.createForTesting();
    FingerprintValueService serviceTwo = FingerprintValueService.createForTesting();
    ConfiguredTargetKey keyOne = createKeyWithConfiguration(targetOne);
    ConfiguredTargetKey keyTwo = createKeyWithConfiguration(targetTwo);
    ByteString serializedOne = serialize(codecsOne, keyOne, serviceOne);
    ByteString serializedTwo = serialize(codecsTwo, keyTwo, serviceTwo);

    // Eliding the top-level platform label plus the OptionsDiff RESET make both invocations
    // serialize the transitioned key to the exact same bytes, which lets them share Skycache
    // entries.
    assertThat(serializedOne).isEqualTo(serializedTwo);

    // Each invocation still reconstructs its own key from those identical bytes.
    assertThat(
            codecsOne.deserializeMemoizedAndBlocking(
                COMPRESSION_SERVICE, serviceOne, serializedOne))
        .isEqualTo(keyOne);
    assertThat(
            codecsTwo.deserializeMemoizedAndBlocking(
                COMPRESSION_SERVICE, serviceTwo, serializedTwo))
        .isEqualTo(keyTwo);
  }

  private static BuildOptions createOptions(String platformLabel, String compilationMode)
      throws Exception {
    return BuildOptions.of(
        ImmutableList.of(CoreOptions.class, PlatformOptions.class),
        "--platforms=" + platformLabel,
        "--compilation_mode=" + compilationMode);
  }

  private static PlatformConfigurationProvider createProvider(
      String topLevelPlatformLabel, BuildOptions baseline) {
    return new DefaultPlatformConfigurationProvider(
        Label.parseCanonicalUnchecked(topLevelPlatformLabel),
        /* targetBaseline= */ baseline,
        /* execBaseline= */ baseline);
  }

  private static ConfiguredTargetKey createKeyWithConfiguration(BuildOptions options) {
    return ConfiguredTargetKey.builder()
        .setLabel(Label.parseCanonicalUnchecked("//p:key"))
        .setConfigurationKey(BuildConfigurationKey.create(options))
        .build();
  }

  private static ObjectCodecs createCodecs(PlatformConfigurationProvider provider) {
    return new ObjectCodecs(
        AutoRegistry.get()
            .getBuilder()
            .add(ConfiguredTargetKey.valueSharingCodec())
            .add(new OptionsDiff.OptionsDiffCodec())
            .build(),
        ImmutableClassToInstanceMap.builder()
            .put(OptionsChecksumCache.class, new MapBackedChecksumCache())
            .put(PlatformConfigurationProvider.class, provider)
            .build());
  }

  private static ByteString serialize(
      ObjectCodecs codecs, ConfiguredTargetKey key, FingerprintValueService fingerprintValueService)
      throws Exception {
    SerializationResult<ByteString> result =
        codecs.serializeMemoizedAndBlocking(COMPRESSION_SERVICE, fingerprintValueService, key);
    ListenableFuture<?> futureToBlockWritesOn = result.getFutureToBlockWritesOn();
    if (futureToBlockWritesOn != null) {
      var _ = futureToBlockWritesOn.get();
    }
    return result.getObject();
  }
}
