// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.OptionsDiff;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.compress.CompressionServiceImpl;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.PlatformConfigurationProvider;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildConfigurationKeyBaselineDiffCodec}. */
@RunWith(JUnit4.class)
public final class BuildConfigurationKeyBaselineDiffCodecTest {

  private static final CompressionService COMPRESSION_SERVICE = new CompressionServiceImpl();

  private static PlatformConfigurationProvider createPlatformConfigurationProvider(
      BuildOptions baselineOptions) {
    return createPlatformConfigurationProvider(baselineOptions, /* topLevelPlatformLabel= */ null);
  }

  private static PlatformConfigurationProvider createPlatformConfigurationProvider(
      BuildOptions baselineOptions, @Nullable Label topLevelPlatformLabel) {
    return new PlatformConfigurationProvider() {
      @Override
      public Label getTopLevelPlatformLabel() {
        return topLevelPlatformLabel;
      }

      @Override
      public BuildOptions getBaseOptionsForPlatform(
          Label platformLabel, boolean isExec, boolean trimTestOptions) {
        return baselineOptions;
      }

      @Override
      public String resolveMnemonic(BuildOptions targetOptions) {
        return "dummy-mnemonic";
      }
    };
  }

  private static ImmutableClassToInstanceMap<Object> createDependencies(
      BuildOptions baselineOptions) {
    return createDependencies(createPlatformConfigurationProvider(baselineOptions));
  }

  private static ImmutableClassToInstanceMap<Object> createDependencies(
      PlatformConfigurationProvider provider) {
    return ImmutableClassToInstanceMap.of(PlatformConfigurationProvider.class, provider);
  }

  private static Object deserialize(ObjectCodecs codecs, ByteString serialized) throws Exception {
    return codecs.deserializeWithSkyframe(
        COMPRESSION_SERVICE, FingerprintValueService.createForTesting(), serialized);
  }

  private static ObjectCodecRegistry createRegistry() {
    return AutoRegistry.get()
        .getBuilder()
        .add(new OptionsDiff.OptionsDiffCodec())
        .add(new BuildConfigurationKeyBaselineDiffCodec())
        .build();
  }

  @Test
  public void testCodec_matchingBaseline_serializesAndReconstructs() throws Exception {
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=opt");
    BuildConfigurationKey key = BuildConfigurationKey.create(options);

    ImmutableClassToInstanceMap<Object> dependencies = createDependencies(options);
    ObjectCodecs codecs = new ObjectCodecs(createRegistry(), dependencies);

    ByteString serialized = codecs.serializeMemoized(key);
    BuildConfigurationKey deserialized = (BuildConfigurationKey) deserialize(codecs, serialized);
    assertThat(deserialized).isEqualTo(key);
  }

  @Test
  public void testCodec_differentBaseline_serializesDiffAndReconstructs() throws Exception {
    BuildOptions baselineOptions =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=opt");
    BuildOptions targetOptions =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=dbg");
    BuildConfigurationKey targetKey = BuildConfigurationKey.create(targetOptions);

    ImmutableClassToInstanceMap<Object> dependencies = createDependencies(baselineOptions);
    ObjectCodecs codecs = new ObjectCodecs(createRegistry(), dependencies);

    ByteString serialized = codecs.serializeMemoized(targetKey);
    BuildConfigurationKey deserialized = (BuildConfigurationKey) deserialize(codecs, serialized);
    assertThat(deserialized).isEqualTo(targetKey);
  }

  @Test
  public void testFingerprint_isIdentical_whenTopLevelPlatformDiffers() throws Exception {
    // Demonstrates that even across two different invocations with different top-level platform
    // labels and different baseline command-line options, targets that remain on their respective
    // top-level platform and undergo the same option transition serialize to identical bytes.
    Label topLevelPlatformOne = Label.parseCanonicalUnchecked("//platform:one");
    BuildOptions baselineOne =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:one",
            "--compilation_mode=opt");
    BuildOptions targetOne =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:one",
            "--compilation_mode=fastbuild");
    BuildConfigurationKey targetKeyOne = BuildConfigurationKey.create(targetOne);

    Label topLevelPlatformTwo = Label.parseCanonicalUnchecked("//platform:two");
    BuildOptions baselineTwo =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:two",
            "--compilation_mode=dbg");
    BuildOptions targetTwo =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--platforms=//platform:two",
            "--compilation_mode=fastbuild");
    BuildConfigurationKey targetKeyTwo = BuildConfigurationKey.create(targetTwo);

    PlatformConfigurationProvider providerOne =
        createPlatformConfigurationProvider(baselineOne, topLevelPlatformOne);
    ImmutableClassToInstanceMap<Object> depsOne = createDependencies(providerOne);
    ObjectCodecs codecsOne = new ObjectCodecs(createRegistry(), depsOne);
    ByteString serializedOne = codecsOne.serializeMemoized(targetKeyOne);

    PlatformConfigurationProvider providerTwo =
        createPlatformConfigurationProvider(baselineTwo, topLevelPlatformTwo);
    ImmutableClassToInstanceMap<Object> depsTwo = createDependencies(providerTwo);
    ObjectCodecs codecsTwo = new ObjectCodecs(createRegistry(), depsTwo);
    ByteString serializedTwo = codecsTwo.serializeMemoized(targetKeyTwo);

    // Verify cross-invocation fingerprint invariance: eliding the top-level platform label plus the
    // OptionsDiff RESET ensure both invocations serialize the transitioned configuration key to the
    // exact same bytes.
    assertThat(serializedOne).isEqualTo(serializedTwo);

    BuildConfigurationKey deserializedOne =
        (BuildConfigurationKey) deserialize(codecsOne, serializedOne);
    assertThat(deserializedOne).isEqualTo(targetKeyOne);

    BuildConfigurationKey deserializedTwo =
        (BuildConfigurationKey) deserialize(codecsTwo, serializedTwo);
    assertThat(deserializedTwo).isEqualTo(targetKeyTwo);
  }

  private static BuildOptions toExecOptions(BuildOptions options) throws Exception {
    BuildOptions execOptions = options.clone();
    CoreOptions coreOptions = execOptions.get(CoreOptions.class);
    // CoreOptions only declares the getter; the setter is generated onto the concrete subclass,
    // which is not visible as an import, so it has to be reached reflectively.
    coreOptions.getClass().getMethod("setIsExec", boolean.class).invoke(coreOptions, true);
    return execOptions;
  }

  @Test
  public void testCodec_isExec_serializesConcretePlatformLabel() throws Exception {
    Label platformLabel = Label.parseCanonicalUnchecked("//test:platform");
    BuildOptions baselineOptions =
        toExecOptions(
            BuildOptions.of(
                ImmutableList.of(CoreOptions.class, PlatformOptions.class),
                "--platforms=//test:platform",
                "--compilation_mode=opt"));
    BuildOptions targetOptions =
        toExecOptions(
            BuildOptions.of(
                ImmutableList.of(CoreOptions.class, PlatformOptions.class),
                "--platforms=//test:platform",
                "--compilation_mode=dbg"));
    BuildConfigurationKey targetKey = BuildConfigurationKey.create(targetOptions);

    PlatformConfigurationProvider provider =
        createPlatformConfigurationProvider(baselineOptions, platformLabel);
    ImmutableClassToInstanceMap<Object> dependencies = createDependencies(provider);
    ObjectCodecs codecs = new ObjectCodecs(createRegistry(), dependencies);

    ByteString serialized = codecs.serializeMemoized(targetKey);
    BuildConfigurationKey deserialized = (BuildConfigurationKey) deserialize(codecs, serialized);
    assertThat(deserialized).isEqualTo(targetKey);

    // Verify concrete platform label was serialized by deserializing with a provider whose
    // top-level platform label differs from platformLabel. If the isTopLevelPlatform flag had been
    // set, deserialization would substitute the provider's top-level platform instead.
    PlatformConfigurationProvider differentTopLevelProvider =
        createPlatformConfigurationProvider(
            baselineOptions, Label.parseCanonicalUnchecked("//other:platform"));
    ObjectCodecs differentCodecs =
        new ObjectCodecs(createRegistry(), createDependencies(differentTopLevelProvider));
    BuildConfigurationKey deserializedWithDifferentTopLevel =
        (BuildConfigurationKey) deserialize(differentCodecs, serialized);
    assertThat(deserializedWithDifferentTopLevel).isEqualTo(targetKey);
  }

  @Test
  public void testCodec_emptyOptions_serializesDirectly() throws Exception {
    BuildOptions emptyOptions = CommonOptions.EMPTY_OPTIONS;
    BuildConfigurationKey key = BuildConfigurationKey.create(emptyOptions);

    // EMPTY_OPTIONS carries CoreOptions but no PlatformOptions, so it cannot be diff encoded and is
    // written out directly. Supplying no dependencies at all is what proves that: the diff encoding
    // path requires a PlatformConfigurationProvider dependency and would fail without one.
    ObjectCodecs codecs = new ObjectCodecs(createRegistry(), ImmutableClassToInstanceMap.of());

    ByteString serialized = codecs.serializeMemoized(key);
    BuildConfigurationKey deserialized = (BuildConfigurationKey) deserialize(codecs, serialized);
    assertThat(deserialized).isEqualTo(key);
  }

  @Test
  public void testFingerprint_isIdentical_irrespectiveOfCommandLineOptions() throws Exception {
    // Build 1 has first baselineOptions ("opt")
    BuildOptions baselineOne =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=opt");
    // Build 2 has second baselineOptions ("dbg")
    BuildOptions baselineTwo =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=dbg");

    // The target configuration to serialize is identical ("fastbuild")
    BuildOptions targetOptions =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--compilation_mode=fastbuild");
    BuildConfigurationKey targetKey = BuildConfigurationKey.create(targetOptions);

    ImmutableClassToInstanceMap<Object> depsOne = createDependencies(baselineOne);
    ObjectCodecs codecsOne = new ObjectCodecs(createRegistry(), depsOne);
    ByteString serializedOne = codecsOne.serializeMemoized(targetKey);

    ImmutableClassToInstanceMap<Object> depsTwo = createDependencies(baselineTwo);
    ObjectCodecs codecsTwo = new ObjectCodecs(createRegistry(), depsTwo);
    ByteString serializedTwo = codecsTwo.serializeMemoized(targetKey);

    // Verify that both serialized byte streams are EXACTLY identical, since OptionsDiffCodec
    // records the original value as "RESET" and the destination value as "fastbuild".
    assertThat(serializedOne).isEqualTo(serializedTwo);

    BuildConfigurationKey deserializedOne =
        (BuildConfigurationKey) deserialize(codecsOne, serializedOne);
    assertThat(deserializedOne).isEqualTo(targetKey);

    BuildConfigurationKey deserializedTwo =
        (BuildConfigurationKey) deserialize(codecsTwo, serializedTwo);
    assertThat(deserializedTwo).isEqualTo(targetKey);
  }

  @Test
  public void testFingerprint_isIdentical_forListResets() throws Exception {
    // Build 1 Baseline: features = [a]
    BuildOptions baselineOne =
        BuildOptions.of(ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--features=a");

    // Build 2 Baseline: features = [b]
    BuildOptions baselineTwo =
        BuildOptions.of(ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--features=b");

    // Target: Reset features list completely to [c] (disjoint relative to baselineOne /
    // baselineTwo)
    BuildOptions targetOne =
        BuildOptions.of(ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--features=c");
    BuildConfigurationKey targetKeyOne = BuildConfigurationKey.create(targetOne);

    BuildOptions targetTwo =
        BuildOptions.of(ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--features=c");
    BuildConfigurationKey targetKeyTwo = BuildConfigurationKey.create(targetTwo);

    ImmutableClassToInstanceMap<Object> depsOne = createDependencies(baselineOne);
    ObjectCodecs codecsOne = new ObjectCodecs(createRegistry(), depsOne);
    ByteString serializedOne = codecsOne.serializeMemoized(targetKeyOne);

    ImmutableClassToInstanceMap<Object> depsTwo = createDependencies(baselineTwo);
    ObjectCodecs codecsTwo = new ObjectCodecs(createRegistry(), depsTwo);
    ByteString serializedTwo = codecsTwo.serializeMemoized(targetKeyTwo);

    // Verify that both serialized diff streams are EXACTLY identical (Reset platform list to C).
    assertThat(serializedOne).isEqualTo(serializedTwo);

    BuildConfigurationKey deserializedOne =
        (BuildConfigurationKey) deserialize(codecsOne, serializedOne);
    assertThat(deserializedOne).isEqualTo(targetKeyOne);

    BuildConfigurationKey deserializedTwo =
        (BuildConfigurationKey) deserialize(codecsTwo, serializedTwo);
    assertThat(deserializedTwo).isEqualTo(targetKeyTwo);
  }

  @Test
  public void testFingerprint_isIdentical_forListAppends() throws Exception {
    // Build 1 Baseline: features = [O2]
    BuildOptions baselineOne =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--features=O2");

    // Build 2 Baseline: features = [O3]
    BuildOptions baselineTwo =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--features=O3");

    // Target 1: Keep O2 baseline feature, and append Wall.
    BuildOptions targetOne =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--features=O2",
            "--features=Wall");
    BuildConfigurationKey targetKeyOne = BuildConfigurationKey.create(targetOne);

    // Target 2: Keep O3 baseline feature, and append Wall.
    BuildOptions targetTwo =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--features=O3",
            "--features=Wall");
    BuildConfigurationKey targetKeyTwo = BuildConfigurationKey.create(targetTwo);

    ImmutableClassToInstanceMap<Object> depsOne = createDependencies(baselineOne);
    ObjectCodecs codecsOne = new ObjectCodecs(createRegistry(), depsOne);
    ByteString serializedOne = codecsOne.serializeMemoized(targetKeyOne);

    ImmutableClassToInstanceMap<Object> depsTwo = createDependencies(baselineTwo);
    ObjectCodecs codecsTwo = new ObjectCodecs(createRegistry(), depsTwo);
    ByteString serializedTwo = codecsTwo.serializeMemoized(targetKeyTwo);

    // Verify that both serialized diff streams are EXACTLY identical (+Wall append).
    assertThat(serializedOne).isEqualTo(serializedTwo);

    BuildConfigurationKey deserializedOne =
        (BuildConfigurationKey) deserialize(codecsOne, serializedOne);
    assertThat(deserializedOne).isEqualTo(targetKeyOne);

    BuildConfigurationKey deserializedTwo =
        (BuildConfigurationKey) deserialize(codecsTwo, serializedTwo);
    assertThat(deserializedTwo).isEqualTo(targetKeyTwo);
  }

  @Test
  public void testCodec_withTrimTestOptions_serializesAndReconstructsWithTrimmedBaseline()
      throws Exception {
    // The two baselines differ in a feature that the target shares with the trimmed one, so that
    // feature is absent from the diff. Reconstructing against the untrimmed baseline would
    // therefore drop it, which means the round trip only succeeds if the trimTestOptions flag
    // travelled on the wire and selected the trimmed baseline on the reading side.
    BuildOptions trimmedBaseline =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--compilation_mode=opt",
            "--features=a");
    BuildOptions untrimmedBaseline =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=opt");
    BuildOptions targetOptions =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class),
            "--compilation_mode=dbg",
            "--features=a");
    BuildConfigurationKey targetKey = BuildConfigurationKey.create(targetOptions);

    List<Boolean> observedTrimTestOptions = new ArrayList<>();
    PlatformConfigurationProvider provider =
        new PlatformConfigurationProvider() {
          @Override
          public Label getTopLevelPlatformLabel() {
            return null;
          }

          @Override
          public BuildOptions getBaseOptionsForPlatform(
              Label platformLabel, boolean isExec, boolean trimTestOptions) {
            observedTrimTestOptions.add(trimTestOptions);
            return trimTestOptions ? trimmedBaseline : untrimmedBaseline;
          }

          @Override
          public String resolveMnemonic(BuildOptions options) {
            return "dummy-mnemonic";
          }
        };
    ObjectCodecs codecs = new ObjectCodecs(createRegistry(), createDependencies(provider));

    ByteString serialized = codecs.serializeMemoized(targetKey);
    BuildConfigurationKey deserialized = (BuildConfigurationKey) deserialize(codecs, serialized);

    assertThat(deserialized).isEqualTo(targetKey);
    // Once while writing and once while reading, both with the flag set.
    assertThat(observedTrimTestOptions).containsExactly(true, true);
  }

  @Test
  public void testDeserialization_whenProviderThrowsSerializationException_propagatesException()
      throws Exception {
    BuildOptions baselineOptions =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=opt");
    BuildOptions targetOptions =
        BuildOptions.of(
            ImmutableList.of(CoreOptions.class, PlatformOptions.class), "--compilation_mode=dbg");
    BuildConfigurationKey targetKey = BuildConfigurationKey.create(targetOptions);

    ObjectCodecs writingCodecs =
        new ObjectCodecs(createRegistry(), createDependencies(baselineOptions));
    ByteString serialized = writingCodecs.serializeMemoized(targetKey);

    // Baseline resolution is allowed to fail on the reading side, and that failure has to surface
    // as a SerializationException rather than leaving the deserialization future unresolved.
    PlatformConfigurationProvider throwingProvider =
        new PlatformConfigurationProvider() {
          @Override
          public Label getTopLevelPlatformLabel() {
            return null;
          }

          @Override
          public BuildOptions getBaseOptionsForPlatform(
              Label platformLabel, boolean isExec, boolean trimTestOptions)
              throws SerializationException {
            throw new SerializationException("no baseline available");
          }

          @Override
          public String resolveMnemonic(BuildOptions options) {
            return "dummy-mnemonic";
          }
        };
    ObjectCodecs readingCodecs =
        new ObjectCodecs(createRegistry(), createDependencies(throwingProvider));

    SerializationException thrown =
        assertThrows(SerializationException.class, () -> deserialize(readingCodecs, serialized));
    assertThat(Throwables.getStackTraceAsString(thrown)).contains("no baseline available");
  }

  @Test
  public void testFingerprint_isIdentical_forExecConfigurationsAcrossDifferentBaselines()
      throws Exception {
    // Both invocations use the same exec platform, because exec configurations never elide their
    // platform label. What differs is the exec baseline, and both targets append the same feature
    // relative to their own baseline, so the diffs match and the bytes must too.
    BuildOptions execBaselineOne =
        toExecOptions(
            BuildOptions.of(
                ImmutableList.of(CoreOptions.class, PlatformOptions.class),
                "--platforms=//test:exec_platform",
                "--features=O2"));
    BuildOptions execTargetOne =
        toExecOptions(
            BuildOptions.of(
                ImmutableList.of(CoreOptions.class, PlatformOptions.class),
                "--platforms=//test:exec_platform",
                "--features=O2",
                "--features=Wall"));
    BuildConfigurationKey execTargetKeyOne = BuildConfigurationKey.create(execTargetOne);

    BuildOptions execBaselineTwo =
        toExecOptions(
            BuildOptions.of(
                ImmutableList.of(CoreOptions.class, PlatformOptions.class),
                "--platforms=//test:exec_platform",
                "--features=O3"));
    BuildOptions execTargetTwo =
        toExecOptions(
            BuildOptions.of(
                ImmutableList.of(CoreOptions.class, PlatformOptions.class),
                "--platforms=//test:exec_platform",
                "--features=O3",
                "--features=Wall"));
    BuildConfigurationKey execTargetKeyTwo = BuildConfigurationKey.create(execTargetTwo);

    ObjectCodecs codecsOne =
        new ObjectCodecs(createRegistry(), createDependencies(execBaselineOne));
    ByteString serializedOne = codecsOne.serializeMemoized(execTargetKeyOne);

    ObjectCodecs codecsTwo =
        new ObjectCodecs(createRegistry(), createDependencies(execBaselineTwo));
    ByteString serializedTwo = codecsTwo.serializeMemoized(execTargetKeyTwo);

    assertThat(serializedOne).isEqualTo(serializedTwo);

    BuildConfigurationKey deserializedOne =
        (BuildConfigurationKey) deserialize(codecsOne, serializedOne);
    assertThat(deserializedOne).isEqualTo(execTargetKeyOne);

    BuildConfigurationKey deserializedTwo =
        (BuildConfigurationKey) deserialize(codecsTwo, serializedTwo);
    assertThat(deserializedTwo).isEqualTo(execTargetKeyTwo);
  }
}
