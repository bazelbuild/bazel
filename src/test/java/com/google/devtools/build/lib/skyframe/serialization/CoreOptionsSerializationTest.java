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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FakeDirectories;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CoreOptionsSerializationTest {

  /**
   * Tests serialization (with {@link BuildOptions} dependency).
   *
   * <p>{@code checkVisibility} is not serialized, but restored from {@link BuildOptions} during
   * deserialization.
   */
  @Test
  public void coreOptionsRoundTrip() throws Exception {
    BuildOptions buildOptionsToSerialize = BuildOptions.of(ImmutableList.of(CoreOptions.class));
    CoreOptions optionsToSerialize = buildOptionsToSerialize.get(CoreOptions.class);
    optionsToSerialize.setCheckVisibility(false);

    BuildOptions buildOptions = BuildOptions.of(ImmutableList.of(CoreOptions.class));
    buildOptions.get(CoreOptions.class).setCheckVisibility(true); // This will be source of truth.

    SerializationTester tester = new SerializationTester(optionsToSerialize);
    for (ObjectCodec<?> codec : SerializationRegistrySetupHelpers.analysisCachingCodecs()) {
      tester.addCodec(codec);
    }
    tester
        .setVerificationFunction(
            (original, deserialized) -> {
              // Deserialized value comes from BuildOptions dependency, not original value.
              assertThat(((CoreOptions) deserialized).getCheckVisibility()).isTrue();
            })
        .addDependency(BuildOptions.class, buildOptions)
        .runTests();
  }

  @Test
  public void emptyOptionsRoundTrip_toSameInstance_withCustomCoreOptionsCodec() throws Exception {
    BuildOptions original = CommonOptions.EMPTY_OPTIONS;

    // Simulates the reader build passing --check_visibility=false.
    BuildOptions readerOptions = BuildOptions.of(ImmutableList.of(CoreOptions.class));
    readerOptions.get(CoreOptions.class).setCheckVisibility(false);

    ObjectCodecRegistry.Builder registryBuilder = AutoRegistry.get().getBuilder();
    for (ObjectCodec<?> codec : SerializationRegistrySetupHelpers.analysisCachingCodecs()) {
      registryBuilder.add(codec);
    }

    registryBuilder.addReferenceConstants(
        SerializationRegistrySetupHelpers.makeReferenceConstants(
            FakeDirectories.BLAZE_DIRECTORIES,
            new ConfiguredRuleClassProvider.Builder()
                .setToolsRepository(RepositoryName.createUnvalidated("bazel_tools"))
                .build(),
            "root"));
    ObjectCodecRegistry registry = registryBuilder.build();

    // Inject the reader options.
    ImmutableClassToInstanceMap<Object> dependencies =
        ImmutableClassToInstanceMap.of(BuildOptions.class, readerOptions);

    ObjectCodecs codecs = new ObjectCodecs(registry, dependencies);

    SerializationTester tester = new SerializationTester(original);
    tester.setObjectCodecs(codecs);

    tester
        .makeMemoizingAndAllowFutureBlocking(true)
        .setVerificationFunction(
            (orig, deserialized) -> {
              // Check that EMPTY_OPTIONS remain untainted by the custom CoreOptions
              // check_visibility trimming.
              assertThat(deserialized).isSameInstanceAs(orig);
            })
        .runTests();
  }
}
