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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
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
}
