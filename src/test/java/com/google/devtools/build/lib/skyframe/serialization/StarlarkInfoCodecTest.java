// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkInfoWithMessage;
import com.google.devtools.build.lib.packages.StarlarkInfoWithSchema;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link StarlarkInfoCodec}. */
@RunWith(JUnit4.class)
public final class StarlarkInfoCodecTest {
  @Test
  public void objectCodecTests() throws Exception {
    StarlarkProvider provider = makeProvider();
    for (int fieldCount : new int[] {0, 1, 2, 3, 4, 5, 6, 17}) {
      ImmutableMap.Builder<String, Object> fields = ImmutableMap.builder();
      for (int i = 0; i < fieldCount; i++) {
        fields.put("field" + i, StarlarkInt.of(i));
      }
      ImmutableMap<String, Object> map = fields.buildOrThrow();
      for (boolean compact : new boolean[] {false, true}) {
        StarlarkInfo[] infos =
            new StarlarkInfo[] {
              StarlarkInfo.create(provider, map),
              // with an error message
              StarlarkInfoWithMessage.createWithCustomMessage(provider, map, "Dummy error: %s"),
              StarlarkInfoWithMessage.createWithCustomMessage(
                  StructProvider.STRUCT, map, "Dummy error: %s")
            };
        if (compact) {
          for (int i = 0; i < infos.length; i++) {
            infos[i] = infos[i].unsafeOptimizeMemoryLayout();
          }
        }
        new SerializationTester((Object[]) infos)
            .addDependency(StructProvider.class, StructProvider.STRUCT)
            .makeMemoizing()
            .setVerificationFunction(StarlarkInfoCodecTest::verificationFunction)
            .runTests();
      }
    }
  }

  private static void verificationFunction(StarlarkInfo original, StarlarkInfo deserialized)
      throws Exception {
    assertThat(deserialized).isEqualTo(original);
    assertThat(deserialized.getFieldNames())
        .containsExactlyElementsIn(original.getFieldNames())
        .inOrder();
    if (original instanceof StarlarkInfoWithSchema) {
      var field = StarlarkInfoWithSchema.class.getDeclaredField("schema");
      field.setAccessible(true);
      assertThat(field.get(deserialized)).isNotNull();
      // StructProvider is supplied as a constant dependency.
      if (original.getProvider() == StructProvider.STRUCT) {
        assertThat(field.get(deserialized)).isSameInstanceAs(field.get(original));
      }
    }
    assertThat(deserialized.getErrorMessageForUnknownField("absent"))
        .isEqualTo(original.getErrorMessageForUnknownField("absent"));
    assertThat(deserialized.hashCode()).isEqualTo(original.hashCode());
  }

  /** Returns an exported, schemaless provider. */
  private static StarlarkProvider makeProvider() {
    return StarlarkProvider.builder(Location.BUILTIN)
        .buildExported(
            new StarlarkProvider.Key(
                keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl")), "foo"));
  }
}
