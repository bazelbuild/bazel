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
    ImmutableMap<String, Object> map =
        ImmutableMap.of("a", StarlarkInt.of(1), "b", StarlarkInt.of(2), "c", StarlarkInt.of(3));
    StarlarkProvider provider = makeProvider();
    new SerializationTester(
            StarlarkInfo.create(provider, map, /* loc= */ null),
            // empty
            StarlarkInfo.create(provider, ImmutableMap.of(), /* loc= */ null),
            // with an error message
            StarlarkInfoWithMessage.createWithCustomMessage(provider, map, "Dummy error: %s"),
            StarlarkInfoWithMessage.createWithCustomMessage(
                StructProvider.STRUCT, map, "Dummy error: %s"))
        .addDependency(StructProvider.class, StructProvider.STRUCT)
        .makeMemoizing()
        .setVerificationFunction(StarlarkInfoCodecTest::verificationFunction)
        .runTests();
  }

  private static void verificationFunction(StarlarkInfo original, StarlarkInfo deserialized) {
    assertThat(deserialized).isEqualTo(original);
    assertThat(deserialized.getFieldNames())
        .containsExactlyElementsIn(original.getFieldNames())
        .inOrder();
  }

  /** Returns an exported, schemaless provider. */
  private static StarlarkProvider makeProvider() {
    return StarlarkProvider.builder(Location.BUILTIN)
        .buildExported(
            new StarlarkProvider.Key(
                keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl")), "foo"));
  }
}
