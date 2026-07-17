// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.testutils;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import net.starlark.java.eval.Module;

/** Helpers for serialization tests. */
public class TestUtils {

  private TestUtils() {}

  /**
   * Asserts that two {@link Module}s have the same structure. Needed because {@link Module} doesn't
   * override {@link Object#equals}.
   */
  public static void assertModulesEqual(Module module1, Module module2) {
    assertThat(module1.getClientData()).isEqualTo(module2.getClientData());
    assertThat(module1.getGlobals()).containsExactlyEntriesIn(module2.getGlobals()).inOrder();
    assertThat(module1.getPredeclaredBindings())
        .containsExactlyEntriesIn(module2.getPredeclaredBindings())
        .inOrder();
  }

  public static ObjectCodecRegistry.Builder getBuilderWithAdditionalCodecs(
      ObjectCodec<?>... codecs) {
    ObjectCodecRegistry.Builder builder = AutoRegistry.get().getBuilder();
    for (ObjectCodec<?> codec : codecs) {
      builder.add(codec);
    }
    return builder;
  }
}
