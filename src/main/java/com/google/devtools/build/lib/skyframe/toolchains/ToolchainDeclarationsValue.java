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
package com.google.devtools.build.lib.skyframe.toolchains;

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * The registered toolchain declarations, in order of precedence.
 *
 * <p>This value only depends on the registration inputs (the {@code --extra_toolchains} flag and
 * the {@code register_toolchains} calls of all modules), not on the full target configuration. This
 * allows {@link RegisteredToolchainsFunction} to share the expansion of the registered toolchains
 * across all configurations.
 *
 * @param labels the labels of the registered targets. This is a list rather than a set so that
 *     values that only differ in the order of the labels aren't equal.
 */
@AutoCodec
public record ToolchainDeclarationsValue(ImmutableList<Label> labels) implements SkyValue {
  public ToolchainDeclarationsValue {
    requireNonNull(labels, "labels");
  }

  /**
   * A {@link SkyKey} for {@link ToolchainDeclarationsValue}.
   *
   * @param extraToolchains the value of {@code --extra_toolchains}, in the order given on the
   *     command line
   */
  @AutoCodec
  public record Key(ImmutableList<String> extraToolchains) implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    /**
     * @deprecated Use {@link #create} instead to ensure interning.
     */
    @Deprecated
    public Key {
      requireNonNull(extraToolchains, "extraToolchains");
    }

    @AutoCodec.Instantiator
    public static Key create(ImmutableList<String> extraToolchains) {
      return interner.intern(new Key(extraToolchains));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TOOLCHAIN_DECLARATIONS;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
