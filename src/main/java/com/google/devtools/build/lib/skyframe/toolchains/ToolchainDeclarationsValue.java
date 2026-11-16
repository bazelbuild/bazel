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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * The registered toolchain declarations, in order of precedence, analyzed as far as possible
 * without a target configuration.
 *
 * <p>This value only depends on the registration inputs (the {@code --extra_toolchains} flag and
 * the {@code register_toolchains} calls of all modules) and the no-config configuration, not on the
 * full target configuration. This allows {@link RegisteredToolchainsFunction} to share the analysis
 * of most toolchain declarations across all configurations.
 *
 * @param labels the labels of the registered targets
 * @param configIndependentToolchains the declared toolchains of the registered targets that could
 *     be analyzed without a target configuration. All other targets have to be analyzed in each
 *     target configuration (e.g. because they use {@code select()}).
 */
@AutoCodec
public record ToolchainDeclarationsValue(
    ImmutableList<Label> labels,
    ImmutableMap<Label, DeclaredToolchainInfo> configIndependentToolchains)
    implements SkyValue {
  public ToolchainDeclarationsValue {
    requireNonNull(labels, "labels");
    requireNonNull(configIndependentToolchains, "configIndependentToolchains");
  }

  // Values are compared by identity so that every configuration observes the declared toolchains of
  // the latest analysis: the equality of DeclaredToolchainInfo doesn't cover everything that
  // affects toolchain resolution, such as the default value of a constraint setting.
  @Override
  public boolean equals(Object other) {
    return this == other;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }

  /**
   * A {@link SkyKey} for {@link ToolchainDeclarationsValue}.
   *
   * @param extraToolchains the value of {@code --extra_toolchains}, in the order given on the
   *     command line
   * @param noConfigOptions the options of the no-config configuration in which the declarations are
   *     analyzed
   */
  @AutoCodec
  public record Key(ImmutableList<String> extraToolchains, BuildOptions noConfigOptions)
      implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    /**
     * @deprecated Use {@link #create} instead to ensure interning.
     */
    @Deprecated
    public Key {
      requireNonNull(extraToolchains, "extraToolchains");
      requireNonNull(noConfigOptions, "noConfigOptions");
    }

    @AutoCodec.Instantiator
    public static Key create(ImmutableList<String> extraToolchains, BuildOptions noConfigOptions) {
      return interner.intern(new Key(extraToolchains, noConfigOptions));
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
