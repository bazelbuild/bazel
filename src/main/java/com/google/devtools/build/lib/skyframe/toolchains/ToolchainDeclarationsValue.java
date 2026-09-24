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
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * The registered toolchain declarations, in order of precedence, analyzed as far as possible
 * without a target configuration.
 *
 * <p>This value only depends on the registration inputs (the {@code --extra_toolchains} flag and
 * the {@code register_toolchains} calls of all modules), not on the full target configuration. This
 * allows {@link RegisteredToolchainsFunction} to share the analysis of most toolchain declarations
 * across all configurations.
 */
@AutoCodec
public record ToolchainDeclarationsValue(ImmutableList<Declaration> declarations)
    implements SkyValue {
  public ToolchainDeclarationsValue {
    requireNonNull(declarations, "declarations");
  }

  /**
   * A single registered toolchain declaration.
   *
   * @param label the label of the registered target
   * @param configIndependentInfo the declared toolchain if it could be analyzed without a target
   *     configuration, or {@code null} if the target has to be analyzed in each target
   *     configuration (e.g. because it uses {@code select()})
   */
  @AutoCodec
  public record Declaration(Label label, @Nullable DeclaredToolchainInfo configIndependentInfo) {
    public Declaration {
      requireNonNull(label, "label");
    }
  }

  public static Key key(ImmutableList<String> extraToolchains) {
    return Key.of(extraToolchains);
  }

  /** A {@link SkyKey} for {@link ToolchainDeclarationsValue}. */
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private final ImmutableList<String> extraToolchains;

    private Key(ImmutableList<String> extraToolchains) {
      this.extraToolchains = requireNonNull(extraToolchains);
    }

    private static Key of(ImmutableList<String> extraToolchains) {
      return interner.intern(new Key(extraToolchains));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TOOLCHAIN_DECLARATIONS;
    }

    /** The value of {@code --extra_toolchains}, in the order given on the command line. */
    ImmutableList<String> extraToolchains() {
      return extraToolchains;
    }

    @Override
    public String toString() {
      return "ToolchainDeclarationsValue.Key{extraToolchains: " + extraToolchains + "}";
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof Key that && extraToolchains.equals(that.extraToolchains);
    }

    @Override
    public int hashCode() {
      return extraToolchains.hashCode();
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
