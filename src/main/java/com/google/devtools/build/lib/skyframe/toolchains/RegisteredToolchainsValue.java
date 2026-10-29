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

package com.google.devtools.build.lib.skyframe.toolchains;

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A value which represents every toolchain known to Bazel and available for toolchain resolution.
 *
 * @param rejectedToolchains Any toolchains that were rejected, along with a reason. The row keys
 *     are the toolchain type labels, column keys are toolchain target (not implementation) labels,
 *     and cells are the reason. Only non-null if {@link RegisteredToolchainsValue.Key#debug} is
 *     {@code true}.
 */
@AutoCodec
public record RegisteredToolchainsValue(
    ImmutableList<DeclaredToolchainInfo> registeredToolchains,
    @Nullable ImmutableTable<Label, Label, String> rejectedToolchains)
    implements SkyValue {
  public RegisteredToolchainsValue {
    requireNonNull(registeredToolchains, "registeredToolchains");
  }

  /**
   * Returns the {@link SkyKey} for {@link RegisteredToolchainsValue}s.
   *
   * @param typeOwnerRepo if not null, the value only contains the toolchains registered with {@code
   *     --extra_toolchains}, by the root module and by the module with this repo
   */
  public static Key key(
      BuildConfigurationKey configurationKey,
      boolean debug,
      @Nullable RepositoryName typeOwnerRepo) {
    return Key.of(configurationKey, debug, typeOwnerRepo);
  }

  /** A {@link SkyKey} for {@code RegisteredToolchainsValue}. */
  @AutoCodec
  public static class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private final BuildConfigurationKey configurationKey;
    private final boolean debug;
    @Nullable private final RepositoryName typeOwnerRepo;

    private Key(
        BuildConfigurationKey configurationKey,
        boolean debug,
        @Nullable RepositoryName typeOwnerRepo) {
      this.configurationKey = configurationKey;
      this.debug = debug;
      this.typeOwnerRepo = typeOwnerRepo;
    }

    private static Key of(
        BuildConfigurationKey configurationKey,
        boolean debug,
        @Nullable RepositoryName typeOwnerRepo) {
      return interner.intern(new Key(configurationKey, debug, typeOwnerRepo));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REGISTERED_TOOLCHAINS;
    }

    BuildConfigurationKey getConfigurationKey() {
      return configurationKey;
    }

    boolean debug() {
      return debug;
    }

    @Nullable
    RepositoryName typeOwnerRepo() {
      return typeOwnerRepo;
    }

    @Override
    public String toString() {
      return "RegisteredToolchainsValue.Key{"
          + "configurationKey: "
          + configurationKey
          + ", debug: "
          + debug
          + ", typeOwnerRepo: "
          + typeOwnerRepo
          + "}";
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Key that)) {
        return false;
      }
      return Objects.equals(this.configurationKey, that.configurationKey)
          && this.debug == that.debug
          && Objects.equals(this.typeOwnerRepo, that.typeOwnerRepo);
    }

    @Override
    public int hashCode() {
      return Objects.hash(configurationKey, debug, typeOwnerRepo);
    }

    @Override
    public final SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  public static RegisteredToolchainsValue create(
      ImmutableList<DeclaredToolchainInfo> registeredToolchains,
      @Nullable ImmutableTable<Label, Label, String> rejectedToolchains) {
    return new RegisteredToolchainsValue(registeredToolchains, rejectedToolchains);
  }

}
