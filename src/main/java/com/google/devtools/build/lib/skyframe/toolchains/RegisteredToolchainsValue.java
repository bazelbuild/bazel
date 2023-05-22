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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value which represents every toolchain known to Bazel and available for toolchain resolution.
 */
@AutoValue
public abstract class RegisteredToolchainsValue implements SkyValue {

  /** Returns the {@link SkyKey} for {@link RegisteredToolchainsValue}s. */
  public static Key key(BuildConfigurationKey configurationKey) {
    return Key.of(configurationKey);
  }

  /** A {@link SkyKey} for {@code RegisteredToolchainsValue}. */
  @AutoCodec
  static class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private final BuildConfigurationKey configurationKey;

    private Key(BuildConfigurationKey configurationKey) {
      this.configurationKey = configurationKey;
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key of(BuildConfigurationKey configurationKey) {
      return interner.intern(new Key(configurationKey));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REGISTERED_TOOLCHAINS;
    }

    BuildConfigurationKey getConfigurationKey() {
      return configurationKey;
    }

    @Override
    public String toString() {
      return new StringBuilder()
          .append("RegisteredToolchainsValue.Key{")
          .append("configurationKey: ")
          .append(configurationKey)
          .append("}")
          .toString();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Key)) {
        return false;
      }
      Key that = (Key) obj;
      return Objects.equals(this.configurationKey, that.configurationKey);
    }

    @Override
    public int hashCode() {
      return configurationKey.hashCode();
    }

    @Override
    public final SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  public static RegisteredToolchainsValue create(
      ImmutableList<DeclaredToolchainInfo> registeredToolchains) {
    return new AutoValue_RegisteredToolchainsValue(registeredToolchains);
  }

  public abstract ImmutableList<DeclaredToolchainInfo> registeredToolchains();
}
