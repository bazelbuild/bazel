// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value which represents every execution platform known to Bazel and available to run actions.
 */
@AutoValue
public abstract class RegisteredExecutionPlatformsValue implements SkyValue {

  /** Returns the {@link SkyKey} for {@link RegisteredExecutionPlatformsValue}s. */
  public static SkyKey key(BuildConfigurationKey configurationKey) {
    return Key.of(configurationKey);
  }

  /** {@link SkyKey} implementation used for {@link RegisteredExecutionPlatformsFunction}. */
  @AutoCodec
  @VisibleForSerialization
  static class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private final BuildConfigurationKey configurationKey;

    private Key(BuildConfigurationKey configurationKey) {
      this.configurationKey = configurationKey;
    }

    private static Key of(BuildConfigurationKey configurationKey) {
      return interner.intern(new Key(configurationKey));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.REGISTERED_EXECUTION_PLATFORMS;
    }

    BuildConfigurationKey getConfigurationKey() {
      return configurationKey;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Key that)) {
        return false;
      }
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

  static RegisteredExecutionPlatformsValue create(
      Iterable<ConfiguredTargetKey> registeredExecutionPlatformKeys) {
    return new AutoValue_RegisteredExecutionPlatformsValue(
        ImmutableList.copyOf(registeredExecutionPlatformKeys));
  }

  public abstract ImmutableList<ConfiguredTargetKey> registeredExecutionPlatformKeys();
}
