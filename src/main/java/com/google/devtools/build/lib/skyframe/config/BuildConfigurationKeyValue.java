// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/** Stores a {@link BuildConfigurationKey} with all platform mappings applied. */
@AutoCodec
public final class BuildConfigurationKeyValue implements SkyValue {

  /** Key for {@link BuildConfigurationKeyValue} based on the build options. */
  @ThreadSafety.Immutable
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    public static Key create(BuildOptions buildOptions) {
      return interner.intern(new Key(buildOptions));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    private final BuildOptions buildOptions;

    private Key(BuildOptions buildOptions) {
      this.buildOptions = buildOptions;
    }

    public BuildOptions buildOptions() {
      return buildOptions;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BUILD_CONFIGURATION_KEY;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      Key key = (Key) o;
      return Objects.equals(buildOptions, key.buildOptions);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(buildOptions);
    }

    @Override
    public String toString() {
      return "BuildConfigurationKeyValue.Key{buildOptions=" + buildOptions.checksum() + "}";
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  public static BuildConfigurationKeyValue create(BuildConfigurationKey buildConfigurationKey) {
    return new BuildConfigurationKeyValue(buildConfigurationKey);
  }

  private final BuildConfigurationKey buildConfigurationKey;

  BuildConfigurationKeyValue(BuildConfigurationKey buildConfigurationKey) {
    this.buildConfigurationKey = buildConfigurationKey;
  }

  public BuildConfigurationKey buildConfigurationKey() {
    return buildConfigurationKey;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof BuildConfigurationKeyValue that)) {
      return false;
    }
    return this.buildConfigurationKey.equals(that.buildConfigurationKey);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(buildConfigurationKey);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("buildConfigurationKey", buildConfigurationKey)
        .toString();
  }
}
