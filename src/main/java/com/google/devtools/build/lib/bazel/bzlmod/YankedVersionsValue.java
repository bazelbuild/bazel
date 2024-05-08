// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;

/** A class holding information about the versions of a particular module that have been yanked. */
@AutoValue
public abstract class YankedVersionsValue implements SkyValue {

  public abstract Optional<ImmutableMap<Version, String>> yankedVersions();

  public static YankedVersionsValue create(Optional<ImmutableMap<Version, String>> yankedVersions) {
    return new AutoValue_YankedVersionsValue(yankedVersions);
  }

  /** The key for {@link YankedVersionsFunction}. */
  @AutoCodec
  @AutoValue
  abstract static class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    abstract String getModuleName();

    abstract String getRegistryUrl();

    @AutoCodec.Instantiator
    static Key create(String moduleName, String registryUrl) {
      return interner.intern(new AutoValue_YankedVersionsValue_Key(moduleName, registryUrl));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.YANKED_VERSIONS;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
