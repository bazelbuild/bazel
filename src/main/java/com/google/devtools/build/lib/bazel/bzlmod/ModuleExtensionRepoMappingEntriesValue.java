// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** The value for {@link ModuleExtensionRepoMappingEntriesFunction}. */
@AutoCodec
@AutoValue
public abstract class ModuleExtensionRepoMappingEntriesValue implements SkyValue {

  public abstract ImmutableMap<String, RepositoryName> getEntries();

  public abstract ModuleKey getModuleKey();

  @AutoCodec.Instantiator
  public static ModuleExtensionRepoMappingEntriesValue create(
      ImmutableMap<String, RepositoryName> entries, ModuleKey moduleKey) {
    return new AutoValue_ModuleExtensionRepoMappingEntriesValue(entries, moduleKey);
  }

  public static ModuleExtensionRepoMappingEntriesValue.Key key(ModuleExtensionId id) {
    return ModuleExtensionRepoMappingEntriesValue.Key.create(id);
  }

  /**
   * The {@link com.google.devtools.build.skyframe.SkyKey} of a {@link
   * ModuleExtensionRepoMappingEntriesValue}.
   */
  @AutoCodec
  public static class Key extends AbstractSkyKey<ModuleExtensionId> {

    private static final SkyKeyInterner<ModuleExtensionRepoMappingEntriesValue.Key> interner =
        SkyKey.newInterner();

    protected Key(ModuleExtensionId arg) {
      super(arg);
    }

    private static ModuleExtensionRepoMappingEntriesValue.Key create(ModuleExtensionId arg) {
      return interner.intern(new ModuleExtensionRepoMappingEntriesValue.Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.MODULE_EXTENSION_REPO_MAPPING_ENTRIES;
    }

    @Override
    public SkyKeyInterner<ModuleExtensionRepoMappingEntriesValue.Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
