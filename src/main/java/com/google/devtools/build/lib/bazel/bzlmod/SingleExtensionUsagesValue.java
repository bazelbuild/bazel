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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.gson.Gson;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;

/**
 * The result of {@link SingleExtensionUsagesFunction}.
 *
 * <p>When adding or exposing new fields to extensions, make sure to update {@link
 * #trimForEvaluation()} as well.
 */
@AutoValue
@GenerateTypeAdapter
public abstract class SingleExtensionUsagesValue implements SkyValue {
  /** All usages of this extension, by the key of the module where the usage occurs. */
  // Note: Equality of SingleExtensionUsagesValue does not check for equality of the order of the
  // entries of this map, but it is tracked implicitly via the order of the abridged modules.
  public abstract ImmutableMap<ModuleKey, ModuleExtensionUsage> getExtensionUsages();

  /**
   * The "unique name" (see {@link BazelDepGraphValue#getExtensionUniqueNames}) of this extension.
   */
  public abstract String getExtensionUniqueName();

  /** All {@link AbridgedModule}s in the dependency graph that used this extension. */
  public abstract ImmutableList<AbridgedModule> getAbridgedModules();

  /** The repo mappings to use for each module that used this extension. */
  public abstract ImmutableMap<ModuleKey, RepositoryMapping> getRepoMappings();

  public static SingleExtensionUsagesValue create(
      ImmutableMap<ModuleKey, ModuleExtensionUsage> extensionUsages,
      String extensionUniqueName,
      ImmutableList<AbridgedModule> abridgedModules,
      ImmutableMap<ModuleKey, RepositoryMapping> repoMappings) {
    return new AutoValue_SingleExtensionUsagesValue(
        extensionUsages, extensionUniqueName, abridgedModules, repoMappings);
  }

  /**
   * Turns the given usages value for a particular extension into a hash that can be compared for
   * equality with another hash obtained in this way and compares equal only if the two values are
   * equivalent for the purpose of evaluating the extension.
   */
  static byte[] hashForEvaluation(Gson gson, SingleExtensionUsagesValue usagesValue) {
    return Hashing.sha256()
        .hashUnencodedChars(gson.toJson(usagesValue.trimForEvaluation()))
        .asBytes();
  }

  /**
   * Returns a new value with only the information that influences the evaluation of the extension
   * and isn't tracked elsewhere.
   */
  SingleExtensionUsagesValue trimForEvaluation() {
    return SingleExtensionUsagesValue.create(
        ImmutableMap.copyOf(
            Maps.transformValues(getExtensionUsages(), ModuleExtensionUsage::trimForEvaluation)),
        getExtensionUniqueName(),
        getAbridgedModules(),
        // repoMappings: The usage of repo mappings by the extension's implementation function is
        // tracked on the level of individual entries and all label attributes are provided as
        // `Label`, which exclusively reference canonical repository names.
        ImmutableMap.of());
  }

  public static Key key(ModuleExtensionId id) {
    return Key.create(id);
  }

  @AutoCodec
  static class Key extends AbstractSkyKey<ModuleExtensionId> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    protected Key(ModuleExtensionId arg) {
      super(arg);
    }

    private static Key create(ModuleExtensionId arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.SINGLE_EXTENSION_USAGES;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
