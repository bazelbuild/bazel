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
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;
import javax.annotation.Nullable;

/** The result of {@link ModuleFileFunction}. */
public abstract class ModuleFileValue implements SkyValue {

  public static final ModuleFileValue.Key KEY_FOR_ROOT_MODULE = key(ModuleKey.ROOT, null);

  /**
   * The module resulting from the module file evaluation. Note that the name and version of this
   * module might not match the one in the requesting {@link SkyKey} in certain circumstances (for
   * example, for the root module, or when non-registry overrides are in play.
   */
  public abstract InterimModule getModule();

  /**
   * Hashes of files obtained (or known to be missing) from registries while obtaining this module
   * file.
   */
  public abstract ImmutableMap<String, Optional<Checksum>> getRegistryFileHashes();

  /** The {@link ModuleFileValue} for non-root modules. */
  @AutoValue
  public abstract static class NonRootModuleFileValue extends ModuleFileValue {

    public static NonRootModuleFileValue create(
        InterimModule module,
        ImmutableMap<String, Optional<Checksum>> registryFileHashes) {
      return new AutoValue_ModuleFileValue_NonRootModuleFileValue(module, registryFileHashes);
    }
  }

  /**
   * The {@link ModuleFileValue} for the root module, containing additional information about
   * overrides.
   */
  @AutoValue
  public abstract static class RootModuleFileValue extends ModuleFileValue {
    /**
     * The overrides specified by the evaluated module file. The key is the module name and the
     * value is the override itself.
     */
    public abstract ImmutableMap<String, ModuleOverride> getOverrides();

    /**
     * A mapping from a canonical repo name to the name of the module. Only works for modules with
     * non-registry overrides.
     */
    public abstract ImmutableMap<RepositoryName, String>
        getNonRegistryOverrideCanonicalRepoNameLookup();

    /**
     * The set of relative paths to the root MODULE.bazel file itself and all its transitive
     * includes.
     */
    public abstract ImmutableSet<PathFragment> getModuleFilePaths();

    @Override
    public ImmutableMap<String, Optional<Checksum>> getRegistryFileHashes() {
      // The root module is not obtained from a registry.
      return ImmutableMap.of();
    }

    public static RootModuleFileValue create(
        InterimModule module,
        ImmutableMap<String, ModuleOverride> overrides,
        ImmutableMap<RepositoryName, String> nonRegistryOverrideCanonicalRepoNameLookup,
        ImmutableSet<PathFragment> moduleFilePaths) {
      return new AutoValue_ModuleFileValue_RootModuleFileValue(
          module,
          overrides,
          nonRegistryOverrideCanonicalRepoNameLookup,
          moduleFilePaths);
    }
  }

  public static Key key(ModuleKey moduleKey, @Nullable ModuleOverride override) {
    return Key.create(moduleKey, override);
  }

  /** {@link SkyKey} for {@link ModuleFileValue} computation. */
  @AutoCodec
  @AutoValue
  public abstract static class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    abstract ModuleKey getModuleKey();

    @Nullable
    abstract ModuleOverride getOverride();

    @AutoCodec.Instantiator
    static Key create(ModuleKey moduleKey, @Nullable ModuleOverride override) {
      return interner.intern(new AutoValue_ModuleFileValue_Key(moduleKey, override));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.MODULE_FILE;
    }

    @Memoized
    @Override
    public abstract int hashCode();

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
