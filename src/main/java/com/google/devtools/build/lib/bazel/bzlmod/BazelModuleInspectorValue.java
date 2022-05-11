// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * The result of running Bazel module inspection pre-processing, containing the un-pruned and
 * augmented wrappers of the Bazel module dependency graph (post-version-resolution).
 */
@AutoValue
public abstract class BazelModuleInspectorValue implements SkyValue {

  @SerializationConstant
  public static final SkyKey KEY = () -> SkyFunctions.BAZEL_MODULE_INSPECTION;

  public static BazelModuleInspectorValue create(
      ImmutableMap<ModuleKey, AugmentedModule> depGraph,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex) {
    return new AutoValue_BazelModuleInspectorValue(depGraph, modulesIndex);
  }

  /**
   * The (bidirectional) inspection dep graph, containing wrappers of the {@link Module}, augmented
   * with references to dependants. The order is non-deterministic, inherited from the {@code
   * completeDepGraph} of {@link BazelModuleResolutionValue}. For any KEY in the returned map, it's
   * guaranteed that {@code depGraph[KEY].getKey() == KEY}.
   */
  public abstract ImmutableMap<ModuleKey, AugmentedModule> getDepGraph();

  /**
   * Index of all module keys mentioned in the un-pruned dep graph (loaded or not) for easy lookup.
   * It is a map from <i>module name</i> to the set of {@link ModuleKey}s that point to a version of
   * that module.
   */
  public abstract ImmutableMap<String, ImmutableSet<ModuleKey>> getModulesIndex();

  /**
   * A wrapper for {@link Module}, augmented with references to dependants (and also those who are
   * not used in the final dep graph).
   */
  @AutoValue
  abstract static class AugmentedModule {
    /** Name of the module. Same as in {@link Module}. */
    abstract String getName();

    /** Version of the module. Same as in {@link Module}. */
    abstract Version getVersion();

    /** {@link ModuleKey} of this module. Same as in {@link Module} */
    abstract ModuleKey getKey();

    /**
     * The set of modules in the resolved dep graph that depend on this module
     * <strong>after</strong> the module resolution.
     */
    abstract ImmutableSet<ModuleKey> getDependants();

    /**
     * The set of modules in the complete dep graph that originally depended on this module *before*
     * the module resolution (can contain unused nodes).
     */
    abstract ImmutableSet<ModuleKey> getOriginalDependants();

    /**
     * A map from the resolved dependencies of this module to the rules that were used for their
     * resolution (can be either the original dependency, changed by the Minimal-Version Selection
     * algorithm or by an override rule
     */
    abstract ImmutableMap<ModuleKey, ResolutionReason> getDeps();

    /**
     * Flag that tell whether the module was loaded and added to the dependency graph. Modules
     * overridden by {@code single_version_override} and {@link NonRegistryOverride} are not loaded
     * so their {@code originalDeps} are (yet) unknown.
     */
    abstract boolean isLoaded();

    /** Flag for checking whether the module is present in the resolved dep graph. */
    boolean isUsed() {
      return !getDependants().isEmpty();
    }

    /** Returns a new {@link AugmentedModule.Builder} with {@code key} set. */
    public static AugmentedModule.Builder builder(ModuleKey key) {
      return new AutoValue_BazelModuleInspectorValue_AugmentedModule.Builder()
          .setName(key.getName())
          .setVersion(key.getVersion())
          .setKey(key)
          .setLoaded(false);
    }

    /** Builder type for {@link AugmentedModule}. */
    @AutoValue.Builder
    public abstract static class Builder {
      public abstract AugmentedModule.Builder setName(String value);

      public abstract AugmentedModule.Builder setVersion(Version value);

      public abstract AugmentedModule.Builder setKey(ModuleKey value);

      public abstract AugmentedModule.Builder setLoaded(boolean value);

      public abstract AugmentedModule.Builder setOriginalDependants(ImmutableSet<ModuleKey> value);

      public abstract AugmentedModule.Builder setDependants(ImmutableSet<ModuleKey> value);

      public abstract AugmentedModule.Builder setDeps(
          ImmutableMap<ModuleKey, ResolutionReason> value);

      abstract ImmutableSet.Builder<ModuleKey> originalDependantsBuilder();

      public AugmentedModule.Builder addOriginalDependant(ModuleKey depKey) {
        originalDependantsBuilder().add(depKey);
        return this;
      }

      abstract ImmutableSet.Builder<ModuleKey> dependantsBuilder();

      public AugmentedModule.Builder addDependant(ModuleKey depKey) {
        dependantsBuilder().add(depKey);
        return this;
      }

      abstract ImmutableMap.Builder<ModuleKey, ResolutionReason> depsBuilder();

      public AugmentedModule.Builder addDep(ModuleKey depKey, ResolutionReason reason) {
        depsBuilder().put(depKey, reason);
        return this;
      }

      abstract AugmentedModule build();
    }

    /** The reason why a final dependency of a module was resolved the way it was. */
    enum ResolutionReason {
      /** The dependency is the original dependency defined in the MODULE.bazel file. */
      ORIGINAL,
      /** The dependency was replaced by the Minimal-Version Selection algorithm. */
      MINIMAL_VERSION_SELECTION,
      /** The dependency was replaced by a {@code single_version_override} rule. */
      SINGLE_VERSION_OVERRIDE,
      /** The dependency was replaced by a {@code multiple_version_override} rule. */
      MULTIPLE_VERSION_OVERRIDE,
      /** The dependency was replaced by a {@link NonRegistryOverride} rule. */
      NON_REGISTRY_OVERRIDE
    }
  }
}
