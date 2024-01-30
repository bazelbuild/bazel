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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;

/** Stores the values of flags and environment variables that affect the resolution */
@AutoValue
@GenerateTypeAdapter
abstract class BzlmodFlagsAndEnvVars {

  public static BzlmodFlagsAndEnvVars create(
      ImmutableList<String> registries,
      ImmutableMap<String, String> moduleOverrides,
      ImmutableList<String> yankedVersions,
      String envVarYankedVersions,
      boolean ignoreDevDeps,
      String directDepsMode,
      String compatabilityMode) {
    return new AutoValue_BzlmodFlagsAndEnvVars(
        registries,
        moduleOverrides,
        yankedVersions,
        envVarYankedVersions,
        ignoreDevDeps,
        directDepsMode,
        compatabilityMode);
  }

  /** Registries provided via command line */
  public abstract ImmutableList<String> cmdRegistries();

  /** ModulesOverride provided via command line */
  public abstract ImmutableMap<String, String> cmdModuleOverrides();

  /** Allowed yanked version in the dependency graph */
  public abstract ImmutableList<String> allowedYankedVersions();

  /** Allowed yanked version in the dependency graph from environment variables */
  public abstract String envVarAllowedYankedVersions();

  /** Whether to ignore things declared as dev dependencies or not */
  public abstract boolean ignoreDevDependency();

  /** Error level of direct dependencies check */
  public abstract String directDependenciesMode();

  /** Error level of bazel compatability check */
  public abstract String compatibilityMode();

  public ImmutableList<String> getDiffFlags(BzlmodFlagsAndEnvVars flags) {
    ImmutableList.Builder<String> diffFlags = new ImmutableList.Builder<>();
    if (!flags.cmdRegistries().equals(cmdRegistries())) {
      diffFlags.add("the value of --registry flag has been modified");
    }
    if (!flags.cmdModuleOverrides().equals(cmdModuleOverrides())) {
      diffFlags.add("the value of --override_module flag has been modified");
    }
    if (!flags.allowedYankedVersions().equals(allowedYankedVersions())) {
      diffFlags.add("the value of --allow_yanked_versions flag has been modified");
    }
    if (!flags.envVarAllowedYankedVersions().equals(envVarAllowedYankedVersions())) {
      diffFlags.add(
          "the value of BZLMOD_ALLOW_YANKED_VERSIONS environment variable has been modified");
    }
    if (flags.ignoreDevDependency() != ignoreDevDependency()) {
      diffFlags.add("the value of --ignore_dev_dependency flag has been modified");
    }
    if (!flags.directDependenciesMode().equals(directDependenciesMode())) {
      diffFlags.add("the value of --check_direct_dependencies flag has been modified");
    }
    if (!flags.compatibilityMode().equals(compatibilityMode())) {
      diffFlags.add("the value of --check_bazel_compatibility flag has been modified");
    }
    return diffFlags.build();
  }
}
