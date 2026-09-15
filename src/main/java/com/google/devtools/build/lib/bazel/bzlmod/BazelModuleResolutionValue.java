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
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;

/**
 * The result of the selection process, containing both the pruned and the un-pruned dependency
 * graphs.
 */
@AutoValue
public abstract class BazelModuleResolutionValue implements SkyValue {

  @SerializationConstant
  public static final SkyKey KEY = () -> SkyFunctions.BAZEL_MODULE_RESOLUTION;

  /** Final dep graph sorted in BFS iteration order, with unused modules removed. */
  abstract ImmutableMap<ModuleKey, Module> getResolvedDepGraph();

  /**
   * Un-pruned dep graph, with updated dep keys, and additionally containing the unused modules
   * which were initially discovered (and their MODULE.bazel files loaded). Does not contain modules
   * overridden by {@code single_version_override} or {@link NonRegistryOverride}, only by {@code
   * multiple_version_override}.
   */
  abstract ImmutableMap<ModuleKey, InterimModule> getUnprunedDepGraph();

  /**
   * Hashes of files obtained (or known to be missing) from registries while performing resolution.
   */
  public abstract ImmutableMap<String, Optional<Checksum>> getRegistryFileHashes();

  /**
   * Selected module versions that are known to be yanked (and hence must have been explicitly
   * allowed by the user).
   */
  abstract ImmutableMap<ModuleKey, String> getSelectedYankedVersions();

  static BazelModuleResolutionValue create(
      ImmutableMap<ModuleKey, Module> resolvedDepGraph,
      ImmutableMap<ModuleKey, InterimModule> unprunedDepGraph,
      ImmutableMap<String, Optional<Checksum>> registryFileHashes,
      ImmutableMap<ModuleKey, String> selectedYankedVersions) {
    return new AutoValue_BazelModuleResolutionValue(
        resolvedDepGraph, unprunedDepGraph, registryFileHashes, selectedYankedVersions);
  }
}
