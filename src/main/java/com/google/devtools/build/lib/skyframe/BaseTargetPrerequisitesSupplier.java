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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import javax.annotation.Nullable;

/**
 * Looks up previously evaluated {@link ConfiguredTargetValue}s and {@link BuildConfigurationValue}s
 * without adding a dependency edge between them and the requesting node.
 *
 * <p>Mainly used by {@link AspectFunction} to look up the {@link ConfiguredTargetValue}s and {@link
 * BuildConfigurationValue} of its target dependencies.
 */
public interface BaseTargetPrerequisitesSupplier {

  /** Directly retrieves configured targets from Skyframe without adding a dependency edge. */
  @Nullable
  ConfiguredTargetValue getPrerequisite(ConfiguredTargetKey key) throws InterruptedException;

  /** Directly retrieves configuration values from Skyframe without adding a dependency edge. */
  @Nullable
  BuildConfigurationValue getPrerequisiteConfiguration(BuildConfigurationKey key)
      throws InterruptedException;

  /**
   * Directly retrieves unloaded toolchain contexts from Skyframe without adding a dependency edge.
   */
  @Nullable
  UnloadedToolchainContext getUnloadedToolchainContext(ToolchainContextKey key)
      throws InterruptedException;
}
