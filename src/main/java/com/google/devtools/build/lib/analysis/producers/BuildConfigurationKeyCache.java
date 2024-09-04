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
package com.google.devtools.build.lib.analysis.producers;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import javax.annotation.Nullable;

/**
 * A cache of {@link BuildOptions} -> {@link BuildConfigurationKey} instances, taking platform
 * mappings and platform based flags into account.
 */
public class BuildConfigurationKeyCache {
  private final Cache<BuildOptions, BuildConfigurationKey> cache = Caffeine.newBuilder().build();

  @Nullable
  public BuildConfigurationKey get(BuildOptions options) {
    return cache.getIfPresent(options);
  }

  public void put(BuildOptions options, BuildConfigurationKey buildConfigurationKey) {
    cache.put(options, buildConfigurationKey);
  }

  public void clear() {
    cache.invalidateAll();
  }
}
