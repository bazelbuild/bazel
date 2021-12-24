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
package com.google.devtools.build.skyframe;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.skyframe.SkyFunction.SkyKeyComputeState;
import javax.annotation.Nullable;

/**
 * Helper class used to support {@link SkyKeyComputeState}.
 *
 * <p>TODO(b/209704702): Make this fancier as needed to reimplement some of the Blaze-on-Skyframe
 * SkyFunctions.
 */
class SkyKeyComputeStateManager {
  private final ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  private final LoadingCache<SkyKey, SkyKeyComputeState> cache;

  SkyKeyComputeStateManager(ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions) {
    this.skyFunctions = skyFunctions;
    this.cache =
        Caffeine.newBuilder()
            .build(k -> skyFunctions.get(k.functionName()).createNewSkyKeyComputeState());
  }

  @Nullable
  SkyKeyComputeState maybeGet(SkyKey skyKey) {
    return skyFunctions.get(skyKey.functionName()).supportsSkyKeyComputeState()
        ? cache.get(skyKey)
        : null;
  }

  void remove(SkyKey skyKey) {
    cache.invalidate(skyKey);
  }

  void removeAll() {
    cache.invalidateAll();
  }
}
