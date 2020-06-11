// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnCache;

/**
 * Module providing a default no-op spawn cache.
 */
public final class NoSpawnCacheModule extends BlazeModule {

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    registryBuilder.register(SpawnCache.class, SpawnCache.NO_CACHE, "no-cache");
  }
}
