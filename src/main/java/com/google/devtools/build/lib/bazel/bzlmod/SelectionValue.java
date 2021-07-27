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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** The result of running selection, containing the dependency graph post-version-resolution. */
@AutoValue
public abstract class SelectionValue implements SkyValue {

  @AutoCodec public static final SkyKey KEY = () -> SkyFunctions.SELECTION;

  public static SelectionValue create(
      String rootModuleName,
      ImmutableMap<ModuleKey, Module> depGraph,
      ImmutableMap<String, ModuleKey> canonicalRepoNameLookup,
      ImmutableMap<String, ModuleKey> moduleNameLookup) {
    return new AutoValue_SelectionValue(
        rootModuleName, depGraph, canonicalRepoNameLookup, moduleNameLookup);
  }

  public abstract String getRootModuleName();

  /** The post-selection dep graph. Must have BFS iteration order, starting from the root module. */
  public abstract ImmutableMap<ModuleKey, Module> getDepGraph();

  /** A mapping from a canonical repo name to the key of the module backing it. */
  public abstract ImmutableMap<String, ModuleKey> getCanonicalRepoNameLookup();

  /**
   * A mapping from a plain module name to the key of the module (only works for modules without
   * multiple-version overrides).
   */
  public abstract ImmutableMap<String, ModuleKey> getModuleNameLookup();
}
