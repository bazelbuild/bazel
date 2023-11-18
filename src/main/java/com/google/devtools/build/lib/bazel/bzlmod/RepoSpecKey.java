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
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/** The key for {@link RepoSpecFunction}. */
@AutoCodec
@AutoValue
abstract class RepoSpecKey implements SkyKey {
  private static final SkyKeyInterner<RepoSpecKey> interner = SkyKey.newInterner();

  static RepoSpecKey of(InterimModule module) {
    Preconditions.checkNotNull(
        module.getRegistry(), "module must not have a non-registry override");
    return create(module.getKey(), module.getRegistry().getUrl());
  }

  abstract ModuleKey getModuleKey();

  abstract String getRegistryUrl();

  @AutoCodec.Instantiator
  static RepoSpecKey create(ModuleKey moduleKey, String registryUrl) {
    return interner.intern(new AutoValue_RepoSpecKey(moduleKey, registryUrl));
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.REPO_SPEC;
  }

  @Override
  public SkyKeyInterner<RepoSpecKey> getSkyKeyInterner() {
    return interner;
  }
}
