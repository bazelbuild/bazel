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

package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.skyframe.LegacySkyKey;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value which represents every toolchain known to Bazel and available for toolchain resolution.
 */
@AutoValue
public abstract class RegisteredToolchainsValue implements SkyValue {

  /** Returns the {@link SkyKey} for {@link RegisteredToolchainsValue}s. */
  public static SkyKey key(BuildConfiguration configuration) {
    return LegacySkyKey.create(SkyFunctions.REGISTERED_TOOLCHAINS, configuration);
  }

  public static RegisteredToolchainsValue create(
      Iterable<DeclaredToolchainInfo> registeredToolchains) {
    return new AutoValue_RegisteredToolchainsValue(ImmutableList.copyOf(registeredToolchains));
  }

  public abstract ImmutableList<DeclaredToolchainInfo> registeredToolchains();
}
