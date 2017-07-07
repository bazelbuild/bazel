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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import java.util.Map;
import javax.annotation.Nullable;

/** Contains toolchain-related information needed for a {@link RuleContext}. */
public class ToolchainContext {
  private final ImmutableList<Label> requiredToolchains;
  private final ImmutableMap<Label, ToolchainInfo> toolchains;

  public ToolchainContext(
      ImmutableList<Label> requiredToolchains, @Nullable Map<Label, ToolchainInfo> toolchains) {
    this.requiredToolchains = requiredToolchains;
    this.toolchains =
        toolchains == null
            ? ImmutableMap.<Label, ToolchainInfo>of()
            : ImmutableMap.copyOf(toolchains);
  }

  public ImmutableList<Label> getRequiredToolchains() {
    return requiredToolchains;
  }

  public SkylarkDict<Label, ToolchainInfo> collectToolchains() {
    return SkylarkDict.<Label, ToolchainInfo>copyOf(null, toolchains);
  }
}
