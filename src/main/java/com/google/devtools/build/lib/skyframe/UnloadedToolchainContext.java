// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * Represents the state of toolchain resolution once the specific required toolchains have been
 * determined, but before the toolchain dependencies have been resolved.
 */
public interface UnloadedToolchainContext extends ToolchainContext, SkyValue {

  /** The map of toolchain type to resolved toolchain to be used. */
  ImmutableBiMap<ToolchainTypeInfo, Label> toolchainTypeToResolved();

  /**
   * Maps from the actual requested {@link Label} to the discovered {@link ToolchainTypeInfo}.
   *
   * <p>Note that the key may be different from {@link ToolchainTypeInfo#typeLabel()} if the
   * requested {@link Label} is an {@code alias}. In this case, there will be two {@link Label
   * labels} for the same {@link ToolchainTypeInfo}.
   */
  ImmutableMap<Label, ToolchainTypeInfo> requestedLabelToToolchainType();

  @Override
  ImmutableSet<Label> resolvedToolchainLabels();
}
