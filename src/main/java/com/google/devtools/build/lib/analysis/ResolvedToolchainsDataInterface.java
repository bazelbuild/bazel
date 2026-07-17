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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;

/**
 * Interface for resolved toolchains data.
 *
 * <p>This interface is used to provide toolchain data to Starlark. This data can be the {@link
 * ToolchainInfo} provider as in {@link ResolvedToolchainContext} for the aspect/rule own
 * toolchains, or it can be collection of aspects providers evaluated on the aspect's base target's
 * toolchains as in {@link AspectBaseTargetResolvedToolchainContext}.
 */
public interface ResolvedToolchainsDataInterface<T extends ResolvedToolchainData>
    extends ToolchainContext {
  /** Returns a description of the target being used, for error messaging. */
  public String targetDescription();

  /** Returns the map from requested {@link Label} to toolchain type provider. */
  public ImmutableMap<Label, ToolchainTypeInfo> requestedToolchainTypeLabels();

  /**
   * Returns the toolchain data for the given type, or {@code null} if the toolchain type was not
   * required in this context.
   */
  @Nullable
  public T forToolchainType(Label toolchainTypeLabel);
}
