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
package com.google.devtools.build.lib.analysis.producers;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ExecGroupCollection;
import com.google.devtools.build.lib.packages.DeclaredExecGroup;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import javax.annotation.Nullable;

/** Collates inputs for the {@link UnloadedToolchainContextsProducer}. */
@AutoValue
public abstract class UnloadedToolchainContextsInputs extends ExecGroupCollection.Builder {
  @Nullable // Null if no toolchain resolution is required.
  public abstract ToolchainContextKey targetToolchainContextKey();

  public static UnloadedToolchainContextsInputs create(
      ImmutableMap<String, DeclaredExecGroup> processedExecGroups,
      @Nullable ToolchainContextKey targetToolchainContextKey) {
    return new AutoValue_UnloadedToolchainContextsInputs(
        processedExecGroups, targetToolchainContextKey);
  }

  public static UnloadedToolchainContextsInputs empty() {
    return new AutoValue_UnloadedToolchainContextsInputs(
        ImmutableMap.of(), /* targetToolchainContextKey= */ null);
  }
}
