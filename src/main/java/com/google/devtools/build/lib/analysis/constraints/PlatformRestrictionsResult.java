// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.constraints;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;

/** Targets that have additional restrictions based on the current platform. */
@AutoValue
public abstract class PlatformRestrictionsResult {
  /** Targets that need be skipped. */
  public abstract ImmutableSet<ConfiguredTarget> targetsToSkip();
  /** Targets that should be skipped, but were explicitly requested on the command line. */
  public abstract ImmutableSet<ConfiguredTarget> targetsWithErrors();

  public static Builder builder() {
    return new AutoValue_PlatformRestrictionsResult.Builder()
        .targetsToSkip(ImmutableSet.of())
        .targetsWithErrors(ImmutableSet.of());
  }

  /** {@link PlatformRestrictionsResult}Builder. */
  @AutoValue.Builder
  public interface Builder {
    Builder targetsToSkip(ImmutableSet<ConfiguredTarget> targetsToSkip);

    Builder targetsWithErrors(ImmutableSet<ConfiguredTarget> targetsWithErrors);

    PlatformRestrictionsResult build();
  }
}
