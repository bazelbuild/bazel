// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.Label;

/** Describes a requirement on a specific toolchain type. */
@AutoValue
public abstract class ToolchainTypeRequirement {

  /** Returns a new {@link ToolchainTypeRequirement}. */
  public static ToolchainTypeRequirement create(Label toolchainType) {
    return builder(toolchainType).build();
  }

  /** Returns a builder for a new {@link ToolchainTypeRequirement}. */
  public static Builder builder(Label toolchainType) {
    return builder().toolchainType(toolchainType);
  }

  /** Returns a builder for a new {@link ToolchainTypeRequirement}. */
  public static Builder builder() {
    return new AutoValue_ToolchainTypeRequirement.Builder().mandatory(true);
  }

  /** Returns the label of the toolchain type that is requested. */
  public abstract Label toolchainType();

  /**
   * Returns whether the toolchain type is mandatory or optional. An optional toolchain type which
   * cannot be found will be skipped, but a mandatory toolchain type which cannot be found will stop
   * the build with an error.
   */
  public abstract boolean mandatory();

  /** A builder for a new {@link ToolchainTypeRequirement}. */
  @AutoValue.Builder
  public interface Builder {
    /** Sets the toolchain type. */
    Builder toolchainType(Label toolchainType);
    /** Sets whether the toolchain type is mandatory. */
    Builder mandatory(boolean mandatory);

    /** Returns the newly built {@link ToolchainTypeRequirement}. */
    ToolchainTypeRequirement build();
  }
}
