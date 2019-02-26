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

package com.google.devtools.build.lib.analysis.platform;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

/**
 * Provider for a toolchain declaration, which associates a toolchain type, the execution and target
 * constraints, and the actual toolchain label. The toolchain is then available for use but will be
 * lazily resolved only when it is actually needed for toolchain-aware rules. Toolchain definitions
 * are exposed to Skylark and Bazel via {@link ToolchainInfo} providers.
 */
@AutoValue
@AutoCodec
public abstract class DeclaredToolchainInfo implements TransitiveInfoProvider {
  /**
   * The type of the toolchain being declared. This will be a label of a toolchain_type() target.
   */
  public abstract ToolchainTypeInfo toolchainType();

  /** The constraints describing the execution environment. */
  public abstract ConstraintCollection execConstraints();

  /** The constraints describing the target environment. */
  public abstract ConstraintCollection targetConstraints();

  /** The label of the toolchain to resolve for use in toolchain-aware rules. */
  public abstract Label toolchainLabel();

  /** Returns a new {@link DeclaredToolchainInfo} with the given data. */
  public static DeclaredToolchainInfo create(
      ToolchainTypeInfo toolchainType,
      ImmutableList<ConstraintValueInfo> execConstraints,
      ImmutableList<ConstraintValueInfo> targetConstraints,
      Label toolchainLabel) {
    return create(
        toolchainType,
        ConstraintCollection.builder().addConstraints(execConstraints).build(),
        ConstraintCollection.builder().addConstraints(targetConstraints).build(),
        toolchainLabel);
  }

  @AutoCodec.Instantiator
  @VisibleForSerialization
  static DeclaredToolchainInfo create(
      ToolchainTypeInfo toolchainType,
      ConstraintCollection execConstraints,
      ConstraintCollection targetConstraints,
      Label toolchainLabel) {
    return new AutoValue_DeclaredToolchainInfo(
        toolchainType, execConstraints, targetConstraints, toolchainLabel);
  }
}
