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

  /** Builder class to assist in creating {@link DeclaredToolchainInfo} instances. */
  public static class Builder {
    private ToolchainTypeInfo toolchainType;
    private ConstraintCollection.Builder execConstraints = ConstraintCollection.builder();
    private ConstraintCollection.Builder targetConstraints = ConstraintCollection.builder();
    private Label toolchainLabel;

    /** Sets the type of the toolchain being declared. */
    public Builder toolchainType(ToolchainTypeInfo toolchainType) {
      this.toolchainType = toolchainType;
      return this;
    }

    /** Adds constraints describing the execution environment. */
    public Builder addExecConstraints(Iterable<ConstraintValueInfo> constraints) {
      this.execConstraints.addConstraints(constraints);
      return this;
    }

    /** Adds constraints describing the execution environment. */
    public Builder addExecConstraints(ConstraintValueInfo... constraints) {
      return addExecConstraints(ImmutableList.copyOf(constraints));
    }

    /** Adds constraints describing the target environment. */
    public Builder addTargetConstraints(Iterable<ConstraintValueInfo> constraints) {
      this.targetConstraints.addConstraints(constraints);
      return this;
    }

    /** Adds constraints describing the target environment. */
    public Builder addTargetConstraints(ConstraintValueInfo... constraints) {
      return addTargetConstraints(ImmutableList.copyOf(constraints));
    }

    /** Sets the label of the toolchain to resolve for use in toolchain-aware rules. */
    public Builder toolchainLabel(Label toolchainLabel) {
      this.toolchainLabel = toolchainLabel;
      return this;
    }

    /** Returns the newly created {@link DeclaredToolchainInfo} instance. */
    public DeclaredToolchainInfo build() {
      // TODO(katre): handle constraint duplication in attributes separately.
      return new AutoValue_DeclaredToolchainInfo(
          toolchainType, execConstraints.build(), targetConstraints.build(), toolchainLabel);
    }
  }

  /** Returns a new {@link Builder} for creating {@link DeclaredToolchainInfo} instances. */
  public static Builder builder() {
    return new Builder();
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
