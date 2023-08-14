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

import com.google.auto.value.AutoOneOf;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker.IncompatibleTargetException;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper.ValidationException;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainException;

/** Tagged union of errors that can be encountered when creating the {@link DependencyContext}. */
@AutoOneOf(DependencyContextError.Kind.class)
public abstract class DependencyContextError {
  /** Tags for errors types that may occur. */
  public enum Kind {
    TOOLCHAIN,
    CONFIGURED_VALUE_CREATION,
    INCOMPATIBLE_TARGET,
    VALIDATION
  }

  public abstract Kind kind();

  public abstract ToolchainException toolchain();

  public abstract ConfiguredValueCreationException configuredValueCreation();

  /** This error is only possible for {@link DependencyContextProducerWithCompatibilityCheck}. */
  public abstract IncompatibleTargetException incompatibleTarget();

  /** This error is only possible for {@link DependencyContextProducerWithCompatibilityCheck}. */
  public abstract ValidationException validation();

  static DependencyContextError of(ToolchainException error) {
    return AutoOneOf_DependencyContextError.toolchain(error);
  }

  static DependencyContextError of(ConfiguredValueCreationException error) {
    return AutoOneOf_DependencyContextError.configuredValueCreation(error);
  }

  static DependencyContextError of(IncompatibleTargetException error) {
    return AutoOneOf_DependencyContextError.incompatibleTarget(error);
  }

  static DependencyContextError of(ValidationException error) {
    return AutoOneOf_DependencyContextError.validation(error);
  }
}
