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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkFunction;

/**
 * Represents a symbolic macro, defined in a .bzl file, that may be instantiated during Package
 * evaluation.
 *
 * <p>This is analogous to {@link RuleClass}. In essence, a {@code MacroClass} consists of the
 * macro's schema and its implementation function.
 */
public final class MacroClass {

  private final String name;
  private final StarlarkFunction implementation;

  public MacroClass(String name, StarlarkFunction implementation) {
    this.name = name;
    this.implementation = implementation;
  }

  /** Returns the macro's exported name. */
  public String getName() {
    return name;
  }

  public StarlarkFunction getImplementation() {
    return implementation;
  }

  /** Builder for {@link MacroClass}. */
  public static final class Builder {
    private final StarlarkFunction implementation;
    @Nullable private String name = null;

    public Builder(StarlarkFunction implementation) {
      this.implementation = implementation;
    }

    @CanIgnoreReturnValue
    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    public MacroClass build() {
      Preconditions.checkNotNull(name);
      return new MacroClass(name, implementation);
    }
  }
}
