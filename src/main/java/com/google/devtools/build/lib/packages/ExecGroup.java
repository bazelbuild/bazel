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

package com.google.devtools.build.lib.packages;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.starlarkbuildapi.ExecGroupApi;
import java.util.Set;
import javax.annotation.Nullable;

/** Resolves the appropriate toolchains for the given parameters. */
@AutoValue
public abstract class ExecGroup implements ExecGroupApi {

  // This is intentionally a string that would fail {@code Identifier.isValid} so that
  // users can't create a group with the same name.
  public static final String DEFAULT_EXEC_GROUP_NAME = "default-exec-group";

  /** Create an exec group that inherits from the default exec group. */
  public static ExecGroup copyFromDefault() {
    return create(ImmutableSet.of(), ImmutableSet.of(), /* copyFrom= */ DEFAULT_EXEC_GROUP_NAME);
  }

  /** Create an exec group with the given toolchains and execution constraints. */
  public static ExecGroup create(Set<Label> requiredToolchains, Set<Label> execCompatibleWith) {
    return create(requiredToolchains, execCompatibleWith, /* copyFrom= */ null);
  }

  private static ExecGroup create(
      Set<Label> requiredToolchains, Set<Label> execCompatibleWith, @Nullable String copyFrom) {
    return new AutoValue_ExecGroup(
        ImmutableSet.copyOf(requiredToolchains), ImmutableSet.copyOf(execCompatibleWith), copyFrom);
  }

  public abstract ImmutableSet<Label> requiredToolchains();

  public abstract ImmutableSet<Label> execCompatibleWith();

  @Nullable
  public abstract String copyFrom();

  /** Creates a new exec group that inherits from the given group. */
  public ExecGroup inheritFrom(ExecGroup other) {
    ImmutableSet<Label> requiredToolchains =
        new ImmutableSet.Builder<Label>()
            .addAll(this.requiredToolchains())
            .addAll(other.requiredToolchains())
            .build();
    ImmutableSet<Label> execCompatibleWith =
        new ImmutableSet.Builder<Label>()
            .addAll(this.execCompatibleWith())
            .addAll(other.execCompatibleWith())
            .build();
    return create(requiredToolchains, execCompatibleWith);
  }
}
