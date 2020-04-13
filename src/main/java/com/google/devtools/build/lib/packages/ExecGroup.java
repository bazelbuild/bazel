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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkbuildapi.ExecGroupApi;
import java.util.Set;

/**
 * Resolves the appropriate toolchains for the given parameters.
 *
 * <p>Currently just a stub.
 */
public class ExecGroup implements ExecGroupApi {
  private final ImmutableSet<Label> requiredToolchains;
  private final ImmutableSet<Label> execCompatibleWith;

  public ExecGroup(Set<Label> requiredToolchains, Set<Label> execCompatibleWith) {
    this.requiredToolchains = ImmutableSet.copyOf(requiredToolchains);
    this.execCompatibleWith = ImmutableSet.copyOf(execCompatibleWith);
  }

  public ImmutableSet<Label> getRequiredToolchains() {
    return requiredToolchains;
  }

  public ImmutableSet<Label> getExecutionPlatformConstraints() {
    return execCompatibleWith;
  }
}
