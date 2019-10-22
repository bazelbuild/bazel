// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import java.util.Objects;

public class NinjaRule {
  private final ImmutableSortedMap<ParameterName, String> parameters;

  public NinjaRule(ImmutableSortedMap<ParameterName, String> parameters) {
    this.parameters = parameters;
  }

  public ImmutableSortedMap<ParameterName, String> getParameters() {
    return parameters;
  }

  public String getName() {
    return Preconditions.checkNotNull(parameters.get(ParameterName.name));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    NinjaRule ninjaRule = (NinjaRule) o;
    return parameters.equals(ninjaRule.parameters);
  }

  @Override
  public int hashCode() {
    return Objects.hash(parameters);
  }

  public enum ParameterName {
    name,
    command,
    depfile,
    deps,
    msvc_deps_prefix,
    description,
    generator,
    restat,
    rspfile,
    rspfile_content,
    pool,

    // These variables are provided by the target.
    in,
    in_newline,
    out;

    private final boolean definedByTarget;

    ParameterName() {
      definedByTarget = false;
    }

    ParameterName(boolean definedByTarget) {
      this.definedByTarget = definedByTarget;
    }

    public static ParameterName nullOrValue(String name) {
      try {
        return valueOf(name);
      } catch (IllegalArgumentException e) {
        return null;
      }
    }

    public boolean isDefinedByTarget() {
      return definedByTarget;
    }
  }
}
