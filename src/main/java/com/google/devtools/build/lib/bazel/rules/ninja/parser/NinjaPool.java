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
//

package com.google.devtools.build.lib.bazel.rules.ninja.parser;


import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import javax.annotation.concurrent.Immutable;

/**
 * Ninja pool representation. A Ninja pool allocates rules to use a finite number of concurrent jobs
 * instead of the default parallelism computed by Ninja.
 *
 * <p>While there is no current logical mapping between pool depths and execution phase resource
 * scheduling, this exists for the completeness of the {@link NinjaParser}, and bringing pool
 * objects into {@link NinjaScope}.
 *
 * <p>{@link NinjaVariableValue} to be replaced for each target according to the scope rules.
 *
 * <p>See <a href="https://ninja-build.org/manual.html#ref_pool">Ninja docs</a> for more info.
 */
@Immutable
public final class NinjaPool {
  private final String name;
  private final Integer depth;

  public NinjaPool(String name, ImmutableSortedMap<NinjaPoolVariable, NinjaVariableValue> variables)
      throws GenericParsingException {
    this.name = name;
    this.depth = validateDepth(variables.get(NinjaPoolVariable.DEPTH));
  }

  /** Returns name of the ninja pool. */
  public String getName() {
    return name;
  }

  /** Returns depth of the ninja pool, or the maximum concurrent number of jobs in this pool. */
  public Integer getDepth() {
    return depth;
  }

  private static Integer validateDepth(NinjaVariableValue value) throws GenericParsingException {
    String rawValue = value.getRawText();
    try {
      return Integer.parseInt(rawValue);
    } catch (NumberFormatException e) {
      throw new GenericParsingException(
          String.format("Expected an integer for the 'depth' value, but got '%s'.", rawValue), e);
    }
  }
}
