// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import javax.annotation.Nullable;

/**
 * Instances of {@link MakeVariableSupplier} passed to {@link ConfigurationMakeVariableContext} will
 * be called before getting value from {@link ConfigurationMakeVariableContext} itself.
 */
public interface MakeVariableSupplier {

  /** Returns Make variable value or null if value is not supplied. */
  @Nullable
  String getMakeVariable(String variableName);

  /** Returns all Make variables that it supplies */
  ImmutableMap<String, String> getAllMakeVariables();

  /** {@link MakeVariableSupplier} that reads variables it supplies from a map. */
  class MapBackedMakeVariableSupplier implements MakeVariableSupplier {

    private final ImmutableMap<String, String> makeVariables;

    public MapBackedMakeVariableSupplier(ImmutableMap<String, String> makeVariables) {
      this.makeVariables = Preconditions.checkNotNull(makeVariables);
    }

    @Nullable
    @Override
    public String getMakeVariable(String variableName) {
      return makeVariables.get(variableName);
    }

    @Override
    public ImmutableMap<String, String> getAllMakeVariables() {
      return makeVariables;
    }
  }
}
