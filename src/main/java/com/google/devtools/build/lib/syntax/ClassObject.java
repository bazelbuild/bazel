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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableCollection;
import javax.annotation.Nullable;

/**
 * An interface for objects behaving like Skylark structs.
 */
// TODO(bazel-team): type checks
public interface ClassObject {

  /**
   * Returns the value associated with the name field in this struct,
   * or null if the field does not exist.
   */
  @Nullable
  Object getValue(String name);

  /**
   * Returns the fields of this struct.
   */
  ImmutableCollection<String> getKeys();

  /**
   * Returns a customized error message to print if the name is not a valid struct field
   * of this struct, or returns null to use the default error message.
   */
  @Nullable String errorMessage(String name);
}
