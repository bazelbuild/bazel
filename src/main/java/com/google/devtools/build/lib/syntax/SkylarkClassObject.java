// Copyright 2018 The Bazel Authors. All rights reserved.
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

import javax.annotation.Nullable;

/** A variant of ClassObject for implementations that require a StarlarkSemantics. */
public interface SkylarkClassObject extends ClassObject {

  /**
   * Returns the value of the field with the given name, or null if the field does not exist.
   *
   * @param semantics the Starlark semantics, which determine the available fields
   * @param name the name of the field to retrieve
   * @throws EvalException if the field exists but could not be retrieved
   */
  @Nullable
  Object getValue(StarlarkSemantics semantics, String name) throws EvalException;
}
