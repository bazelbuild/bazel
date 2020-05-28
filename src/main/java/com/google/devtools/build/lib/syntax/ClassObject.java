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

/** An interface for Starlark values (such as structs) that have fields. */
// TODO(adonovan): rename "HasFields".
public interface ClassObject extends StarlarkValue {

  /**
   * Returns the value of the field with the given name, or null if the field does not exist. The
   * interpreter (Starlark code) calls the getValue below, which has access to StarlarkSemantics.
   *
   * <p>The set of names for which {@code getValue} returns non-null should match {@code
   * getFieldNames} if possible.
   *
   * @throws EvalException if a user-visible error occurs (other than non-existent field).
   */
  // TODO(adonovan): rename "getField".
  @Nullable
  Object getValue(String name) throws EvalException;

  /**
   * Returns the value of the field with the given name, or null if the field does not exist. The
   * interpreter (Starlark code) calls this getValue, but client code cannot be relied upon to do
   * so, so any checks done on the semantics are incompletely enforced.
   *
   * @param semantics the Starlark semantics, which determine the available fields
   * @param name the name of the field to retrieve
   * @throws EvalException if the field exists but could not be retrieved
   */
  @Nullable
  default Object getValue(StarlarkSemantics semantics, String name) throws EvalException {
    return this.getValue(name);
  }

  /**
   * Returns the names of this value's fields, in some undefined but stable order.
   *
   * <p>A call to {@code getValue} for each of these names should return non-null, though this is
   * not enforced.
   *
   * <p>The Starlark expression {@code dir(x)} reports the union of {@code getFieldNames()} and any
   * StarlarkMethod-annotated fields and methods of this value.
   */
  ImmutableCollection<String> getFieldNames();

  /**
   * Returns the error message to print for an attempt to access an undefined field.
   *
   * <p>May return null to use a default error message.
   */
  @Nullable
  String getErrorMessageForUnknownField(String field);
}
