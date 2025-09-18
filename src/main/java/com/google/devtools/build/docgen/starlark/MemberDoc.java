// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.docgen.starlark;

import com.google.common.collect.ImmutableList;

/**
 * Documentation for a method or struct field of a Java class annotated with {@link
 * net.starlark.java.annot.StarlarkBuiltin}, or for a field of a Starlark-defined struct.
 */
public abstract class MemberDoc extends StarlarkDoc {

  protected MemberDoc(StarlarkDocExpander expander) {
    super(expander);
  }

  /** Returns whether the value is documented. */
  public abstract boolean documented();

  /**
   * Returns whether the value can be called as a function.
   *
   * <p>For example, {@code ctx.label} is not callable.
   */
  public abstract boolean isCallable();

  /**
   * For a callable value, returns the name for the return type; or the name of the value's own type
   * otherwise.
   */
  public abstract String getReturnType();

  /**
   * For a callable value, returns a string containing additional documentation about the return
   * value.
   *
   * <p>Returns an empty string by default.
   */
  public String getReturnTypeExtraMessage() {
    return "";
  }

  /** Returns true if the value is callable and is a constructor of its type. */
  public boolean isConstructor() {
    return false;
  }

  /**
   * Returns the value's name within its module.
   *
   * <p>In most cases, this is the same as {@link #getName}. The exception is for overloaded methods
   * in a {@link net.starlark.java.annot.StarlarkBuiltin}-annotated Java class. In that case, this
   * method would return the name of the method, while {@link #getName} would return the method
   * signature with parameters, e.g. {@code method_name(arg1, arg2)}.
   */
  public String getShortName() {
    return getName();
  }

  /**
   * For a callable value, returns a list containing the documentation for each of the method's
   * parameters; or an empty list otherwise.
   */
  public abstract ImmutableList<? extends ParamDoc> getParams();

  /**
   * For a callable value, returns the string representation of the parameters, for example {@code
   * "arg1, arg2=None, **kwargs"}; or an empty string otherwise.
   */
  protected abstract String getParameterString();

  /**
   * For a callable value, returns the string representing the method signature of the Starlark
   * method, which contains HTML links to the documentation of parameter types if available. For a
   * non-callable value, returns the string representation of the value's type (with HTML links to
   * the type's documentation, if available) and name.
   */
  public abstract String getSignature();
}
