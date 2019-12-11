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

import com.google.devtools.build.lib.events.Location;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * The StarlarkCallable interface is implemented by all Starlark values that may be called from
 * Starlark like a function, including built-in functions and methods, Starlark functions, and
 * application-defined objects (such as rules, aspects, and providers in Bazel).
 */
public interface StarlarkCallable extends StarlarkValue {

  /**
   * Call this function with the given arguments.
   *
   * <p>Neither the callee nor the caller may modify the args List or kwargs Map.
   *
   * @param args the list of positional arguments
   * @param kwargs the mapping of named arguments
   * @param call the syntax tree of the function call
   * @param thread the StarlarkThread in which the function is called
   * @return the result of the call
   * @throws EvalException if there was an error invoking this function
   */
  // TODO(adonovan):
  // - make StarlarkThread the first parameter.
  // - eliminate the FuncallExpression parameter (which can be accessed through thread).
  Object call(
      List<Object> args,
      @Nullable Map<String, Object> kwargs,
      @Nullable FuncallExpression call,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  /** Returns the form this callable value should take in a stack trace. */
  String getName();

  /**
   * Returns the location of the definition of this callable value, or BUILTIN if it was not defined
   * in Starlark code.
   */
  default Location getLocation() {
    return Location.BUILTIN;
  }
}
