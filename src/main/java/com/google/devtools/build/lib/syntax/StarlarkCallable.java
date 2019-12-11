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
// TODO(adonovan): rename to just "Callable", since it's unambiguous.
public interface StarlarkCallable extends StarlarkValue {

  /**
   * Defines the implementation of function calling for a callable value.
   *
   * <p>Do not call this function directly. Use the {@link Starlark#call} function to make a call,
   * as it handles necessary book-keeping such as maintenance of the call stack, exception handling,
   * and so on.
   *
   * @param thread the StarlarkThread in which the function is called
   * @param call the function call expression (going away)
   * @param args positional arguments
   * @param kwargs named arguments
   */
  // TODO(adonovan): optimize the calling convention; see FUNCALL in Eval.java.
  Object callImpl(
      StarlarkThread thread,
      @Nullable FuncallExpression call, // TODO(adonovan): eliminate
      List<Object> args,
      Map<String, Object> kwargs)
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
