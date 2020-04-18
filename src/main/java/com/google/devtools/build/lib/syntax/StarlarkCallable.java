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

import com.google.common.collect.Maps;
import java.util.LinkedHashMap;

/**
 * The StarlarkCallable interface is implemented by all Starlark values that may be called from
 * Starlark like a function, including built-in functions and methods, Starlark functions, and
 * application-defined objects (such as rules, aspects, and providers in Bazel).
 *
 * <p>It defines two methods: {@code fastcall}, for performance, or {@code call} for convenience. By
 * default, {@code fastcall} delegates to {@code call}, and call throws an exception, so an
 * implementer may override either one.
 */
public interface StarlarkCallable extends StarlarkValue {

  /**
   * Defines the "convenient" implementation of function calling for a callable value.
   *
   * <p>Do not call this function directly. Use the {@link Starlark#call} function to make a call,
   * as it handles necessary book-keeping such as maintenance of the call stack, exception handling,
   * and so on.
   *
   * <p>The default implementation throws an exception.
   *
   * @param thread the StarlarkThread in which the function is called
   * @param args a tuple of the arguments passed by position
   * @param kwargs a new, mutable dict of the arguments passed by keyword. Iteration order is
   *     determined by keyword order in the call expression.
   */
  default Object call(StarlarkThread thread, Tuple<Object> args, Dict<String, Object> kwargs)
      throws EvalException, InterruptedException {
    throw Starlark.errorf("function %s not implemented", getName());
  }

  /**
   * Defines the "fast" implementation of function calling for a callable value.
   *
   * <p>Do not call this function directly. Use the {@link Starlark#call} or {@link
   * Starlark#fastcall} function to make a call, as it handles necessary book-keeping such as
   * maintenance of the call stack, exception handling, and so on.
   *
   * <p>This method defines the low-level or "fast" calling convention. A more convenient interface
   * is provided by the {@link #call} method, which provides a signature analogous to {@code def
   * f(*args, **kwargs)}, or possibly the "self-call" feature of the {@link
   * SkylarkCallable#selfCall} annotation mechanism. Implementations may elect to use {@code
   * Starlark.matchSignature} to assist with argument processing.
   *
   * <p>The default implementation forwards the call to {@code call}, after rejecting any duplicate
   * named arguments. Other implementations of this method should similarly reject duplicates.
   *
   * @param thread the StarlarkThread in which the function is called
   * @param positional a list of positional arguments
   * @param named a list of named arguments, as alternating Strings/Objects. May contain dups.
   */
  default Object fastcall(
      StarlarkThread thread,
      Object[] positional,
      Object[] named)
      throws EvalException, InterruptedException {
    LinkedHashMap<String, Object> kwargs = Maps.newLinkedHashMapWithExpectedSize(named.length >> 1);
    for (int i = 0; i < named.length; i += 2) {
      if (kwargs.put((String) named[i], named[i + 1]) != null) {
        throw Starlark.errorf("%s got multiple values for parameter '%s'", this, named[i]);
      }
    }
    return call(thread, Tuple.of(positional), Dict.wrap(thread.mutability(), kwargs));
  }

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
