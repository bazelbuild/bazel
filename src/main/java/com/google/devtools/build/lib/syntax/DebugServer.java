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

import java.util.function.Function;

/** A debug server interface, called from core skylark code. */
public interface DebugServer {

  /**
   * Executes the given callable and returns its result, while making any skylark evaluation visible
   * to the debugger. This method should be used to evaluate all debuggable Skylark code.
   *
   * @param env the Skylark execution environment
   * @param threadName the descriptive name of the thread
   * @param callable the callable object whose execution will be tracked
   * @param <T> the result type of the callable
   * @return the value returned by the callable
   */
  <T> T runWithDebugging(Environment env, String threadName, DebugCallable<T> callable)
      throws EvalException, InterruptedException;

  /** Shuts down the debug server, closing any open sockets, etc. */
  void close();

  /**
   * Returns a custom {@link Eval} supplier used to intercept statement execution to check for
   * breakpoints.
   */
  Function<Environment, Eval> evalOverride();

  /** Represents an invocation that will be tracked as a thread by the Skylark debug server. */
  interface DebugCallable<T> {

    /**
     * The invocation that will be tracked.
     *
     * @return the result
     */
    T call() throws EvalException, InterruptedException;
  }
}
