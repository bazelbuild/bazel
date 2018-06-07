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

import com.google.devtools.build.lib.syntax.DebugServer.DebugCallable;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * A helper class for enabling/disabling skylark debugging.
 *
 * <p>{@code runWithDebuggingIfEnabled} must be called after {@code initializeDebugServer}, and
 * before {@code disableDebugging}.
 */
public final class DebugServerUtils {

  private DebugServerUtils() {}

  private static final AtomicReference<DebugServer> instance = new AtomicReference<>();

  /**
   * Called at the start of a debuggable skylark session to enable debugging. The custom {@link
   * Eval} supplier provided should intercept statement execution to check for breakpoints.
   */
  public static void initializeDebugServer(DebugServer server) {
    instance.set(server);
    Eval.setEvalSupplier(server.evalOverride());
  }

  /**
   * Called at the end of a debuggable skylark session to shut down the debug server and disable
   * debugging.
   */
  public static void disableDebugging() {
    DebugServer server = instance.getAndSet(null);
    if (server != null) {
      server.close();
    }
    Eval.removeCustomEval();
  }

  /**
   * Tracks the execution of the given callable object in the debug server.
   *
   * <p>If the skylark debugger is not enabled, runs {@code callable} directly.
   *
   * @param env the Skylark execution environment
   * @param threadName the descriptive name of the thread
   * @param callable the callable object whose execution will be tracked
   * @param <T> the result type of the callable
   * @return the value returned by the callable
   */
  public static <T> T runWithDebuggingIfEnabled(
      Environment env, Supplier<String> threadName, DebugCallable<T> callable)
      throws EvalException, InterruptedException {
    DebugServer server = instance.get();
    return server != null
        ? server.runWithDebugging(env, threadName.get(), callable)
        : callable.call();
  }
}
