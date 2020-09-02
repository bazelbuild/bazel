// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery.Code;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FileOptions;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInput;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.SyntaxError;
import java.io.OutputStream;

/**
 * Starlark output formatter for cquery results. Each configured target will result in an evaluation
 * of the Starlark expression specified by {@code --expr}.
 */
public class StarlarkOutputFormatterCallback extends CqueryThreadsafeCallback {
  private static final Object[] NO_ARGS = new Object[0];

  // Starlark function with single required parameter "target", a ConfiguredTarget query result.
  private final StarlarkFunction exprEvalFn;

  StarlarkOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor)
      throws QueryException, InterruptedException {
    super(eventHandler, options, out, skyframeExecutor, accessor);

    // Validate that options.expr is a pure expression (for example, that it does not attempt
    // to escape its scope via unbalanced parens).
    ParserInput exprParserInput = ParserInput.fromString(options.expr, "--starlark:expr");
    try {
      Expression.parse(exprParserInput);
    } catch (SyntaxError.Exception ex) {
      throw new QueryException(
          "invalid --starlark:expr: " + ex.getMessage(), Code.STARLARK_SYNTAX_ERROR);
    }

    // Create a synthetic file that defines a function with single parameter "target",
    // whose body is provided by the user's expression.
    String fileBody = "def f(target): return (" + options.expr + ")\n" + "f";
    ParserInput input = ParserInput.fromString(fileBody, "--starlark:expr");

    try (Mutability mu = Mutability.create("formatter")) {
      StarlarkThread thread = new StarlarkThread(mu, StarlarkSemantics.DEFAULT);
      this.exprEvalFn =
          (StarlarkFunction) Starlark.execFile(input, FileOptions.DEFAULT, Module.create(), thread);
    } catch (SyntaxError.Exception ex) {
      throw new QueryException(
          "invalid --starlark:expr: " + ex.getMessage(), Code.STARLARK_SYNTAX_ERROR);
    } catch (EvalException ex) {
      throw new QueryException(
          "invalid --starlark:expr: " + ex.getMessageWithStack(), Code.STARLARK_EVAL_ERROR);
    }
  }

  @Override
  public String getName() {
    return "starlark";
  }

  @Override
  public void processOutput(Iterable<ConfiguredTarget> partialResult) throws InterruptedException {
    StarlarkThread thread =
        new StarlarkThread(Mutability.create("cquery evaluation"), StarlarkSemantics.DEFAULT);
    thread.setMaxExecutionSteps(500_000L);

    for (ConfiguredTarget target : partialResult) {
      try {
        // Invoke exprEvalFn with `target` argument.
        Object result = Starlark.fastcall(thread, this.exprEvalFn, new Object[] {target}, NO_ARGS);

        addResult(Starlark.str(result));
      } catch (EvalException ex) {
        eventHandler.handle(
            Event.error(
                String.format(
                    "Starlark evaluation error for %s: %s",
                    target.getLabel(), ex.getMessageWithStack())));
        continue;
      }
    }
  }
}
