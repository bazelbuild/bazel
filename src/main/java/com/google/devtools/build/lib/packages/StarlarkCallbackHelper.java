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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;

/**
 * A helper class for calling Starlark functions from Java, where the argument values are supplied
 * by the fields of a ClassObject, as in the case of computed attribute defaults and computed
 * implicit outputs.
 *
 * TODO(adonovan): eliminate the need for this class by making the Starlark calls in the same
 * Starlark thread that instantiated the rule.
 */
@AutoCodec
public class StarlarkCallbackHelper {

  private final StarlarkFunction callback;
  private final FuncallExpression ast;
  private final StarlarkSemantics starlarkSemantics;
  private final BazelStarlarkContext starlarkContext;

  public StarlarkCallbackHelper(
      StarlarkFunction callback,
      FuncallExpression ast,
      StarlarkSemantics starlarkSemantics,
      BazelStarlarkContext starlarkContext) {
    this.callback = callback;
    this.ast = ast;
    this.starlarkSemantics = starlarkSemantics;
    this.starlarkContext = starlarkContext;
  }

  public ImmutableList<String> getParameterNames() {
    return callback.getSignature().getSignature().getNames();
  }

  // TODO(adonovan): opt: all current callers are forced to construct a temporary ClassObject.
  // Instead, make them supply a map.
  public Object call(EventHandler eventHandler, ClassObject ctx, Object... arguments)
      throws EvalException, InterruptedException {
    try (Mutability mutability = Mutability.create("callback %s", callback)) {
      Environment env =
          Environment.builder(mutability)
              .setSemantics(starlarkSemantics)
              .setEventHandler(eventHandler)
              .setStarlarkContext(starlarkContext)
              .build();
      return callback.call(buildArgumentList(ctx, arguments), null, ast, env);
    } catch (ClassCastException | IllegalArgumentException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  /**
   * Creates a list of actual arguments that contains the given arguments and all attribute values
   * required from the specified context.
   */
  private ImmutableList<Object> buildArgumentList(ClassObject ctx, Object... arguments)
      throws EvalException {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    ImmutableList<String> names = getParameterNames();
    int requiredParameters = names.size() - arguments.length;
    for (int pos = 0; pos < requiredParameters; ++pos) {
      String name = names.get(pos);
      Object value = ctx.getValue(name);
      if (value == null) {
          throw new IllegalArgumentException(ctx.getErrorMessageForUnknownField(name));
      }
      builder.add(value);
    }
    return builder.add(arguments).build();
  }
}
