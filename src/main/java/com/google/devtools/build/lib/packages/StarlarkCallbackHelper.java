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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;

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

  // These fields, parts of the state of the loading-phase
  // thread that instantiated a rule, must be propagated to
  // the child threads (implicit outputs, attribute defaults).
  // This includes any other thread-local state, such as
  // the Label.HasRepoMapping or PackageFactory.PackageContext.
  // TODO(adonovan): it would be cleaner and less error prone to
  // perform these callbacks in the actual loading-phase thread,
  // at the end of BUILD file execution.
  // Alternatively (or additionally), we could put PackageContext
  // into BazelStarlarkContext so there's only a single blob of state.
  private final StarlarkSemantics starlarkSemantics;
  private final BazelStarlarkContext context;

  public StarlarkCallbackHelper(
      StarlarkFunction callback,
      FuncallExpression ast,
      StarlarkSemantics starlarkSemantics,
      BazelStarlarkContext context) {
    this.callback = callback;
    this.ast = ast;
    this.starlarkSemantics = starlarkSemantics;
    this.context = context;
  }

  public ImmutableList<String> getParameterNames() {
    return callback.getSignature().getParameterNames();
  }

  // TODO(adonovan): opt: all current callers are forced to construct a temporary ClassObject.
  // Instead, make them supply a map.
  public Object call(EventHandler eventHandler, ClassObject ctx, Object... arguments)
      throws EvalException, InterruptedException {
    try (Mutability mutability = Mutability.create("callback", callback)) {
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .setSemantics(starlarkSemantics)
              .setEventHandler(eventHandler)
              .build();
      context.storeInThread(thread);
      return Starlark.call(
          thread, callback, ast, buildArgumentList(ctx, arguments), /*kwargs=*/ ImmutableMap.of());
    } catch (ClassCastException | IllegalArgumentException e) { // TODO(adonovan): investigate
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
