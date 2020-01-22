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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import javax.annotation.Nullable;

/** A StarlarkFunction is the function value created by a Starlark {@code def} statement. */
public final class StarlarkFunction extends BaseFunction {

  private final String name;
  private final FunctionSignature signature;
  private final Location location;
  private final ImmutableList<Statement> statements;
  private final Module module; // a function closes over its defining module
  private final Tuple<Object> defaultValues;

  // isToplevel indicates that this is the <toplevel> function containing
  // top-level statements of a file. It causes assignments to unresolved
  // identifiers to update the module, not the lexical frame.
  // TODO(adonovan): remove this hack when identifier resolution is accurate.
  boolean isToplevel;

  // TODO(adonovan): make this private. The CodecTests should go through interpreter to instantiate
  // such things.
  public StarlarkFunction(
      String name,
      Location location,
      FunctionSignature signature,
      Tuple<Object> defaultValues,
      ImmutableList<Statement> statements,
      Module module) {
    this.name = name;
    this.signature = signature;
    this.location = location;
    this.statements = statements;
    this.module = module;
    this.defaultValues = defaultValues;
  }

  @Override
  public Tuple<Object> getDefaultValues() {
    return defaultValues;
  }

  @Override
  public FunctionSignature getSignature() {
    return signature;
  }

  @Override
  public Location getLocation() {
    return location;
  }

  @Override
  public String getName() {
    return name;
  }

  /** Returns the value denoted by the function's doc string literal, or null if absent. */
  @Nullable
  public String getDocumentation() {
    if (statements.isEmpty()) {
      return null;
    }
    Statement first = statements.get(0);
    if (!(first instanceof ExpressionStatement)) {
      return null;
    }
    Expression expr = ((ExpressionStatement) first).getExpression();
    if (!(expr instanceof StringLiteral)) {
      return null;
    }
    return ((StringLiteral) expr).getValue();
  }

  public Module getModule() {
    return module;
  }

  @Override
  public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named)
      throws EvalException, InterruptedException {
    if (thread.mutability().isFrozen()) {
      throw Starlark.errorf("Trying to call in frozen environment");
    }
    if (thread.isRecursiveCall(this)) {
      throw Starlark.errorf("function '%s' called recursively", name);
    }

    // Compute the effective parameter values
    // and update the corresponding variables.
    Object[] arguments =
        Starlark.matchSignature(
            getSignature(), this, getDefaultValues(), thread.mutability(), positional, named);

    StarlarkThread.CallFrame fr = thread.frame(0);
    ImmutableList<String> names = getSignature().getParameterNames();
    for (int i = 0; i < names.size(); ++i) {
      fr.locals.put(names.get(i), arguments[i]);
    }

    return Eval.execFunctionBody(fr, statements);
  }

  @Override
  public void repr(Printer printer) {
    Object label = module.getLabel();

    printer.append("<function " + getName());
    if (label != null) {
      printer.append(" from " + label);
    }
    printer.append(">");
  }
}
