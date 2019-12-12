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

/** A StarlarkFunction is the function value created by a Starlark {@code def} statement. */
public final class StarlarkFunction extends BaseFunction {

  private final String name;
  private final Location location;
  private final ImmutableList<Statement> statements;
  private final Module module; // a function closes over its defining module
  private final Tuple<Object> defaultValues;

  // TODO(adonovan): make this private. The CodecTests should go through interpreter to instantiate
  // such things.
  public StarlarkFunction(
      String name,
      Location location,
      FunctionSignature signature,
      Tuple<Object> defaultValues,
      ImmutableList<Statement> statements,
      Module module) {
    super(signature);
    this.name = name;
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
  public Location getLocation() {
    return location;
  }

  @Override
  public String getName() {
    return name;
  }

  /** @deprecated Do not assume function values are represented as syntax trees. */
  // TODO(adonovan): the only non-test use is to obtain the function's doc string. Add API for that.
  @Deprecated
  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public Module getModule() {
    return module;
  }

  @Override
  protected Object call(Object[] arguments, FuncallExpression ast, StarlarkThread thread)
      throws EvalException, InterruptedException {
    if (thread.mutability().isFrozen()) {
      throw new EvalException(null, "Trying to call in frozen environment");
    }
    if (thread.isRecursiveCall(this)) {
      throw new EvalException(null, String.format("function '%s' called recursively", name));
    }

    // Registering the functions's arguments as variables in the local StarlarkThread
    // foreach loop is not used to avoid iterator overhead
    ImmutableList<String> names = getSignature().getParameterNames();
    for (int i = 0; i < names.size(); ++i) {
      thread.update(names.get(i), arguments[i]);
    }

    return Eval.execStatements(thread, statements);
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
