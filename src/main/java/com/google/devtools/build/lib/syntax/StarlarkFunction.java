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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.StarlarkThread.LexicalFrame;

/** A StarlarkFunction is the function value created by a Starlark {@code def} statement. */
public class StarlarkFunction extends BaseFunction {

  private final ImmutableList<Statement> statements;

  // we close over the globals at the time of definition
  private final StarlarkThread.GlobalFrame definitionGlobals;
  private final ImmutableMap<String, Integer> nameToIndex;
  private final int[] parameterNameIndices;

  public StarlarkFunction(
      String name,
      Location location,
      FunctionSignature.WithValues<Object, SkylarkType> signature,
      ImmutableList<Statement> statements,
      StarlarkThread.GlobalFrame definitionGlobals,
      ImmutableMap<String, Integer> nameToIndex) {
    super(name, signature, location);
    this.statements = statements;
    this.definitionGlobals = definitionGlobals;
    this.nameToIndex = nameToIndex;
    this.parameterNameIndices = signature.getSignature().getNames().stream()
        .mapToInt(nameToIndex::get).toArray();
  }

  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public StarlarkThread.GlobalFrame getDefinitionGlobals() {
    return definitionGlobals;
  }

  @Override
  public Object call(Object[] arguments, FuncallExpression ast, StarlarkThread thread)
      throws EvalException, InterruptedException {
    if (thread.mutability().isFrozen()) {
      throw new EvalException(getLocation(), "Trying to call in frozen environment");
    }
    if (thread.isRecursiveCall(this)) {
      throw new EvalException(
          getLocation(),
          String.format(
              "Recursion was detected when calling '%s' from '%s'",
              getName(), thread.getCurrentFunction().getName()));
    }

    ImmutableList<String> names = signature.getSignature().getNames();
    LexicalFrame lexicalFrame = LexicalFrame.createMutable(thread.mutability(), nameToIndex);
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.STARLARK_USER_FN, getName())) {
      thread.enterScope(this, lexicalFrame, ast, definitionGlobals);

      // Registering the functions's arguments as variables in the local StarlarkThread
      // foreach loop is not used to avoid iterator overhead
      for (int i = 0; i < names.size(); ++i) {
        thread.updateByIndex(names.get(i), parameterNameIndices[i], arguments[i]);
      }

      return Eval.execStatements(thread, statements);
    } finally {
      thread.exitScope();
    }
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    Object label = this.definitionGlobals.getLabel();

    printer.append("<function " + getName());
    if (label != null) {
      printer.append(" from " + label);
    }
    printer.append(">");
  }
}
