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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.Environment.LexicalFrame;

/**
 * The actual function registered in the environment. This function is defined in the parsed code
 * using {@link FunctionDefStatement}.
 */
public class UserDefinedFunction extends BaseFunction {

  private final ImmutableList<Statement> statements;

  // we close over the globals at the time of definition
  private final Environment.GlobalFrame definitionGlobals;

  public UserDefinedFunction(
      String name,
      Location location,
      FunctionSignature.WithValues<Object, SkylarkType> signature,
      ImmutableList<Statement> statements,
      Environment.GlobalFrame definitionGlobals) {
    super(name, signature, location);
    this.statements = statements;
    this.definitionGlobals = definitionGlobals;
  }

  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public Environment.GlobalFrame getDefinitionGlobals() {
    return definitionGlobals;
  }

  @Override
  public Object call(Object[] arguments, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    if (env.mutability().isFrozen()) {
      throw new EvalException(getLocation(), "Trying to call in frozen environment");
    }
    if (env.isRecursiveCall(this)) {
      throw new EvalException(getLocation(),
          String.format("Recursion was detected when calling '%s' from '%s'",
              getName(), env.getCurrentFunction().getName()));
    }

    ImmutableList<String> names = signature.getSignature().getNames();
    LexicalFrame lexicalFrame = LexicalFrame.create(env.mutability(), /*numArgs=*/ names.size());
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.STARLARK_USER_FN, getName())) {
      env.enterScope(this, lexicalFrame, ast, definitionGlobals);

      // Registering the functions's arguments as variables in the local Environment
      // foreach loop is not used to avoid iterator overhead
      for (int i = 0; i < names.size(); ++i) {
        env.update(names.get(i), arguments[i]);
      }

      Eval eval = Eval.fromEnvironment(env);
      try {
        for (Statement stmt : statements) {
          if (stmt instanceof ReturnStatement) {
            // Performance optimization.
            // Executing the statement would throw an exception, which is slow.
            Expression returnExpr = ((ReturnStatement) stmt).getReturnExpression();
            if (returnExpr == null) {
              return Runtime.NONE;
            }
            return returnExpr.eval(env);
          } else {
            eval.exec(stmt);
          }
        }
      } catch (ReturnStatement.ReturnException e) {
        return e.getValue();
      }
      return Runtime.NONE;
    } finally {
      env.exitScope();
    }
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    Label label = this.definitionGlobals.getLabel();

    printer.append("<function " + getName());
    if (label != null) {
      printer.append(" from " + label);
    }
    printer.append(">");
  }
}
