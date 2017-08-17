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
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Syntax node for a function definition.
 */
public final class FunctionDefStatement extends Statement {

  private final Identifier identifier;
  private final FunctionSignature.WithValues<Expression, Expression> signature;
  private final ImmutableList<Statement> statements;
  private final ImmutableList<Parameter<Expression, Expression>> parameters;

  public FunctionDefStatement(Identifier identifier,
      Iterable<Parameter<Expression, Expression>> parameters,
      FunctionSignature.WithValues<Expression, Expression> signature,
      Iterable<Statement> statements) {
    this.identifier = identifier;
    this.parameters = ImmutableList.copyOf(parameters);
    this.signature = signature;
    this.statements = ImmutableList.copyOf(statements);
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    List<Expression> defaultExpressions = signature.getDefaultValues();
    ArrayList<Object> defaultValues = null;

    if (defaultExpressions != null) {
      defaultValues = new ArrayList<>(defaultExpressions.size());
      for (Expression expr : defaultExpressions) {
        defaultValues.add(expr.eval(env));
      }
    }

    FunctionSignature sig = signature.getSignature();
    if (env.getSemantics().incompatibleDisallowKeywordOnlyArgs
        && sig.getShape().getMandatoryNamedOnly() > 0) {
      throw new EvalException(
          getLocation(),
          "Keyword-only argument is forbidden. You can temporarily disable this "
              + "error using the flag --incompatible_disallow_keyword_only_args=false");
    }

    env.update(
        identifier.getName(),
        new UserDefinedFunction(
            identifier,
            FunctionSignature.WithValues.create(sig, defaultValues, /*types=*/null),
            statements,
            env.getGlobals()));
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    buffer.append("def ");
    identifier.prettyPrint(buffer);
    buffer.append('(');
    String sep = "";
    for (Parameter<?, ?> param : parameters) {
      buffer.append(sep);
      param.prettyPrint(buffer);
      sep = ", ";
    }
    buffer.append("):\n");
    printSuite(buffer, statements, indentLevel);
  }

  @Override
  public String toString() {
    return "def " + identifier + "(" + signature + "): ...\n";
  }

  public Identifier getIdentifier() {
    return identifier;
  }

  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public ImmutableList<Parameter<Expression, Expression>> getParameters() {
    return parameters;
  }

  public FunctionSignature.WithValues<Expression, Expression> getSignature() {
    return signature;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }
}
