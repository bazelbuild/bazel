// Copyright 2017 The Bazel Authors. All rights reserved.
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
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Evaluation code for the Skylark AST. At the moment, it can execute only statements (and defers to
 * Expression.eval for evaluating expressions).
 */
public class Eval {
  protected final Environment env;
  private Object result = Runtime.NONE;

  public static Eval fromEnvironment(Environment env) {
    return evalSupplier.apply(env);
  }

  public static void setEvalSupplier(Function<Environment, Eval> evalSupplier) {
    Eval.evalSupplier = evalSupplier;
  }

  /** Reset Eval supplier to the default. */
  public static void removeCustomEval() {
    evalSupplier = Eval::new;
  }

  // TODO(bazel-team): remove this static state in favor of storing Eval instances in Environment
  private static Function<Environment, Eval> evalSupplier = Eval::new;

  /**
   * This constructor should never be called directly. Call {@link #fromEnvironment(Environment)}
   * instead.
   */
  protected Eval(Environment env) {
    this.env = env;
  }

  /** getResult returns the value returned by executing a ReturnStatement. */
  Object getResult() {
    return this.result;
  }

  void execAssignment(AssignmentStatement node) throws EvalException, InterruptedException {
    Object rvalue = node.getExpression().eval(env);
    node.getLValue().assign(rvalue, env, node.getLocation());
  }

  void execAugmentedAssignment(AugmentedAssignmentStatement node)
      throws EvalException, InterruptedException {
    node.getLValue()
        .assignAugmented(node.getOperator(), node.getExpression(), env, node.getLocation());
  }

  TokenKind execIfBranch(IfStatement.ConditionalStatements node)
      throws EvalException, InterruptedException {
    return execStatements(node.getStatements());
  }

  TokenKind execFor(ForStatement node) throws EvalException, InterruptedException {
    Object o = node.getCollection().eval(env);
    Iterable<?> col = EvalUtils.toIterable(o, node.getLocation(), env);
    EvalUtils.lock(o, node.getLocation());
    try {
      for (Object it : col) {
        node.getVariable().assign(it, env, node.getLocation());

        switch (execStatements(node.getBlock())) {
          case PASS:
          case CONTINUE:
            // Stay in loop.
            continue;
          case BREAK:
            // Finish loop, execute next statement after loop.
            return TokenKind.PASS;
          case RETURN:
            // Finish loop, return from function.
            return TokenKind.RETURN;
          default:
            throw new IllegalStateException("unreachable");
        }
      }
    } finally {
      EvalUtils.unlock(o, node.getLocation());
    }
    return TokenKind.PASS;
  }

  void execDef(FunctionDefStatement node) throws EvalException, InterruptedException {
    List<Expression> defaultExpressions = node.getSignature().getDefaultValues();
    ArrayList<Object> defaultValues = null;

    if (defaultExpressions != null) {
      defaultValues = new ArrayList<>(defaultExpressions.size());
      for (Expression expr : defaultExpressions) {
        defaultValues.add(expr.eval(env));
      }
    }

    // TODO(laurentlb): Could be moved to the Parser or the ValidationEnvironment?
    FunctionSignature sig = node.getSignature().getSignature();
    if (sig.getShape().getMandatoryNamedOnly() > 0) {
      throw new EvalException(node.getLocation(), "Keyword-only argument is forbidden.");
    }

    env.updateAndExport(
        node.getIdentifier().getName(),
        new UserDefinedFunction(
            node.getIdentifier().getName(),
            node.getIdentifier().getLocation(),
            FunctionSignature.WithValues.create(sig, defaultValues, /*types=*/ null),
            node.getStatements(),
            env.getGlobals()));
  }

  TokenKind execIf(IfStatement node) throws EvalException, InterruptedException {
    ImmutableList<IfStatement.ConditionalStatements> thenBlocks = node.getThenBlocks();
    // Avoid iterator overhead - most of the time there will be one or few "if"s.
    for (int i = 0; i < thenBlocks.size(); i++) {
      IfStatement.ConditionalStatements stmt = thenBlocks.get(i);
      if (EvalUtils.toBoolean(stmt.getCondition().eval(env))) {
        return exec(stmt);
      }
    }
    return execStatements(node.getElseBlock());
  }

  void execLoad(LoadStatement node) throws EvalException, InterruptedException {
    for (LoadStatement.Binding binding : node.getBindings()) {
      try {
        Identifier name = binding.getLocalName();
        Identifier declared = binding.getOriginalName();

        if (declared.isPrivate() && !node.mayLoadInternalSymbols()) {
          throw new EvalException(
              node.getLocation(),
              "symbol '" + declared.getName() + "' is private and cannot be imported.");
        }
        // The key is the original name that was used to define the symbol
        // in the loaded bzl file.
        env.importSymbol(node.getImport().getValue(), name, declared.getName());
      } catch (Environment.LoadFailedException e) {
        throw new EvalException(node.getLocation(), e.getMessage());
      }
    }
  }

  TokenKind execReturn(ReturnStatement node) throws EvalException, InterruptedException {
    Expression ret = node.getReturnExpression();
    if (ret != null) {
      this.result = ret.eval(env);
    }
    return TokenKind.RETURN;
  }

  /**
   * Execute the statement.
   *
   * @throws EvalException if execution of the statement could not be completed.
   * @throws InterruptedException may be thrown in a sub class.
   */
  protected TokenKind exec(Statement st) throws EvalException, InterruptedException {
    try {
      return execDispatch(st);
    } catch (EvalException ex) {
      throw st.maybeTransformException(ex);
    }
  }

  TokenKind execDispatch(Statement st) throws EvalException, InterruptedException {
    switch (st.kind()) {
      case ASSIGNMENT:
        execAssignment((AssignmentStatement) st);
        return TokenKind.PASS;
      case AUGMENTED_ASSIGNMENT:
        execAugmentedAssignment((AugmentedAssignmentStatement) st);
        return TokenKind.PASS;
      case CONDITIONAL:
        return execIfBranch((IfStatement.ConditionalStatements) st);
      case EXPRESSION:
        ((ExpressionStatement) st).getExpression().eval(env);
        return TokenKind.PASS;
      case FLOW:
        return ((FlowStatement) st).getKind();
      case FOR:
        return execFor((ForStatement) st);
      case FUNCTION_DEF:
        execDef((FunctionDefStatement) st);
        return TokenKind.PASS;
      case IF:
        return execIf((IfStatement) st);
      case LOAD:
        execLoad((LoadStatement) st);
        return TokenKind.PASS;
      case RETURN:
        return execReturn((ReturnStatement) st);
    }
    throw new IllegalArgumentException("unexpected statement: " + st.kind());
  }

  public TokenKind execStatements(ImmutableList<Statement> statements)
      throws EvalException, InterruptedException {
    // Hot code path, good chance of short lists which don't justify the iterator overhead.
    for (int i = 0; i < statements.size(); i++) {
      TokenKind flow = exec(statements.get(i));
      if (flow != TokenKind.PASS) {
        return flow;
      }
    }
    return TokenKind.PASS;
  }
}
