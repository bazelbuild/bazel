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
import java.util.Map;

/**
 * Evaluation code for the Skylark AST. At the moment, it can execute only statements (and defers to
 * Expression.eval for evaluating expressions).
 */
public class Eval {
  private final Environment env;

  /** An exception that signals changes in the control flow (e.g. break or continue) */
  private static class FlowException extends EvalException {
    FlowException(String message) {
      super(null, message);
    }

    @Override
    public boolean canBeAddedToStackTrace() {
      return false;
    }
  }

  private static final FlowException breakException = new FlowException("FlowException - break");
  private static final FlowException continueException =
      new FlowException("FlowException - continue");

  public Eval(Environment env) {
    this.env = env;
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

  void execIfBranch(IfStatement.ConditionalStatements node)
      throws EvalException, InterruptedException {
    execStatements(node.getStatements());
  }

  void execFor(ForStatement node) throws EvalException, InterruptedException {
    Object o = node.getCollection().eval(env);
    Iterable<?> col = EvalUtils.toIterable(o, node.getLocation(), env);
    EvalUtils.lock(o, node.getLocation());
    try {
      for (Object it : col) {
        node.getVariable().assign(it, env, node.getLocation());

        try {
          execStatements(node.getBlock());
        } catch (FlowException ex) {
          if (ex == breakException) {
            return;
          }
        }
      }
    } finally {
      EvalUtils.unlock(o, node.getLocation());
    }
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

    FunctionSignature sig = node.getSignature().getSignature();
    if (env.getSemantics().incompatibleDisallowKeywordOnlyArgs()
        && sig.getShape().getMandatoryNamedOnly() > 0) {
      throw new EvalException(
          node.getLocation(),
          "Keyword-only argument is forbidden. You can temporarily disable this "
              + "error using the flag --incompatible_disallow_keyword_only_args=false");
    }

    env.update(
        node.getIdentifier().getName(),
        new UserDefinedFunction(
            node.getIdentifier().getName(),
            node.getIdentifier().getLocation(),
            FunctionSignature.WithValues.create(sig, defaultValues, /*types=*/ null),
            node.getStatements(),
            env.getGlobals()));
  }

  void execIf(IfStatement node) throws EvalException, InterruptedException {
    for (IfStatement.ConditionalStatements stmt : node.getThenBlocks()) {
      if (EvalUtils.toBoolean(stmt.getCondition().eval(env))) {
        exec(stmt);
        return;
      }
    }
    execStatements(node.getElseBlock());
  }

  void execLoad(LoadStatement node) throws EvalException, InterruptedException {
    if (env.getSemantics().incompatibleLoadArgumentIsLabel()) {
      String s = node.getImport().getValue();
      if (!s.startsWith("//") && !s.startsWith(":") && !s.startsWith("@")) {
        throw new EvalException(
            node.getLocation(),
            "First argument of 'load' must be a label and start with either '//', ':', or '@'. "
                + "Use --incompatible_load_argument_is_label=false to temporarily disable this "
                + "check.");
      }
    }

    for (Map.Entry<Identifier, String> entry : node.getSymbolMap().entrySet()) {
      try {
        Identifier name = entry.getKey();
        Identifier declared = new Identifier(entry.getValue());

        if (declared.isPrivate()) {
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

  void execReturn(ReturnStatement node) throws EvalException, InterruptedException {
    Expression ret = node.getReturnExpression();
    if (ret == null) {
      throw new ReturnStatement.ReturnException(node.getLocation(), Runtime.NONE);
    }
    throw new ReturnStatement.ReturnException(ret.getLocation(), ret.eval(env));
  }

  /**
   * Execute the statement.
   *
   * @throws EvalException if execution of the statement could not be completed.
   * @throws InterruptedException may be thrown in a sub class.
   */
  public void exec(Statement st) throws EvalException, InterruptedException {
    try {
      execDispatch(st);
    } catch (EvalException ex) {
      throw st.maybeTransformException(ex);
    }
  }

  void execDispatch(Statement st) throws EvalException, InterruptedException {
    switch (st.kind()) {
      case ASSIGNMENT:
        execAssignment((AssignmentStatement) st);
        break;
      case AUGMENTED_ASSIGNMENT:
        execAugmentedAssignment((AugmentedAssignmentStatement) st);
        break;
      case CONDITIONAL:
        execIfBranch((IfStatement.ConditionalStatements) st);
        break;
      case EXPRESSION:
        ((ExpressionStatement) st).getExpression().eval(env);
        break;
      case FLOW:
        throw ((FlowStatement) st).getKind() == FlowStatement.Kind.BREAK
            ? breakException
            : continueException;
      case FOR:
        execFor((ForStatement) st);
        break;
      case FUNCTION_DEF:
        execDef((FunctionDefStatement) st);
        break;
      case IF:
        execIf((IfStatement) st);
        break;
      case LOAD:
        execLoad((LoadStatement) st);
        break;
      case RETURN:
        execReturn((ReturnStatement) st);
        break;
    }
  }

  private void execStatements(ImmutableList<Statement> statements)
      throws EvalException, InterruptedException {
    // Hot code path, good chance of short lists which don't justify the iterator overhead.
    for (int i = 0; i < statements.size(); i++) {
      exec(statements.get(i));
    }
  }
}
