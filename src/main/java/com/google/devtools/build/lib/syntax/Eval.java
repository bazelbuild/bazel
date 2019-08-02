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
import com.google.devtools.build.lib.events.Location;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Function;

/**
 * Evaluation code for the Skylark AST. At the moment, it can execute only statements (and defers to
 * Expression.eval for evaluating expressions).
 */
public class Eval {
  protected final Environment env;

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

  private static final FlowException breakException = new FlowException("FlowException - break");
  private static final FlowException continueException =
      new FlowException("FlowException - continue");

  /**
   * This constructor should never be called directly. Call {@link #fromEnvironment(Environment)}
   * instead.
   */
  protected Eval(Environment env) {
    this.env = env;
  }

  void execAssignment(AssignmentStatement node) throws EvalException, InterruptedException {
    Object rvalue = node.getRHS().eval(env);
    assign(node.getLHS(), rvalue, env, node.getLocation());
  }

  void execAugmentedAssignment(AugmentedAssignmentStatement node)
      throws EvalException, InterruptedException {
    assignAugmented(node.getLHS(), node.getOperator(), node.getRHS(), env, node.getLocation());
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
        assign(node.getLHS(), it, env, node.getLocation());

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

  void execIf(IfStatement node) throws EvalException, InterruptedException {
    ImmutableList<IfStatement.ConditionalStatements> thenBlocks = node.getThenBlocks();
    // Avoid iterator overhead - most of the time there will be one or few "if"s.
    for (int i = 0; i < thenBlocks.size(); i++) {
      IfStatement.ConditionalStatements stmt = thenBlocks.get(i);
      if (EvalUtils.toBoolean(stmt.getCondition().eval(env))) {
        exec(stmt);
        return;
      }
    }
    execStatements(node.getElseBlock());
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
      case PASS:
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

  /**
   * Updates the environment bindings, and possibly mutates objects, so as to assign the given value
   * to the given expression. The expression must be valid for an {@code LValue}.
   */
  // TODO(adonovan): make this a private instance method once all Expression.eval methods move here.
  static void assign(Expression expr, Object value, Environment env, Location loc)
      throws EvalException, InterruptedException {
    if (expr instanceof Identifier) {
      assignIdentifier((Identifier) expr, value, env);
    } else if (expr instanceof IndexExpression) {
      Object object = ((IndexExpression) expr).getObject().eval(env);
      Object key = ((IndexExpression) expr).getKey().eval(env);
      assignItem(object, key, value, env, loc);
    } else if (expr instanceof ListLiteral) {
      ListLiteral list = (ListLiteral) expr;
      assignList(list, value, env, loc);
    } else {
      // Not possible for validated ASTs.
      throw new EvalException(loc, "cannot assign to '" + expr + "'");
    }
  }

  /** Binds a variable to the given value in the environment. */
  private static void assignIdentifier(Identifier ident, Object value, Environment env)
      throws EvalException {
    env.updateAndExport(ident.getName(), value);
  }

  /**
   * Adds or changes an object-key-value relationship for a list or dict.
   *
   * <p>For a list, the key is an in-range index. For a dict, it is a hashable value.
   *
   * @throws EvalException if the object is not a list or dict
   */
  @SuppressWarnings("unchecked")
  private static void assignItem(
      Object object, Object key, Object value, Environment env, Location loc) throws EvalException {
    if (object instanceof SkylarkDict) {
      SkylarkDict<Object, Object> dict = (SkylarkDict<Object, Object>) object;
      dict.put(key, value, loc, env);
    } else if (object instanceof SkylarkList.MutableList) {
      SkylarkList.MutableList<Object> list = (SkylarkList.MutableList<Object>) object;
      int index = EvalUtils.getSequenceIndex(key, list.size(), loc);
      list.set(index, value, loc, env.mutability());
    } else {
      throw new EvalException(
          loc,
          "can only assign an element in a dictionary or a list, not in a '"
              + EvalUtils.getDataTypeName(object)
              + "'");
    }
  }

  /**
   * Recursively assigns an iterable value to a list literal.
   *
   * @throws EvalException if the list literal has length 0, or if the value is not an iterable of
   *     matching length
   */
  private static void assignList(ListLiteral list, Object value, Environment env, Location loc)
      throws EvalException, InterruptedException {
    Collection<?> collection = EvalUtils.toCollection(value, loc, env);
    int len = list.getElements().size();
    if (len == 0) {
      throw new EvalException(
          loc, "lists or tuples on the left-hand side of assignments must have at least one item");
    }
    if (len != collection.size()) {
      throw new EvalException(
          loc,
          String.format(
              "assignment length mismatch: left-hand side has length %d, but right-hand side"
                  + " evaluates to value of length %d",
              len, collection.size()));
    }
    int i = 0;
    for (Object item : collection) {
      assign(list.getElements().get(i), item, env, loc);
      i++;
    }
  }

  /**
   * Evaluates an augmented assignment that mutates this {@code LValue} with the given right-hand
   * side's value.
   *
   * <p>The left-hand side expression is evaluated only once, even when it is an {@link
   * IndexExpression}. The left-hand side is evaluated before the right-hand side to match Python's
   * behavior (hence why the right-hand side is passed as an expression rather than as an evaluated
   * value).
   */
  private static void assignAugmented(
      Expression expr, TokenKind op, Expression rhs, Environment env, Location loc)
      throws EvalException, InterruptedException {
    if (expr instanceof Identifier) {
      Object result =
          BinaryOperatorExpression.evaluateAugmented(op, expr.eval(env), rhs.eval(env), env, loc);
      assignIdentifier((Identifier) expr, result, env);
    } else if (expr instanceof IndexExpression) {
      IndexExpression indexExpression = (IndexExpression) expr;
      // The object and key should be evaluated only once, so we don't use expr.eval().
      Object object = indexExpression.getObject().eval(env);
      Object key = indexExpression.getKey().eval(env);
      Object oldValue = IndexExpression.evaluate(object, key, env, loc);
      // Evaluate rhs after lhs.
      Object rhsValue = rhs.eval(env);
      Object result = BinaryOperatorExpression.evaluateAugmented(op, oldValue, rhsValue, env, loc);
      assignItem(object, key, result, env, loc);
    } else if (expr instanceof ListLiteral) {
      throw new EvalException(loc, "cannot perform augmented assignment on a list literal");
    } else {
      // Not possible for validated ASTs.
      throw new EvalException(loc, "cannot perform augmented assignment on '" + expr + "'");
    }
  }
}
