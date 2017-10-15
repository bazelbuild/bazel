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
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Base class for list and dict comprehension expressions.
 *
 * <p> A comprehension contains one or more clauses, e.g.
 *   [a+d for a in b if c for d in e]
 * contains three clauses: "for a in b", "if c", "for d in e".
 * For and If clauses can happen in any order, except that the first one has to be a For.
 *
 * <p> The code above can be expanded as:
 * <pre>
 *   for a in b:
 *     if c:
 *       for d in e:
 *         result.append(a+d)
 * </pre>
 * result is initialized to [] (list) or {} (dict) and is the return value of the whole expression.
 */
public abstract class AbstractComprehension extends Expression {

  /**
   * The interface implemented by ForClause and (later) IfClause.
   * A comprehension consists of one or many Clause.
   */
  public interface Clause extends Serializable {

    /** Enum for distinguishing clause types. */
    enum Kind {
      FOR,
      IF
    }

    /**
     * Returns whether this is a For or If clause.
     *
     * <p>This avoids having to rely on reflection, or on checking whether {@link #getLValue} is
     * null.
     */
    Kind getKind();

    /**
     * The evaluation of the comprehension is based on recursion. Each clause may
     * call recursively evalStep (ForClause will call it multiple times, IfClause will
     * call it zero or one time) which will evaluate the next clause. To know which clause
     * is the next one, we pass a step argument (it represents the index in the clauses
     * list). Results are aggregated in the result argument, and are populated by
     * evalStep.
     *
     * @param env environment in which we do the evaluation.
     * @param collector the aggregated results of the comprehension.
     * @param step the index of the next clause to evaluate.
     */
    void eval(Environment env, OutputCollector collector, int step)
        throws EvalException, InterruptedException;

    /**
     * The LValue defined in Clause, i.e. the loop variables for ForClause and null for
     * IfClause. This is needed for SyntaxTreeVisitor.
     */
    @Nullable  // for the IfClause
    LValue getLValue();

    /**
     * The Expression defined in Clause, i.e. the collection for ForClause and the
     * condition for IfClause. This is needed for SyntaxTreeVisitor.
     */
    Expression getExpression();

    /** Pretty print to a buffer. */
    void prettyPrint(Appendable buffer) throws IOException;
  }

  /**
   * A for clause in a comprehension, e.g. "for a in b" in the example above.
   */
  public static final class ForClause implements Clause {
    private final LValue lvalue;
    private final Expression iterable;

    @Override
    public Kind getKind() {
      return Kind.FOR;
    }

    public ForClause(LValue lvalue, Expression iterable) {
      this.lvalue = lvalue;
      this.iterable = iterable;
    }

    @Override
    public void eval(Environment env, OutputCollector collector, int step)
        throws EvalException, InterruptedException {
      Object iterableObject = iterable.eval(env);
      Location loc = collector.getLocation();
      Iterable<?> listValue = EvalUtils.toIterable(iterableObject, loc, env);
      EvalUtils.lock(iterableObject, loc);
      try {
        for (Object listElement : listValue) {
          lvalue.assign(listElement, env, loc);
          evalStep(env, collector, step);
        }
      } finally {
        EvalUtils.unlock(iterableObject, loc);
      }
    }

    @Override
    public LValue getLValue() {
      return lvalue;
    }

    @Override
    public Expression getExpression() {
      return iterable;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append("for ");
      lvalue.prettyPrint(buffer);
      buffer.append(" in ");
      iterable.prettyPrint(buffer);
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      try {
        prettyPrint(builder);
      } catch (IOException e) {
        // Not possible for StringBuilder.
        throw new AssertionError(e);
      }
      return builder.toString();
    }
  }

  /**
   * A if clause in a comprehension, e.g. "if c" in the example above.
   */
  public static final class IfClause implements Clause {
    private final Expression condition;

    @Override
    public Kind getKind() {
      return Kind.IF;
    }

    public IfClause(Expression condition) {
      this.condition = condition;
    }

    @Override
    public void eval(Environment env, OutputCollector collector, int step)
        throws EvalException, InterruptedException {
      if (EvalUtils.toBoolean(condition.eval(env))) {
        evalStep(env, collector, step);
      }
    }

    @Override
    public LValue getLValue() {
      return null;
    }

    @Override
    public Expression getExpression() {
      return condition;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append("if ");
      condition.prettyPrint(buffer);
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      try {
        prettyPrint(builder);
      } catch (IOException e) {
        // Not possible for StringBuilder.
        throw new AssertionError(e);
      }
      return builder.toString();
    }
  }

  /**
   * The output expressions, e.g. "a+d" in the example above. This list has either one (list) or two
   * (dict) items.
   */
  private final ImmutableList<Expression> outputExpressions;

  private final ImmutableList<Clause> clauses;

  public AbstractComprehension(List<Clause> clauses, Expression... outputExpressions) {
    this.clauses = ImmutableList.copyOf(clauses);
    this.outputExpressions = ImmutableList.copyOf(outputExpressions);
  }

  protected abstract char openingBracket();

  protected abstract char closingBracket();

  public ImmutableList<Expression> getOutputExpressions() {
    return outputExpressions;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    buffer.append(openingBracket());
    printExpressions(buffer);
    for (Clause clause : clauses) {
      buffer.append(' ');
      clause.prettyPrint(buffer);
    }
    buffer.append(closingBracket());
  }

  /** Base class for comprehension builders. */
  public abstract static class AbstractBuilder {

    protected final List<Clause> clauses = new ArrayList<>();

    public void addFor(LValue lvalue, Expression iterable) {
      Clause forClause = new ForClause(lvalue, iterable);
      clauses.add(forClause);
    }

    public void addIf(Expression condition) {
      clauses.add(new IfClause(condition));
    }

    public abstract AbstractComprehension build();
  }

  public ImmutableList<Clause> getClauses() {
    return clauses;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.COMPREHENSION;
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    OutputCollector collector = createCollector(env);
    evalStep(env, collector, 0);
    Object result = collector.getResult(env);

    if (!env.getSemantics().incompatibleComprehensionVariablesDoNotLeak) {
      return result;
    }

    // Undefine loop variables (remove them from the environment).
    // This code is useful for the transition, to make sure no one relies on the old behavior
    // (where loop variables were leaking).
    // TODO(laurentlb): Instead of removing variables, we should create them in a nested scope.
    for (Clause clause : clauses) {
      // Check if a loop variable conflicts with another local variable.
      LValue lvalue = clause.getLValue();
      if (lvalue != null) {
        for (Identifier ident : lvalue.boundIdentifiers()) {
          env.removeLocalBinding(ident.getName());
        }
      }
    }
    return result;
  }

  /**
   * Evaluate the clause indexed by step, or elementExpression. When we evaluate the
   * comprehension, step is 0 and we evaluate the first clause. Each clause may
   * recursively call evalStep any number of times. After the last clause,
   * the output expression(s) is/are evaluated and added to the results.
   *
   * <p> In the expanded example above, you can consider that evalStep is equivalent to
   * evaluating the line number step.
   */
  private static void evalStep(Environment env, OutputCollector collector, int step)
      throws EvalException, InterruptedException {
    List<Clause> clauses = collector.getClauses();
    if (step >= clauses.size()) {
      collector.evaluateAndCollect(env);
    } else {
      clauses.get(step).eval(env, collector, step + 1);
    }
  }

  /** Pretty-prints the output expression(s). */
  protected abstract void printExpressions(Appendable buffer) throws IOException;

  abstract OutputCollector createCollector(Environment env);

  /**
   * Interface for collecting the intermediate output of an {@code AbstractComprehension} and for
   * providing access to the final results.
   */
  interface OutputCollector {

    /** Returns the location for the comprehension we are evaluating. */
    Location getLocation();

    /** Returns the list of clauses for the comprehension we are evaluating. */
    List<Clause> getClauses();

    /**
     * Evaluates the output expression(s) of the comprehension and collects the result.
     */
    void evaluateAndCollect(Environment env) throws EvalException, InterruptedException;

    /**
     * Returns the final result of the comprehension.
     */
    Object getResult(Environment env) throws EvalException;
  }
}
