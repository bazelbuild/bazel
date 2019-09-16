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
import java.util.ArrayList;

/**
 * Syntax node for list and dict comprehensions.
 *
 * <p>A comprehension contains one or more clauses, e.g. [a+d for a in b if c for d in e] contains
 * three clauses: "for a in b", "if c", "for d in e". For and If clauses can happen in any order,
 * except that the first one has to be a For.
 *
 * <p>The code above can be expanded as:
 *
 * <pre>
 *   for a in b:
 *     if c:
 *       for d in e:
 *         result.append(a+d)
 * </pre>
 *
 * result is initialized to [] (list) or {} (dict) and is the return value of the whole expression.
 */
public final class Comprehension extends Expression {

  /** For or If */
  public abstract static class Clause {}

  /** A for clause in a comprehension, e.g. "for a in b" in the example above. */
  public static final class For extends Clause {
    private final Expression vars;
    private final Expression iterable;

    For(Expression vars, Expression iterable) {
      this.vars = vars;
      this.iterable = iterable;
    }

    public Expression getVars() {
      return vars;
    }

    public Expression getIterable() {
      return iterable;
    }
  }

  /** A if clause in a comprehension, e.g. "if c" in the example above. */
  public static final class If extends Clause {
    private final Expression condition;

    If(Expression condition) {
      this.condition = condition;
    }

    public Expression getCondition() {
      return condition;
    }
  }

  private final boolean isDict; // {k: v for vars in iterable}
  private final ASTNode body; // Expression or DictionaryLiteral.Entry
  private final ImmutableList<Clause> clauses;

  Comprehension(boolean isDict, ASTNode body, ImmutableList<Clause> clauses) {
    this.isDict = isDict;
    this.body = body;
    this.clauses = clauses;
  }

  public boolean isDict() {
    return isDict;
  }

  /**
   * Returns the loop body: an expression for a list comprehension, or a DictionaryLiteral.Entry for
   * a dict comprehension.
   */
  public ASTNode getBody() {
    return body;
  }

  public ImmutableList<Clause> getClauses() {
    return clauses;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    buffer.append(isDict ? '{' : '[');
    body.prettyPrint(buffer);
    for (Clause clause : clauses) {
      buffer.append(' ');
      if (clause instanceof For) {
        For forClause = (For) clause;
        buffer.append("for ");
        forClause.vars.prettyPrint(buffer);
        buffer.append(" in ");
        forClause.iterable.prettyPrint(buffer);
      } else {
        If ifClause = (If) clause;
        buffer.append("if ");
        ifClause.condition.prettyPrint(buffer);
      }
    }
    buffer.append(isDict ? '}' : ']');
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
    final SkylarkDict<Object, Object> dict = isDict ? SkylarkDict.of(env) : null;
    final ArrayList<Object> list = isDict ? null : new ArrayList<>();

    // The Lambda class serves as a recursive lambda closure.
    class Lambda {
      // execClauses(index) recursively executes the clauses starting at index,
      // and finally evaluates the body and adds its value to the result.
      void execClauses(int index) throws EvalException, InterruptedException {
        // recursive case: one or more clauses
        if (index < clauses.size()) {
          Clause clause = clauses.get(index);
          if (clause instanceof For) {
            For forClause = (For) clause;

            Object iterable = forClause.getIterable().eval(env);
            Location loc = getLocation();
            Iterable<?> listValue = EvalUtils.toIterable(iterable, loc, env);
            EvalUtils.lock(iterable, loc);
            try {
              for (Object elem : listValue) {
                Eval.assign(forClause.getVars(), elem, env, loc);
                execClauses(index + 1);
              }
            } finally {
              EvalUtils.unlock(iterable, loc);
            }

          } else {
            If ifClause = (If) clause;
            if (EvalUtils.toBoolean(ifClause.condition.eval(env))) {
              execClauses(index + 1);
            }
          }
          return;
        }

        // base case: evaluate body and add to result.
        if (dict != null) {
          DictionaryLiteral.Entry body = (DictionaryLiteral.Entry) Comprehension.this.body;
          Object k = body.getKey().eval(env);
          EvalUtils.checkValidDictKey(k, env);
          Object v = body.getValue().eval(env);
          dict.put(k, v, getLocation(), env);
        } else {
          list.add(((Expression) body).eval(env));
        }
      }
    }
    new Lambda().execClauses(0);

    // Undefine loop variables (remove them from the environment).
    // This code is useful for the transition, to make sure no one relies on the old behavior
    // (where loop variables were leaking).
    // TODO(laurentlb): Instead of removing variables, we should create them in a nested scope.
    for (Clause clause : clauses) {
      // Check if a loop variable conflicts with another local variable.
      if (clause instanceof For) {
        for (Identifier ident : Identifier.boundIdentifiers(((For) clause).getVars())) {
          env.removeLocalBinding(ident.getName());
        }
      }
    }

    return isDict ? dict : SkylarkList.MutableList.copyOf(env, list);
  }

}
