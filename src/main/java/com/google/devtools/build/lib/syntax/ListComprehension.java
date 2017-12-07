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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Syntax node for lists comprehension expressions.
 */
public final class ListComprehension extends AbstractComprehension {
  private final Expression outputExpression;

  public ListComprehension(List<Clause> clauses, Expression outputExpression) {
    super(clauses, outputExpression);
    this.outputExpression = outputExpression;
  }

  @Override
  protected char openingBracket() {
    return '[';
  }

  @Override
  protected char closingBracket() {
    return ']';
  }

  public Expression getOutputExpression() {
    return outputExpression;
  }

  @Override
  protected void printExpressions(Appendable buffer) throws IOException {
    outputExpression.prettyPrint(buffer);
  }

  /** Builder for {@link ListComprehension}. */
  public static class Builder extends AbstractBuilder {

    private Expression outputExpression;

    public Builder setOutputExpression(Expression outputExpression) {
      this.outputExpression = outputExpression;
      return this;
    }

    @Override
    public ListComprehension build() {
      Preconditions.checkState(!clauses.isEmpty());
      return new ListComprehension(clauses, Preconditions.checkNotNull(outputExpression));
    }
  }

  @Override
  OutputCollector createCollector(Environment env) {
    return new ListOutputCollector();
  }

  /**
   * Helper class that collects the intermediate results of the {@code ListComprehension} and
   * provides access to the resulting {@code List}.
   */
  private final class ListOutputCollector implements OutputCollector {
    private final List<Object> result;

    ListOutputCollector() {
      result = new ArrayList<>();
    }

    @Override
    public Location getLocation() {
      return ListComprehension.this.getLocation();
    }

    @Override
    public List<Clause> getClauses() {
      return ListComprehension.this.getClauses();
    }

    @Override
    public void evaluateAndCollect(Environment env) throws EvalException, InterruptedException {
      result.add(outputExpression.eval(env));
    }

    @Override
    public Object getResult(Environment env) throws EvalException {
      return MutableList.copyOf(env, result);
    }
  }
}
