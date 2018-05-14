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
import java.io.IOException;
import java.util.List;
import java.util.Map;

/** Syntax node for dictionary comprehension expressions. */
public final class DictComprehension extends AbstractComprehension {
  private final Expression keyExpression;
  private final Expression valueExpression;

  public DictComprehension(
      List<Clause> clauses, Expression keyExpression, Expression valueExpression) {
    super(clauses, keyExpression, valueExpression);
    this.keyExpression = keyExpression;
    this.valueExpression = valueExpression;
  }

  @Override
  protected char openingBracket() {
    return '{';
  }

  @Override
  protected char closingBracket() {
    return '}';
  }

  public Expression getKeyExpression() {
    return keyExpression;
  }

  public Expression getValueExpression() {
    return valueExpression;
  }

  @Override
  protected void printExpressions(Appendable buffer) throws IOException {
    keyExpression.prettyPrint(buffer);
    buffer.append(": ");
    valueExpression.prettyPrint(buffer);
  }

  /** Builder for {@link DictComprehension}. */
  public static class Builder extends AbstractBuilder {

    private Expression keyExpression;
    private Expression valueExpression;

    public Builder setKeyExpression(Expression keyExpression) {
      this.keyExpression = keyExpression;
      return this;
    }

    public Builder setValueExpression(Expression valueExpression) {
      this.valueExpression = valueExpression;
      return this;
    }

    @Override
    public DictComprehension build() {
      Preconditions.checkState(!clauses.isEmpty());
      return new DictComprehension(
          clauses,
          Preconditions.checkNotNull(keyExpression),
          Preconditions.checkNotNull(valueExpression));
    }
  }

  @Override
  OutputCollector createCollector(Environment env) {
    return new DictOutputCollector(env);
  }

  /**
   * Helper class that collects the intermediate results of the {@link DictComprehension} and
   * provides access to the resulting {@link Map}.
   */
  private final class DictOutputCollector implements OutputCollector {
    private final SkylarkDict<Object, Object> result;

    DictOutputCollector(Environment env) {
      // We want to keep the iteration order
      result = SkylarkDict.of(env);
    }

    @Override
    public Location getLocation() {
      return DictComprehension.this.getLocation();
    }

    @Override
    public List<Clause> getClauses() {
      return DictComprehension.this.getClauses();
    }

    @Override
    public void evaluateAndCollect(Environment env) throws EvalException, InterruptedException {
      Object key = keyExpression.eval(env);
      EvalUtils.checkValidDictKey(key);
      result.put(key, valueExpression.eval(env), getLocation(), env);
    }

    @Override
    public Object getResult(Environment env) throws EvalException {
      return result;
    }
  }
}
