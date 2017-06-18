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
import java.util.List;

/**
 * Syntax node for dictionary literals.
 */
public final class DictionaryLiteral extends Expression {

  /** Node for an individual key-value pair in a dictionary literal. */
  public static final class DictionaryEntryLiteral extends ASTNode {

    private final Expression key;
    private final Expression value;

    public DictionaryEntryLiteral(Expression key, Expression value) {
      this.key = key;
      this.value = value;
    }

    public Expression getKey() {
      return key;
    }

    public Expression getValue() {
      return value;
    }

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append(key);
      sb.append(": ");
      sb.append(value);
      return sb.toString();
    }

    @Override
    public void accept(SyntaxTreeVisitor visitor) {
      visitor.visit(this);
    }
  }

  private final ImmutableList<DictionaryEntryLiteral> entries;

  public DictionaryLiteral(List<DictionaryEntryLiteral> exprs) {
    this.entries = ImmutableList.copyOf(exprs);
  }

  /** A new literal for an empty dictionary, onto which a new location can be specified */
  public static DictionaryLiteral emptyDict() {
    return new DictionaryLiteral(ImmutableList.<DictionaryEntryLiteral>of());
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    SkylarkDict<Object, Object> dict = SkylarkDict.<Object, Object>of(env);
    Location loc = getLocation();
    for (DictionaryEntryLiteral entry : entries) {
      if (entry == null) {
        throw new EvalException(loc, "null expression in " + this);
      }
      Object key = entry.key.eval(env);
      Object val = entry.value.eval(env);
      dict.put(key, val, loc, env);
    }
    return dict;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("{");
    String sep = "";
    for (DictionaryEntryLiteral e : entries) {
      sb.append(sep);
      sb.append(e);
      sep = ", ";
    }
    sb.append("}");
    return sb.toString();
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  public ImmutableList<DictionaryEntryLiteral> getEntries() {
    return entries;
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    for (DictionaryEntryLiteral entry : entries) {
      entry.key.validate(env);
      entry.value.validate(env);
    }
  }
}
