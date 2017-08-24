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
    public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
      key.prettyPrint(buffer);
      buffer.append(": ");
      value.prettyPrint(buffer);
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
    return new DictionaryLiteral(ImmutableList.of());
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    SkylarkDict<Object, Object> dict = SkylarkDict.of(env);
    Location loc = getLocation();
    for (DictionaryEntryLiteral entry : entries) {
      Object key = entry.key.eval(env);
      Object val = entry.value.eval(env);
      if (env.getSemantics().incompatibleDictLiteralHasNoDuplicates && dict.containsKey(key)) {
        throw new EvalException(
            loc, "Duplicated key " + Printer.repr(key) + " when creating dictionary");
      }
      dict.put(key, val, loc, env);
    }
    return dict;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    buffer.append("{");
    String sep = "";
    for (DictionaryEntryLiteral e : entries) {
      buffer.append(sep);
      e.prettyPrint(buffer);
      sep = ", ";
    }
    buffer.append("}");
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.DICTIONARY_LITERAL;
  }

  public ImmutableList<DictionaryEntryLiteral> getEntries() {
    return entries;
  }
}
