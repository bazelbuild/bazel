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

import com.google.devtools.build.lib.events.Location;
import java.io.IOException;

/**
 * An index expression ({@code obj[field]}). Not to be confused with a slice expression ({@code
 * obj[from:to]}). The object may be either a sequence or an associative mapping (most commonly
 * lists and dictionaries).
 */
public final class IndexExpression extends Expression {

  private final Expression object;

  private final Expression key;

  public IndexExpression(Expression object, Expression key) {
    this.object = object;
    this.key = key;
  }

  public Expression getObject() {
    return object;
  }

  public Expression getKey() {
    return key;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    object.prettyPrint(buffer);
    buffer.append('[');
    key.prettyPrint(buffer);
    buffer.append(']');
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    return evaluate(object.eval(env), key.eval(env), env, getLocation());
  }

  /**
   * Retrieves the value associated with a key in the given object.
   *
   * @throws EvalException if {@code object} is not a list or dictionary
   */
  public static Object evaluate(Object object, Object key, Environment env, Location loc)
      throws EvalException, InterruptedException {
    if (object instanceof SkylarkIndexable) {
      Object result = ((SkylarkIndexable) object).getIndex(key, loc, env.getStarlarkContext());
      // TODO(bazel-team): We shouldn't have this convertToSkylark call here. If it's needed at all,
      // it should go in the implementations of SkylarkIndexable#getIndex that produce non-Skylark
      // values.
      return SkylarkType.convertToSkylark(result, env);
    } else if (object instanceof String) {
      String string = (String) object;
      int index = EvalUtils.getSequenceIndex(key, string.length(), loc);
      return string.substring(index, index + 1);
    } else {
      throw new EvalException(
          loc,
          String.format(
              "type '%s' has no operator [](%s)",
              EvalUtils.getDataTypeName(object), EvalUtils.getDataTypeName(key)));
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.INDEX;
  }
}
