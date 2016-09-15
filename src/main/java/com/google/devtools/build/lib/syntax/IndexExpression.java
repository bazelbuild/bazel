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

/** Syntax node for an index expression. e.g. obj[field], but not obj[from:to] */
public final class IndexExpression extends Expression {

  private final Expression obj;

  private final Expression key;

  public IndexExpression(Expression obj, Expression key) {
    this.obj = obj;
    this.key = key;
  }

  public Expression getObject() {
    return obj;
  }

  public Expression getKey() {
    return key;
  }

  @Override
  public String toString() {
    return String.format("%s[%s]", obj, key);
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    Object objValue = obj.eval(env);
    Object keyValue = key.eval(env);
    return eval(objValue, keyValue, getLocation(), env);
  }

  /**
   * Returns the field of the given key of the struct objValue, or null if no such field exists.
   */
  private Object eval(Object objValue, Object keyValue, Location loc, Environment env)
      throws EvalException {

    if (objValue instanceof SkylarkIndexable) {
      Object result = ((SkylarkIndexable) objValue).getIndex(keyValue, loc);
      return SkylarkType.convertToSkylark(result, env);
    } else if (objValue instanceof String) {
      String string = (String) objValue;
      int index = MethodLibrary.getListIndex(keyValue, string.length(), loc);
      return string.substring(index, index + 1);
    }

    throw new EvalException(
        loc,
        Printer.format(
            "Type %s has no operator [](%s)",
            EvalUtils.getDataTypeName(objValue),
            EvalUtils.getDataTypeName(keyValue)));
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    obj.validate(env);
  }
}
