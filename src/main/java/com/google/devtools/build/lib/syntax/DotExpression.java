// Copyright 2014 Google Inc. All rights reserved.
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

/**
 * Syntax node for a dot expression.
 * e.g.  obj.field, but not obj.method()
 */
public final class DotExpression extends Expression {

  private final Expression obj;

  private final Ident field;

  public DotExpression(Expression obj, Ident field) {
    this.obj = obj;
    this.field = field;
  }

  public Expression getObj() {
    return obj;
  }

  public Ident getField() {
    return field;
  }

  @Override
  public String toString() {
    return obj + "." + field;
  }

  @Override
  Object eval(Environment env) throws EvalException, InterruptedException {
    Object objValue = obj.eval(env);
    if (objValue instanceof ClassObject) {
      Object result = ((ClassObject) objValue).getValue(field.getName());
      if (result == null) {
        // TODO(bazel_team): Throw an exception?
        return Environment.NONE;
      }
      return result;
    }

    throw new EvalException(getLocation(), "Object of type '"
        + EvalUtils.getDatatypeName(objValue) + "' has no field '" + field + "'");
 }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  SkylarkType validate(ValidationEnvironment env) throws EvalException {
    SkylarkType objType = obj.validate(env);
    // TODO(bazel_team): check existance of field
    return SkylarkType.UNKNOWN;
  }
}
