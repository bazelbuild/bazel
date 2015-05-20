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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.FuncallExpression.MethodDescriptor;

import java.util.List;

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
    String name = field.getName();
    Object result = eval(objValue, name, getLocation());
    if (result == null) {
      if (objValue instanceof ClassObject) {
        String customErrorMessage = ((ClassObject) objValue).errorMessage(name);
        if (customErrorMessage != null) {
          throw new EvalException(getLocation(), customErrorMessage);
        }
      }
      throw new EvalException(getLocation(), String.format("Object of type '%s' has no field %s",
              EvalUtils.getDataTypeName(objValue), EvalUtils.prettyPrintValue(name)));
    }
    return result;
  }

  /**
   * Returns the field of the given name of the struct objValue, or null if no such field exists.
   */
  public static Object eval(Object objValue, String name, Location loc) throws EvalException {
    if (objValue instanceof ClassObject) {
      Object result = ((ClassObject) objValue).getValue(name);
      result = SkylarkType.convertToSkylark(result, loc);
      // If we access NestedSets using ClassObject.getValue() we won't know the generic type,
      // so we have to disable it. This should not happen.
      SkylarkType.checkTypeAllowedInSkylark(result, loc);
      return result;
    }
    List<MethodDescriptor> methods = FuncallExpression.getMethods(objValue.getClass(),
          name, 0, loc);
    if (methods != null && !methods.isEmpty()) {
      MethodDescriptor method = Iterables.getOnlyElement(methods);
      if (method.getAnnotation().structField()) {
        return FuncallExpression.callMethod(method, name, objValue, new Object[] {}, loc);
      }
    }
    return null;
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
