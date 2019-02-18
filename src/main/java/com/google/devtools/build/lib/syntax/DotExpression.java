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
import com.google.devtools.build.lib.util.SpellChecker;
import java.io.IOException;

/** Syntax node for a dot expression. e.g. obj.field, but not obj.method() */
public final class DotExpression extends Expression {

  private final Expression object;

  private final Identifier field;

  public DotExpression(Expression object, Identifier field) {
    this.object = object;
    this.field = field;
  }

  public Expression getObject() {
    return object;
  }

  public Identifier getField() {
    return field;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    object.prettyPrint(buffer);
    buffer.append('.');
    field.prettyPrint(buffer);
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    Object objValue = object.eval(env);
    String name = field.getName();
    Object result = eval(objValue, name, getLocation(), env);
    return checkResult(objValue, result, name, getLocation(), env.getSemantics());
  }

  /** Throws the correct error message if the result is null depending on the objValue. */
  public static Object checkResult(
      Object objValue, Object result, String name, Location loc, StarlarkSemantics semantics)
      throws EvalException {
    if (result != null) {
      return result;
    }
    throw getMissingFieldException(objValue, name, loc, semantics, "field");
  }

  static EvalException getMissingFieldException(
      Object objValue, String name, Location loc, StarlarkSemantics semantics, String accessName) {
    String suffix = "";
    EvalException toSuppress = null;
    if (objValue instanceof ClassObject) {
      String customErrorMessage = ((ClassObject) objValue).getErrorMessageForUnknownField(name);
      if (customErrorMessage != null) {
        return new EvalException(loc, customErrorMessage);
      }
      try {
        suffix = SpellChecker.didYouMean(name, ((ClassObject) objValue).getFieldNames());
      } catch (EvalException ee) {
        toSuppress = ee;
      }
    } else {
      suffix =
          SpellChecker.didYouMean(
              name,
              FuncallExpression.getStructFieldNames(
                  semantics,
                  objValue instanceof Class ? (Class<?>) objValue : objValue.getClass()));
    }
    if (suffix.isEmpty() && hasMethod(semantics, objValue, name)) {
      // If looking up the field failed, then we know that this method must have struct_field=false
      suffix = ", however, a method of that name exists";
    }
    EvalException ee =
        new EvalException(
            loc,
            String.format(
                "object of type '%s' has no %s '%s'%s",
                EvalUtils.getDataTypeName(objValue), accessName, name, suffix));
    if (toSuppress != null) {
      ee.addSuppressed(toSuppress);
    }
    return ee;
  }

  /** Returns whether the given object has a method with the given name. */
  static boolean hasMethod(StarlarkSemantics semantics, Object obj, String name) {
    Class<?> cls = obj instanceof Class ? (Class<?>) obj : obj.getClass();
    if (Runtime.getBuiltinRegistry().getFunctionNames(cls).contains(name)) {
      return true;
    }

    return FuncallExpression.getMethodNames(semantics, cls).contains(name);
  }

  /**
   * Returns the field of the given name of the struct objValue, or null if no such field exists.
   */
  public static Object eval(Object objValue, String name,
      Location loc, Environment env) throws EvalException, InterruptedException {

    MethodDescriptor method =
        objValue instanceof Class<?>
            ? FuncallExpression.getMethod(env.getSemantics(), (Class<?>) objValue, name)
            : FuncallExpression.getMethod(env.getSemantics(), objValue.getClass(), name);

    if (method != null && method.isStructField()) {
      return method
          .call(
              objValue,
              FuncallExpression.extraInterpreterArgs(method, /* ast = */ null, loc, env)
                  .toArray(),
              loc,
              env);
    }

    if (objValue instanceof SkylarkClassObject) {
      try {
        return ((SkylarkClassObject) objValue).getValue(name);
      } catch (IllegalArgumentException ex) {
        throw new EvalException(loc, ex);
      }
    } else if (objValue instanceof ClassObject) {
      Object result = null;
      try {
        result = ((ClassObject) objValue).getValue(name);
      } catch (IllegalArgumentException ex) {
        throw new EvalException(loc, ex);
      }
      // ClassObjects may have fields that are annotated with @SkylarkCallable.
      // Since getValue() does not know about those, we cannot expect that result is a valid object.
      if (result != null) {
        result = SkylarkType.convertToSkylark(result, env);
        // If we access NestedSets using ClassObject.getValue() we won't know the generic type,
        // so we have to disable it. This should not happen.
        SkylarkType.checkTypeAllowedInSkylark(result, loc);
        return result;
      }
    }

    return null;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.DOT;
  }
}
