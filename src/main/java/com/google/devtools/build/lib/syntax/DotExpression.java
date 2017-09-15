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

import com.google.common.collect.Streams;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.FuncallExpression.MethodDescriptor;
import com.google.devtools.build.lib.util.SpellChecker;
import java.io.IOException;
import java.util.Optional;

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
    return checkResult(objValue, result, name, getLocation());
  }

  /**
   * Throws the correct error message if the result is null depending on the objValue.
   */
  public static Object checkResult(Object objValue, Object result, String name, Location loc)
      throws EvalException {
    if (result != null) {
      return result;
    }
    String suffix = "";
    if (objValue instanceof ClassObject) {
      String customErrorMessage = ((ClassObject) objValue).errorMessage(name);
      if (customErrorMessage != null) {
        throw new EvalException(loc, customErrorMessage);
      }
      suffix = SpellChecker.didYouMean(name, ((ClassObject) objValue).getKeys());
    }
    throw new EvalException(
        loc,
        String.format(
            "object of type '%s' has no field '%s'%s",
            EvalUtils.getDataTypeName(objValue), name, suffix));
  }

  /**
   * Returns the field of the given name of the struct objValue, or null if no such field exists.
   */
  public static Object eval(Object objValue, String name,
      Location loc, Environment env) throws EvalException {
    if (objValue instanceof ClassObject) {
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

    Iterable<MethodDescriptor> methods =
        objValue instanceof Class<?>
            ? FuncallExpression.getMethods((Class<?>) objValue, name)
            : FuncallExpression.getMethods(objValue.getClass(), name);

    if (methods != null) {
      Optional<MethodDescriptor> method =
          Streams.stream(methods)
              .filter(methodDescriptor -> methodDescriptor.getAnnotation().structField())
              .findFirst();
      if (method.isPresent() && method.get().getAnnotation().structField()) {
        return FuncallExpression.callMethod(
            method.get(), name, objValue, new Object[] {}, loc, env);
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
