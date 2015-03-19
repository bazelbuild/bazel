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

import com.google.common.collect.ImmutableMap;

import java.util.LinkedHashMap;

/**
 * Syntax node for dictionary comprehension expressions.
 */
public class DictComprehension extends Expression {

  private final Expression keyExpression;
  private final Expression valueExpression;
  private final LValue loopVar;
  private final Expression listExpression;

  public DictComprehension(Expression keyExpression, Expression valueExpression, Expression loopVar,
      Expression listExpression) {
    this.keyExpression = keyExpression;
    this.valueExpression = valueExpression;
    this.loopVar = new LValue(loopVar);
    this.listExpression = listExpression;
  }

  Expression getKeyExpression() {
    return keyExpression;
  }

  Expression getValueExpression() {
    return valueExpression;
  }

  LValue getLoopVar() {
    return loopVar;
  }

  Expression getListExpression() {
    return listExpression;
  }

  @Override
  Object eval(Environment env) throws EvalException, InterruptedException {
    // We want to keep the iteration order
    LinkedHashMap<Object, Object> map = new LinkedHashMap<>();
    Iterable<?> elements = EvalUtils.toIterable(listExpression.eval(env), getLocation());
    for (Object element : elements) {
      loopVar.assign(env, getLocation(), element);
      Object key = keyExpression.eval(env);
      map.put(key, valueExpression.eval(env));
    }
    return ImmutableMap.copyOf(map);
  }

  @Override
  SkylarkType validate(ValidationEnvironment env) throws EvalException {
    SkylarkType elementsType = listExpression.validate(env);
    SkylarkType listElementType = SkylarkType.getGenericArgType(elementsType);
    loopVar.validate(env, getLocation(), listElementType);
    SkylarkType keyType = keyExpression.validate(env);
    if (!keyType.isSimple()) {
      // TODO(bazel-team): this is most probably dead code but it's better to have it here
      // in case we enable e.g. list of lists or we validate function calls on Java objects
      throw new EvalException(getLocation(), "Dict comprehension key must be of a simple type");
    }
    valueExpression.validate(env);
    if (elementsType != SkylarkType.UNKNOWN && !elementsType.isList()) {
      throw new EvalException(getLocation(), "Dict comprehension elements must be a list");
    }
    return SkylarkType.of(SkylarkType.MAP, keyType);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append('{').append(keyExpression).append(": ").append(valueExpression);
    sb.append(" for ").append(loopVar).append(" in ").append(listExpression);
    sb.append('}');
    return sb.toString();
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.accept(this);
  }
}
