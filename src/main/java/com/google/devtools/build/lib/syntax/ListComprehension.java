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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Syntax node for lists comprehension expressions.
 */
public final class ListComprehension extends Expression {

  private final Expression elementExpression;
  // This cannot be a map, because we need to both preserve order _and_ allow duplicate identifiers.
  private final List<Map.Entry<Ident, Expression>> lists;

  /**
   * [elementExpr (for var in listExpr)+]
   */
  public ListComprehension(Expression elementExpression) {
    this.elementExpression = elementExpression;
    lists = new ArrayList<>();
  }

  @Override
  Object eval(Environment env) throws EvalException, InterruptedException {
    if (lists.isEmpty()) {
      return convert(new ArrayList<>(), env);
    }

    List<Map.Entry<Ident, Iterable<?>>> listValues = Lists.newArrayListWithCapacity(lists.size());
    int size = 1;
    for (Map.Entry<Ident, Expression> list : lists) {
      Object listValueObject = list.getValue().eval(env);
      final Iterable<?> listValue = EvalUtils.toIterable(listValueObject, getLocation());
      int listSize = EvalUtils.size(listValue);
      if (listSize == 0) {
        return convert(new ArrayList<>(), env);
      }
      size *= listSize;
      listValues.add(Maps.<Ident, Iterable<?>>immutableEntry(list.getKey(), listValue));
    }
    List<Object> resultList = Lists.newArrayListWithCapacity(size);
    evalLists(env, listValues, resultList);
    return convert(resultList, env);
  }

  private Object convert(List<Object> list, Environment env) throws EvalException {
    if (env.isSkylarkEnabled()) {
      return SkylarkList.list(list, getLocation());
    } else {
      return list;
    }
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append('[').append(elementExpression);
    for (Map.Entry<Ident, Expression> list : lists) {
      sb.append(" for ").append(list.getKey()).append(" in ").append(list.getValue());
    }
    sb.append(']');
    return sb.toString();
  }

  public Expression getElementExpression() {
    return elementExpression;
  }

  public void add(Ident ident, Expression listExpression) {
    lists.add(Maps.immutableEntry(ident, listExpression));
  }

  public List<Map.Entry<Ident, Expression>> getLists() {
    return lists;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  /**
   * Evaluates element expression over all combinations of list element values.
   *
   * <p>Iterates over all elements in outermost list (list at index 0) and
   * updates the value of the list variable in the environment on each
   * iteration. If there are no other lists to iterate over added evaluation
   * of the element expression to the result. Otherwise calls itself recursively
   * with all the lists except the outermost.
   */
  private void evalLists(Environment env, List<Map.Entry<Ident, Iterable<?>>> listValues,
      List<Object> result) throws EvalException, InterruptedException {
    Map.Entry<Ident, Iterable<?>> listValue = listValues.get(0);
    for (Object listElement : listValue.getValue()) {
      env.update(listValue.getKey().getName(), listElement);
      if (listValues.size() == 1) {
        result.add(elementExpression.eval(env));
      } else {
        evalLists(env, listValues.subList(1, listValues.size()), result);
      }
    }
  }

  @Override
  SkylarkType validate(ValidationEnvironment env) throws EvalException {
    for (Map.Entry<Ident, Expression> list : lists) {
      // TODO(bazel-team): Get the type of elements
      SkylarkType type = list.getValue().validate(env);
      env.checkIterable(type, getLocation());
      env.update(list.getKey().getName(), SkylarkType.UNKNOWN, getLocation());
    }
    elementExpression.validate(env);
    return SkylarkType.LIST;
  }
}
