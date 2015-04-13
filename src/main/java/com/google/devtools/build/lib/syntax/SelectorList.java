// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;

import java.util.List;

/**
 * An attribute value consisting of a concatenation of native types and selects, e.g:
 *
 * <pre>
 *   rule(
 *       name = 'myrule',
 *       deps =
 *           [':defaultdep']
 *           + select({
 *               'a': [':adep'],
 *               'b': [':bdep'],})
 *           + select({
 *               'c': [':cdep'],
 *               'd': [':ddep'],})
 *   )
 * </pre>
 */
public final class SelectorList {
  private final Class<?> type;
  private final List<Object> elements;

  private SelectorList(Class<?> type, List<Object> elements) {
    this.type = type;
    this.elements = elements;
  }

  /**
   * Returns an ordered list of the elements in this expression. Each element may be a
   * native type or a select.
   */
  public List<Object> getElements() {
    return elements;
  }

  /**
   * Returns the native type contained by this expression.
   */
  private Class<?> getType() {
    return type;
  }

  /**
   * Creates a "wrapper" list that consists of a single select.
   */
  public static SelectorList of(SelectorValue selector) {
    return new SelectorList(selector.getType(), ImmutableList.<Object>of(selector));
  }

  /**
   * Creates a list that concatenates two values, where each value may be either a native
   * type or a select over that type.
   *
   * @throws EvalException if the values don't have the same underlying type
   */
  public static SelectorList concat(Location location, Object value1, Object value2)
      throws EvalException {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    Class<?> type1 = addValue(value1, builder);
    Class<?> type2 = addValue(value2, builder);
    if (type1 != type2) {
      throw new EvalException(location, "'+' operator applied to incompatible types");
    }
    return new SelectorList(type1, builder.build());
  }
  
  private static Class<?> addValue(Object value, ImmutableList.Builder<Object> builder) {
    if (value instanceof SelectorList) {
      SelectorList selectorList = (SelectorList) value;
      builder.addAll(selectorList.getElements());
      return selectorList.getType();
    } else if (value instanceof SelectorValue) {
      builder.add(value);
      return ((SelectorValue) value).getType();
    } else {
      builder.add(value);
      return value.getClass();
    }
  }

  @Override
  public String toString() {
    return Joiner.on(" + ").join(elements);
  }
}