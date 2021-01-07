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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.BuildType.Selector;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * {@link AttributeMap} implementation that returns raw attribute information as contained within
 * a {@link Rule} via {@link #getRawAttributeValue}. In particular, configurable attributes
 * of the form { config1: "value1", config2: "value2" } are passed through without being resolved
 * to a final value when obtained via that method.
 */
public class RawAttributeMapper extends AbstractAttributeMapper {

  RawAttributeMapper(Rule rule) {
    super(rule);
  }

  public static RawAttributeMapper of(Rule rule) {
    return new RawAttributeMapper(rule);
  }

  /**
   * Variation of {@link #get} that merges the values of configurable lists together (with
   * duplicates removed).
   *
   * <p>For example, given:
   * <pre>
   *   attr = select({
   *       ':condition1': [A, B, C],
   *       ':condition2': [C, D]
   *       }),
   * </pre>
   * this returns the value <code>[A, B, C, D]</code>.
   *
   * <p>If the attribute isn't configurable (e.g. <code>attr = [A, B]</code>), returns
   * its raw value.
   *
   * <p>Throws an {@link IllegalStateException} if the attribute isn't a list type.
   */
  @Nullable
  public <T> Collection<T> getMergedValues(String attributeName, Type<List<T>> type) {
    Preconditions.checkState(type instanceof Type.ListType);
    if (!isConfigurable(attributeName)) {
      return get(attributeName, type);
    }

    ImmutableSet.Builder<T> mergedValues = ImmutableSet.builder();
    for (Selector<List<T>> selector : getSelectorList(attributeName, type).getSelectors()) {
      for (List<T> configuredList : selector.getEntries().values()) {
        mergedValues.addAll(configuredList);
      }
    }
    return mergedValues.build();
  }

  /**
   * If the attribute is configurable for this rule instance, returns its configuration
   * keys. Else returns an empty list.
   */
  public <T> Iterable<Label> getConfigurabilityKeys(String attributeName, Type<T> type) {
    SelectorList<T> selectorList = getSelectorList(attributeName, type);
    if (selectorList == null) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Label> builder = ImmutableList.builder();
    for (Selector<T> selector : selectorList.getSelectors()) {
      builder.addAll(selector.getEntries().keySet());
    }
    return builder.build();
  }

  /**
   * See {@link #getRawAttributeValue(Rule, Attribute)}.
   *
   * <p>{@param attrName} must be the name of an {@link Attribute} defined by the {@param rule}'s
   * {@link RuleClass}.
   */
  @Nullable
  public Object getRawAttributeValue(Rule rule, String attrName) {
    Attribute attr =
        Preconditions.checkNotNull(getAttributeDefinition(attrName), "%s %s", rule, attrName);
    return getRawAttributeValue(rule, attr);
  }

  /**
   * Returns the object associated with the {@param rule}'s {@param attr}.
   *
   * <p>Handles the special case of the "visibility" attribute by returning {@param rule}'s {@link
   * RuleVisibility}'s declared labels.
   *
   * <p>The returned object will be a {@link SelectorList} if the attribute value contains a
   * {@code select(...)} expression.
   *
   * <p>The returned object will be a {@link ComputedDefault} if the rule doesn't explicitly
   * declare an attribute value and the rule's class provides a computed default for it.
   *
   * <p>Otherwise, the returned object will be the type declared by the {@param attr}, or {@code
   * null}.
   */
  @Nullable
  public Object getRawAttributeValue(Rule rule, Attribute attr) {
    // This special case for the visibility attribute is needed because its value is replaced
    // with an empty list during package loading if it is public or private in order not to visit
    // the package called 'visibility'.
    if (attr.getName().equals("visibility")) {
      return rule.getVisibility().getDeclaredLabels();
    }

    // If the attribute value contains one or more select(...) expressions, then return
    // the SelectorList object representing those expressions.
    SelectorList<?> selectorList = getSelectorList(attr.getName(), attr.getType());
    if (selectorList != null) {
      return selectorList;
    }

    // If the attribute value is not explicitly declared, and the rule class provides a computed
    // default value for it, then we should return the ComputedDefault object.
    //
    // We check for the existence of a ComputedDefault value because AbstractAttributeMapper#get
    // returns either an explicitly declared attribute value or the result of evaluating the
    // computed default function, but does not specify which one it is.
    ComputedDefault computedDefault = getComputedDefault(attr.getName(), attr.getType());
    if (computedDefault != null) {
      return computedDefault;
    }
    return get(attr.getName(), attr.getType());
  }
}
