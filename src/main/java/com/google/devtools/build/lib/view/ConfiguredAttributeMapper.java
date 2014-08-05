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
package com.google.devtools.build.lib.view;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

/**
 * {@link AttributeMap} implementation that binds a rule's attribute as follows:
 *
 * <ol>
 *   <li>If the attribute is selectable (i.e. its BUILD declaration is of the form
 *   "attr = { config1: "value1", "config2: "value2", ... }", returns the subset of values
 *   chosen by the current configuration in accordance with Bazel's documented policy on
 *   configurable attribute selection.
 *   <li>If the attribute is not selectable (i.e. its value is static), returns that value with
 *   no additional processing.
 *
 * <p>Example usage:
 * <pre>
 *   Label fooLabel = ConfiguredAttributeMapper.of(ruleConfiguredTarget).get("foo", Type.LABEL);
 * </pre>
 * </ol>
 */
public class ConfiguredAttributeMapper extends AbstractAttributeMapper {

  private final BuildConfiguration configuration;

  private ConfiguredAttributeMapper(Rule rule, BuildConfiguration configuration) {
    super(Preconditions.checkNotNull(rule).getPackage(), rule.getRuleClassObject(), rule.getLabel(),
        rule.getAttributeContainer());
    this.configuration = Preconditions.checkNotNull(configuration);
  }

  /**
   * "Do-it-all" constructor that just needs a {@link RuleConfiguredTarget}.
   */
  public static ConfiguredAttributeMapper of(RuleConfiguredTarget ct) {
    return new ConfiguredAttributeMapper(ct.getTarget(), ct.getConfiguration());
  }

  public static ConfiguredAttributeMapper of(Rule rule, BuildConfiguration configuration) {
    return new ConfiguredAttributeMapper(rule, configuration);
  }

  @Override
  public <T> T get(String attributeName, Type<T> type) {
    Type.Selector<T> selector = getSelector(attributeName, type);
    if (selector != null) {
      Label selectionKey = configuration.getAttributeSelector();
      T value = selectionKey != null ? selector.getEntries().get(selectionKey) : null;
      if (value == null) {
        value = selector.getDefault();
      }
      return value;
    } else {
      return super.get(attributeName, type);
    }
  }

  @Override
  protected <T> Iterable<T> visitAttribute(String attributeName, Type<T> type) {
    T value = get(attributeName, type);
    return value == null ? ImmutableList.<T>of() : ImmutableList.of(value);
  }
}
