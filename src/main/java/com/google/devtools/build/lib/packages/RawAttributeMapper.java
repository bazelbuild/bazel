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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.Label;

/**
 * {@link AttributeMap} implementation that returns raw attribute information as contained
 * within a {@link Rule}. In particular, configurable attributes of the form
 * { config1: "value1", config2: "value2" } are passed through without being resolved to a
 * final value.
 */
public class RawAttributeMapper extends AbstractAttributeMapper {
  RawAttributeMapper(Package pkg, RuleClass ruleClass, Label ruleLabel,
      AttributeContainer attributes) {
    super(pkg, ruleClass, ruleLabel, attributes);
  }

  public static RawAttributeMapper of(Rule rule) {
    return new RawAttributeMapper(rule.getPackage(), rule.getRuleClassObject(), rule.getLabel(),
        rule.getAttributeContainer());
  }

  @Override
  protected <T> Iterable<T> visitAttribute(String attributeName, Type<T> type) {
    T value = get(attributeName, type);
    return value == null ? ImmutableList.<T>of() : ImmutableList.of(value);
  }

  /**
   * Returns true if the given attribute is configurable for this rule instance, false
   * otherwise.
   */
  public <T> boolean isConfigurable(String attributeName, Type<T> type) {
    return getSelector(attributeName, type) != null;
  }

  /**
   * If the attribute is configurable for this rule instance, returns its configuration
   * keys. Else returns an empty list.
   */
  public <T> Iterable<Label> getConfigurabilityKeys(String attributeName, Type<T> type) {
    Type.Selector<T> selector = getSelector(attributeName, type);
    return selector == null ? ImmutableList.<Label>of() : selector.getEntries().keySet();
  }
}
