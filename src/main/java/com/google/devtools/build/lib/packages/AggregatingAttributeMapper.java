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

import java.util.Map;

/**
 * {@link AttributeMap} implementation that provides the ability to retrieve *all possible*
 * values an attribute might take.
 */
public class AggregatingAttributeMapper extends AbstractAttributeMapper {

  public AggregatingAttributeMapper(Rule rule) {
    super(rule.getPackage(), rule.getRuleClassObject(), rule.getLabel(),
        rule.getAttributeContainer());
  }

  /**
   * Returns a list of all possible values an attribute can take for this rule.
   */
  @Override
  protected <T> Iterable<T> visitAttribute(String attributeName, Type<T> type) {
    Type.Selector<T> selector = getSelector(attributeName, type);
    if (selector != null) {
      ImmutableList.Builder<T> builder = ImmutableList.builder();
      for (Map.Entry<String, T> entry : selector.getEntries().entrySet()) {
        builder.add(entry.getValue());
      }
      return builder.build();
    } else {
      T value = get(attributeName, type);
      return value == null ? ImmutableList.<T>of() : ImmutableList.of(value);
    }
  }
}
