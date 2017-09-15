// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options;

import com.google.common.base.Preconditions;
import com.google.common.collect.ListMultimap;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * The name and value of an option with additional metadata describing its priority, source, whether
 * it was set via an implicit dependency, and if so, by which other option.
 */
public class OptionValueDescription {
  private final OptionDefinition optionDefinition;
  private final boolean isDefaultValue;
  @Nullable private final String originalValueString;
  @Nullable private final Object value;
  @Nullable private final OptionPriority priority;
  @Nullable private final String source;
  @Nullable private final OptionDefinition implicitDependant;
  @Nullable private final OptionDefinition expandedFrom;

  private OptionValueDescription(
      OptionDefinition optionDefinition,
      boolean isDefaultValue,
      @Nullable String originalValueString,
      @Nullable Object value,
      @Nullable OptionPriority priority,
      @Nullable String source,
      @Nullable OptionDefinition implicitDependant,
      @Nullable OptionDefinition expandedFrom) {
    this.optionDefinition = optionDefinition;
    this.isDefaultValue = isDefaultValue;
    this.originalValueString = originalValueString;
    this.value = value;
    this.priority = priority;
    this.source = source;
    this.implicitDependant = implicitDependant;
    this.expandedFrom = expandedFrom;
  }

  public static OptionValueDescription newOptionValue(
      OptionDefinition optionDefinition,
      @Nullable String originalValueString,
      @Nullable Object value,
      @Nullable OptionPriority priority,
      @Nullable String source,
      @Nullable OptionDefinition implicitDependant,
      @Nullable OptionDefinition expandedFrom) {
    return new OptionValueDescription(
        optionDefinition,
        false,
        originalValueString,
        value,
        priority,
        source,
        implicitDependant,
        expandedFrom);
  }

  public static OptionValueDescription newDefaultValue(OptionDefinition optionDefinition) {
    return new OptionValueDescription(
        optionDefinition, true, null, null, OptionPriority.DEFAULT, null, null, null);
  }

  public OptionDefinition getOptionDefinition() {
    return optionDefinition;
  }

  public String getName() {
    return optionDefinition.getOptionName();
  }

  public String getOriginalValueString() {
    return originalValueString;
  }

  // Need to suppress unchecked warnings, because the "multiple occurrence"
  // options use unchecked ListMultimaps due to limitations of Java generics.
  @SuppressWarnings({"unchecked", "rawtypes"})
  public Object getValue() {
    if (isDefaultValue) {
      // If no value was present, we want the default value for this option.
      return optionDefinition.getDefaultValue();
    }
    if (getAllowMultiple() && value != null) {
      // Sort the results by option priority and return them in a new list.
      // The generic type of the list is not known at runtime, so we can't
      // use it here. It was already checked in the constructor, so this is
      // type-safe.
      List result = new ArrayList<>();
      ListMultimap realValue = (ListMultimap) value;
      for (OptionPriority priority : OptionPriority.values()) {
        // If there is no mapping for this key, this check avoids object creation (because
        // ListMultimap has to return a new object on get) and also an unnecessary addAll call.
        if (realValue.containsKey(priority)) {
          result.addAll(realValue.get(priority));
        }
      }
      return result;
    }
    return value;
  }

  /**
   * @return the priority of the thing that set this value for this flag
   */
  public OptionPriority getPriority() {
    return priority;
  }

  /**
   * @return the thing that set this value for this flag
   */
  public String getSource() {
    return source;
  }

  public OptionDefinition getImplicitDependant() {
    return implicitDependant;
  }

  public boolean isImplicitDependency() {
    return implicitDependant != null;
  }

  public OptionDefinition getExpansionParent() {
    return expandedFrom;
  }

  public boolean isExpansion() {
    return expandedFrom != null;
  }

  public boolean getAllowMultiple() {
    return optionDefinition.allowsMultiple();
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append("option '").append(optionDefinition.getOptionName()).append("' ");
    if (isDefaultValue) {
      result
          .append("set to its default value: '")
          .append(optionDefinition.getUnparsedDefaultValue())
          .append("'");
      return result.toString();
    } else {
      result.append("set to '").append(value).append("' ");
      result.append("with priority ").append(priority);
      if (source != null) {
        result.append(" and source '").append(source).append("'");
      }
      if (implicitDependant != null) {
        result.append(" implicitly by ");
      }
      return result.toString();
    }
  }

  // Need to suppress unchecked warnings, because the "multiple occurrence"
  // options use unchecked ListMultimaps due to limitations of Java generics.
  @SuppressWarnings({"unchecked", "rawtypes"})
  void addValue(OptionPriority addedPriority, Object addedValue) {
    Preconditions.checkState(optionDefinition.allowsMultiple());
    Preconditions.checkState(!isDefaultValue);
    ListMultimap optionValueList = (ListMultimap) value;
    if (addedValue instanceof List<?>) {
      optionValueList.putAll(addedPriority, (List<?>) addedValue);
    } else {
      optionValueList.put(addedPriority, addedValue);
    }
  }
}


