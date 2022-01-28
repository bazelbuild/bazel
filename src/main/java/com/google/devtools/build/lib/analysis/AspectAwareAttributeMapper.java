// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AbstractAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * An {@link AttributeMap} that supports queries on both a rule and its aspects.
 *
 * <p>When both the rule and aspect declare the same attribute, the aspect's value takes precedence.
 * This is because aspects expect access to the merged rule/attribute data (including possible
 * aspect overrides) while rules evaluate before aspects are attached.
 *
 * <p>Note that public aspect attributes must be strings that inherit the values of the equivalent
 * rule attributes. Only private (implicit) attributes can have different values.
 */
final class AspectAwareAttributeMapper extends AbstractAttributeMapper {
  private final AbstractAttributeMapper ruleAttributes;
  // Attribute name -> definition.
  private final ImmutableMap<String, Attribute> aspectAttributes;
  // Attribute name -> value. null values (which are valid) are excluded from this map. Use
  // aspectAttributes to check attribute existence.
  private final ImmutableMap<String, Object> aspectAttributeValues;

  public AspectAwareAttributeMapper(
      Rule rule,
      AbstractAttributeMapper ruleAttributes,
      ImmutableMap<String, Attribute> aspectAttributes) {
    super(rule);
    this.ruleAttributes = ruleAttributes;
    this.aspectAttributes = aspectAttributes;
    ImmutableMap.Builder<String, Object> valueBuilder = new ImmutableMap.Builder<>();
    for (Map.Entry<String, Attribute> aspectAttribute : aspectAttributes.entrySet()) {
      Attribute attribute = aspectAttribute.getValue();
      Object defaultValue = attribute.getDefaultValue(rule);
      Object attributeValue =
          attribute
              .getType()
              .cast(
                  defaultValue instanceof ComputedDefault
                      ? ((ComputedDefault) defaultValue).getDefault(ruleAttributes)
                      : defaultValue);
      if (attributeValue != null) {
        valueBuilder.put(aspectAttribute.getKey(), attributeValue);
      }
    }
    this.aspectAttributeValues = valueBuilder.buildOrThrow();
  }

  @Override
  public <T> T get(String attributeName, Type<T> type) {
    Attribute aspectAttribute = aspectAttributes.get(attributeName);
    if (aspectAttribute != null) {
      if (aspectAttribute.getType() != type) {
        throw new IllegalArgumentException(
            String.format(
                "attribute %s has type %s, not expected type %s",
                attributeName, aspectAttribute.getType(), type));
      }
      return type.cast(aspectAttributeValues.get(attributeName));
    }
    if (ruleAttributes.has(attributeName, type)) {
      return ruleAttributes.get(attributeName, type);
    }
    throw new IllegalArgumentException(
        String.format(
            "no attribute '%s' in either %s or its aspects",
            attributeName, ruleAttributes.getLabel()));
  }

  @Override
  public Iterable<String> getAttributeNames() {
    return ImmutableList.<String>builder()
        .addAll(ruleAttributes.getAttributeNames())
        .addAll(aspectAttributes.keySet())
        .build();
  }

  @Override
  public Type<?> getAttributeType(String attrName) {
    Attribute aspectAttribute = aspectAttributes.get(attrName);
    if (aspectAttribute != null) {
      return aspectAttribute.getType();
    }
    return ruleAttributes.getAttributeType(attrName);
  }

  @Override
  public Attribute getAttributeDefinition(String attrName) {
    Attribute aspectAttribute = aspectAttributes.get(attrName);
    if (aspectAttribute != null) {
      return aspectAttribute;
    }
    return ruleAttributes.getAttributeDefinition(attrName);
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    return ruleAttributes.isAttributeValueExplicitlySpecified(attributeName);
  }

  @Override
  public void visitLabels(DependencyFilter filter, BiConsumer<Attribute, Label> consumer) {
    ImmutableList.Builder<Attribute> combined = ImmutableList.builder();
    combined.addAll(rule.getAttributes());
    aspectAttributes.values().forEach(combined::add);
    visitLabels(combined.build(), filter, consumer);
  }

  @Override
  public <T> void visitLabels(Attribute attribute, Type<T> type, Type.LabelVisitor visitor) {
    Attribute aspectAttr = aspectAttributes.get(attribute.getName());
    if (aspectAttr != null) {
      T value = type.cast(aspectAttributeValues.get(attribute.getName()));
      if (value != null) { // null values are particularly possible for computed defaults.
        type.visitLabels(visitor, value, attribute);
      }
      return; // If both the aspect and rule have this attribute, the aspect instance overrides.
    }
    if (ruleAttributes.has(attribute.getName(), type)) {
      ruleAttributes.visitLabels(attribute, type, visitor);
    }
  }

  @Override
  public boolean has(String attrName) {
    if (ruleAttributes.has(attrName)) {
      return true;
    } else {
      return aspectAttributes.containsKey(attrName);
    }
  }

  @Override
  public <T> boolean has(String attrName, Type<T> type) {
    if (ruleAttributes.has(attrName, type)) {
      return true;
    } else {
      return aspectAttributes.containsKey(attrName)
          && aspectAttributes.get(attrName).getType() == type;
    }
  }
}
