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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.PackageArgs;
import com.google.devtools.build.lib.packages.Type;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * An {@link AttributeMap} that supports attribute type queries on both a rule and its aspects and
 * attribute value queries on the rule.
 *
 * <p>An attribute type query is anything accessible from {@link Attribute} (i.e. anything about how
 * the attribute is integrated into the {@link com.google.devtools.build.lib.packages.RuleClass}).
 * An attribute value query is anything related to the actual value an attribute takes.
 *
 * <p>For example, given {@code deps = [":adep"]}, checking that {@code deps} exists or that it's
 * type is {@link com.google.devtools.build.lib.packages.BuildType#LABEL_LIST} are type queries.
 * Checking that its value is explicitly set in the BUILD File or that its value {@code [":adep"]}
 * are value queries.
 *
 * <p>Value queries on aspect attributes trigger {@link UnsupportedOperationException}.
 */
class AspectAwareAttributeMapper implements AttributeMap {
  private final AttributeMap ruleAttributes;
  private final ImmutableMap<String, Attribute> aspectAttributes;

  public AspectAwareAttributeMapper(AttributeMap ruleAttributes,
      ImmutableMap<String, Attribute> aspectAttributes) {
    this.ruleAttributes = ruleAttributes;
    this.aspectAttributes = aspectAttributes;
  }

  @Override
  public String getName() {
    return ruleAttributes.getName();
  }

  @Override
  public Label getLabel() {
    return ruleAttributes.getLabel();
  }

  @Override
  public <T> T get(String attributeName, Type<T> type) {
    if (ruleAttributes.has(attributeName, type)) {
      return ruleAttributes.get(attributeName, type);
    } else {
      Attribute attribute = aspectAttributes.get(attributeName);
      if (attribute == null) {
        throw new IllegalArgumentException(String.format(
            "no attribute '%s' in either %s or its aspects",
            attributeName, ruleAttributes.getLabel()));
      } else if (attribute.getType() != type) {
        throw new IllegalArgumentException(String.format(
            "attribute %s has type %s, not expected type %s",
            attributeName, attribute.getType(), type));
      } else {
        throw new UnsupportedOperationException(
            String.format(
                "Attribute '%s' comes from an aspect. "
                    + "Value retrieval for aspect attributes is not supported.",
                attributeName));
      }
    }
  }

  @Override
  public boolean isConfigurable(String attributeName) {
    return ruleAttributes.isConfigurable(attributeName);
  }

  @Override
  public Iterable<String> getAttributeNames() {
    return Iterables.concat(ruleAttributes.getAttributeNames(), aspectAttributes.keySet());
  }

  @Override
  @Nullable
  public Type<?> getAttributeType(String attrName) {
    Type<?> type = ruleAttributes.getAttributeType(attrName);
    if (type != null) {
      return type;
    } else {
      Attribute attribute = aspectAttributes.get(attrName);
      return attribute != null ? attribute.getType() : null;
    }
  }

  @Override
  public Attribute getAttributeDefinition(String attrName) {
    Attribute attribute = ruleAttributes.getAttributeDefinition(attrName);
    if (attribute != null) {
      return attribute;
    } else {
      return aspectAttributes.get(attrName);
    }
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    return ruleAttributes.isAttributeValueExplicitlySpecified(attributeName);
  }

  @Override
  public void visitAllLabels(BiConsumer<Attribute, Label> consumer) {
    throw new UnsupportedOperationException("rule + aspects label visition is not supported");
  }

  @Override
  public void visitLabels(String attributeName, Consumer<Label> consumer) {
    throw new UnsupportedOperationException("rule + aspects label visition is not supported");
  }

  @Override
  public void visitLabels(DependencyFilter filter, BiConsumer<Attribute, Label> consumer) {
    throw new UnsupportedOperationException("rule + aspects label visition is not supported");
  }

  @Override
  public PackageArgs getPackageArgs() {
    return ruleAttributes.getPackageArgs();
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
