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

import javax.annotation.Nullable;

/**
 * Base {@link AttributeMap} implementation providing direct, unmanipulated access to
 * underlying attribute data as stored within the Rule.
 *
 * <p>Any instantiable subclass should define a clear policy of what it does with this
 * data before exposing it to consumers.
 */
public abstract class AbstractAttributeMapper implements AttributeMap {

  private final Package pkg;
  private final RuleClass ruleClass;
  private final Label ruleLabel;
  private final AttributeContainer attributes;

  public AbstractAttributeMapper(Package pkg, RuleClass ruleClass, Label ruleLabel,
      AttributeContainer attributes) {
    this.pkg = pkg;
    this.ruleClass = ruleClass;
    this.ruleLabel = ruleLabel;
    this.attributes = attributes;
  }

  @Override
  public String getName() {
    return ruleLabel.getName();
  }

  @Override
  public Label getLabel() {
    return ruleLabel;
  }

  @Nullable
  @Override
  public <T> T get(String attributeName, Type<T> type) {
    int index = getIndexWithTypeCheck(attributeName, type);
    Object value = attributes.getAttributeValue(index);
    if (value instanceof Attribute.ComputedDefault) {
      value = ((Attribute.ComputedDefault) value).getDefault(this);
    }
    return type.cast(value);
  }

  /**
   * Returns the given attribute if it's a computed default, null otherwise.
   *
   * @throws IllegalArgumentException if the given attribute doesn't exist with the specified
   *         type. This happens whether or not it's a computed default.
   */
  protected <T> Attribute.ComputedDefault getComputedDefault(String attributeName, Type<T> type) {
    int index = getIndexWithTypeCheck(attributeName, type);
    Object value = attributes.getAttributeValue(index);
    if (value instanceof Attribute.ComputedDefault) {
      return (Attribute.ComputedDefault) value;
    } else {
      return null;
    }
  }

  @Override
  public Iterable<String> getAttributeNames() {
    ImmutableList.Builder<String> names = ImmutableList.builder();
    for (Attribute a : ruleClass.getAttributes()) {
      names.add(a.getName());
    }
    return names.build();
  }

  @Nullable
  @Override
  public Type<?> getAttributeType(String attrName) {
    Attribute attr = getAttributeDefinition(attrName);
    return attr == null ? null : attr.getType();
  }

  @Nullable
  @Override
  public Attribute getAttributeDefinition(String attrName) {
    return ruleClass.getAttributeByNameMaybe(attrName);
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    return attributes.isAttributeValueExplicitlySpecified(attributeName);
  }

  @Override
  public String getPackageDefaultHdrsCheck() {
    return pkg.getDefaultHdrsCheck();
  }

  @Override
  public Boolean getPackageDefaultObsolete() {
    return pkg.getDefaultObsolete();
  }

  @Override
  public Boolean getPackageDefaultTestOnly() {
    return pkg.getDefaultTestOnly();
  }

  @Override
  public String getPackageDefaultDeprecation() {
    return pkg.getDefaultDeprecation();
  }

  @Override
  public ImmutableList<String> getPackageDefaultCopts() {
    return pkg.getDefaultCopts();
  }

  @Override
  public void visitLabels(AcceptsLabelAttribute observer) {
    for (Attribute attribute : ruleClass.getAttributes()) {
      Type<?> type = attribute.getType();
      // TODO(bazel-team): This is incoherent: we shouldn't have to special-case these types
      // for our visitation policy. But this is the semantics the calling code requires. Audit
      // exactly which calling code expects what and clean up this interface.
      if (type == Type.OUTPUT || type == Type.OUTPUT_LIST
              || type == Type.NODEP_LABEL || type == Type.NODEP_LABEL_LIST) {
        continue;
      }
      for (Object value : visitAttribute(attribute.getName(), type)) {
        if (value == null) {
          // This is particularly possible for computed defaults.
          continue;
        }
        for (Label label : type.getLabels(value)) {
          observer.acceptLabelAttribute(label, attribute);
        }
      }
    }
  }

  /**
   * Implementations should provide policy-appropriate mappings when an attribute is requested in
   * the context of a rule visitation.
   */
  protected abstract <T> Iterable<T> visitAttribute(String attributeName, Type<T> type);

  /**
   * Returns a {@link Type.Selector} for the given attribute if the attribute is configurable
   * for this rule, null otherwise.
   *
   * @return a {@link Type.Selector} if the attribute takes the form
   *     "attrName = { 'a': value1_of_type_T, 'b': value2_of_type_T }") for this rule, null
   *     if it takes the form "attrName = value_of_type_T", null if it doesn't exist
   * @throws IllegalArgumentException if the attribute is configurable but of the wrong type
   */
  @Nullable
  protected <T> Type.Selector<T> getSelector(String attributeName, Type<T> type) {
    Integer index = ruleClass.getAttributeIndex(attributeName);
    if (index == null) {
      return null;
    }
    Object attrValue = attributes.getAttributeValue(index);
    if (!(attrValue instanceof Type.Selector<?>)) {
      return null;
    }
    if (((Type.Selector<?>) attrValue).getOriginalType() != type) {
      throw new IllegalArgumentException("Attribute " + attributeName
          + " is not of type " + type + " in rule " + ruleLabel.getName());
    }
    return (Type.Selector<T>) attrValue;
  }

  /**
   * Returns the index of the specified attribute, if its type is 'type'. Throws
   * an exception otherwise.
   */
  private int getIndexWithTypeCheck(String attrName, Type<?> type) {
    Integer index = ruleClass.getAttributeIndex(attrName);
    if (index == null) {
      throw new IllegalArgumentException("No such attribute " + attrName
          + " in rule " + ruleLabel.getName());
    }
    Attribute attr = ruleClass.getAttribute(index);
    if (attr.getType() != type) {
      throw new IllegalArgumentException("Attribute " + attrName
          + " is not of type " + type + " in rule " + ruleLabel.getName());
    }
    return index;
  }

  /**
   * Helper routine that just checks the given attribute has the given type for this rule and
   * throws an IllegalException if not.
   */
  protected void checkType(String attrName, Type<?> type) {
    getIndexWithTypeCheck(attrName, type);
  }

  @Override
  public boolean has(String attrName, Type<?> type) {
    Attribute attribute = ruleClass.getAttributeByNameMaybe(attrName);
    return attribute != null && attribute.getType() == type;
  }
}
