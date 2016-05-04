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

import com.google.common.base.Function;
import com.google.devtools.build.lib.events.Location;

import java.util.BitSet;

/**
 * Provides attribute setting and retrieval for a Rule. Encapsulating attribute access
 * here means it can be passed around independently of the Rule itself. In particular,
 * it can be consumed by independent {@link AttributeMap} instances that can apply
 * varying kinds of logic for determining the "value" of an attribute. For example,
 * a configurable attribute's "value" may be a { config --> value } dictionary
 * or a configuration-bound lookup on that dictionary, depending on the context in
 * which it's requested.
 *
 * <p>This class provides the lowest-level access to attribute information. It is *not*
 * intended to be a robust public interface, but rather just an input to {@link AttributeMap}
 * instances. Use those instances for all domain-level attribute access.
 */
public class AttributeContainer {

  private final RuleClass ruleClass;

  // Attribute values, keyed by attribute index:
  private final Object[] attributeValues;

  // Whether an attribute value has been set explicitly in the BUILD file, keyed by attribute index.
  private final BitSet attributeValueExplicitlySpecified;

  // Attribute locations, keyed by attribute index:
  private final Location[] attributeLocations;

  /**
   * Create a container for a rule of the given rule class.
   */
  public AttributeContainer(RuleClass ruleClass) {
   this(ruleClass, new Location[ruleClass.getAttributeCount()]);
  }
  
  AttributeContainer(RuleClass ruleClass, Location[] locations) {
    this.ruleClass = ruleClass;
    this.attributeValues = new Object[ruleClass.getAttributeCount()];
    this.attributeValueExplicitlySpecified = new BitSet(ruleClass.getAttributeCount());
    this.attributeLocations = locations;
  }

  /**
   * Returns an attribute value by instance, or null on no match.
   */
  public Object getAttr(Attribute attribute) {
    return getAttr(attribute.getName());
  }

  /**
   * Returns an attribute value by name, or null on no match.
   */
  public Object getAttr(String attrName) {
    Integer idx = ruleClass.getAttributeIndex(attrName);
    return idx != null ? attributeValues[idx] : null;
  }

  /**
   * Returns true iff the given attribute exists for this rule and its value
   * is explicitly set in the BUILD file (as opposed to its default value).
   */
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    return isAttributeValueExplicitlySpecified(attribute.getName());
  }

  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    Integer idx = ruleClass.getAttributeIndex(attributeName);
    return idx != null && attributeValueExplicitlySpecified.get(idx);
  }

  /**
   * Returns the location of the attribute definition for this rule, or null if not found.
   */
  public Location getAttributeLocation(String attrName) {
    Integer idx = ruleClass.getAttributeIndex(attrName);
    return idx != null ? attributeLocations[idx] : null;
  }

  Object getAttributeValue(int index) {
    return attributeValues[index];
  }

  void setAttributeValue(Attribute attribute, Object value, boolean explicit) {
    Integer index = ruleClass.getAttributeIndex(attribute.getName());
    attributeValues[index] = value;
    attributeValueExplicitlySpecified.set(index, explicit);
  }

  void setAttributeValueByName(String attrName, Object value) {
    Integer index = ruleClass.getAttributeIndex(attrName);
    attributeValues[index] = value;
    attributeValueExplicitlySpecified.set(index);
  }

  void setAttributeLocation(int attrIndex, Location location) {
    attributeLocations[attrIndex] = location;
  }

  void setAttributeLocation(Attribute attribute, Location location) {
    Integer index = ruleClass.getAttributeIndex(attribute.getName());
    attributeLocations[index] = location;
  }

  public static final Function<RuleClass, AttributeContainer> ATTRIBUTE_CONTAINER_FACTORY =
      new Function<RuleClass, AttributeContainer>() {
        @Override
        public AttributeContainer apply(RuleClass ruleClass) {
          return new AttributeContainer(ruleClass);
        }
      };
}
