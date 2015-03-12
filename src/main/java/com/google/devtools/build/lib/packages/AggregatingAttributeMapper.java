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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.Label;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * {@link AttributeMap} implementation that provides the ability to retrieve *all possible*
 * values an attribute might take.
 */
public class AggregatingAttributeMapper extends AbstractAttributeMapper {

  /**
   * Store for all of this rule's attributes that are non-configurable. These are
   * unconditionally  available to computed defaults no matter what dependencies
   * they've declared.
   */
  private final List<String> nonconfigurableAttributes;

  private AggregatingAttributeMapper(Rule rule) {
    super(rule.getPackage(), rule.getRuleClassObject(), rule.getLabel(),
        rule.getAttributeContainer());

    ImmutableList.Builder<String> nonconfigurableAttributesBuilder = ImmutableList.builder();
    for (Attribute attr : rule.getAttributes()) {
      if (!attr.isConfigurable()) {
        nonconfigurableAttributesBuilder.add(attr.getName());
      }
    }
    nonconfigurableAttributes = nonconfigurableAttributesBuilder.build();
  }

  public static AggregatingAttributeMapper of(Rule rule) {
    return new AggregatingAttributeMapper(rule);
  }

  /**
   * Override that also visits the rule's configurable attribute keys (which are
   * themselves labels).
   */
  @Override
  public void visitLabels(AcceptsLabelAttribute observer) {
    super.visitLabels(observer);
    for (String attrName : getAttributeNames()) {
      Attribute attribute = getAttributeDefinition(attrName);
      Type.Selector<?> selector = getSelector(attrName, attribute.getType());
      if (selector != null) {
        for (Label configLabel : selector.getEntries().keySet()) {
          if (!Type.Selector.isReservedLabel(configLabel)) {
            observer.acceptLabelAttribute(configLabel, attribute);
          }
        }
      }
    }
  }

  /**
   * Returns a list of all possible values an attribute can take for this rule.
   */
  @Override
  public <T> Iterable<T> visitAttribute(String attributeName, Type<T> type) {
    // If this attribute value is configurable, visit all possible values.
    Type.Selector<T> selector = getSelector(attributeName, type);
    if (selector != null) {
      ImmutableList.Builder<T> builder = ImmutableList.builder();
      for (Map.Entry<Label, T> entry : selector.getEntries().entrySet()) {
        builder.add(entry.getValue());
      }
      return builder.build();
    }

    // If this attribute is a computed default, feed it all possible value combinations of
    // its declared dependencies and return all computed results. For example, if this default
    // uses attributes x and y, x can configurably be x1 or x2, and y can configurably be y1
    // or y1, then compute default values for the (x1,y1), (x1,y2), (x2,y1), and (x2,y2) cases.
    Attribute.ComputedDefault computedDefault = getComputedDefault(attributeName, type);
    if (computedDefault != null) {
      // This will hold every (value1, value2, ..) combination of the declared dependencies.
      List<Map<String, Object>> depMaps = new LinkedList<>();
      // Collect those combinations.
      mapDepsForComputedDefault(computedDefault.dependencies(), depMaps,
          ImmutableMap.<String, Object>of());
      List<T> possibleValues = new ArrayList<>(); // Not ImmutableList.Builder: values may be null.
      // For each combination, call getDefault on a specialized AttributeMap providing those values.
      for (Map<String, Object> depMap : depMaps) {
        possibleValues.add(type.cast(computedDefault.getDefault(mapBackedAttributeMap(depMap))));
      }
      return possibleValues;
    }

    // For any other attribute, just return its direct value.
    T value = get(attributeName, type);
    return value == null ? ImmutableList.<T>of() : ImmutableList.of(value);
  }

  /**
   * Given (possibly configurable) attributes that a computed default depends on, creates an
   * {attrName -> attrValue} map for every possible combination of those attribute values and
   * returns a list of all the maps. This defines the complete dependency space that can affect
   * the computed default's values.
   *
   * <p>For example, given dependencies x and y, which might respectively have values x1, x2 and
   * y1, y2, this returns:
   * <pre>
   *   [
   *    {x: x1, y: y1},
   *    {x: x1, y: y2},
   *    {x: x2, y: y1},
   *    {x: x2, y: y2}
   *   ]
   * </pre>
   *
   * @param depAttributes the names of the attributes this computed default depends on
   * @param mappings the list of {attrName --> attrValue} maps defining the computed default's
   *                 dependency space. This is where this method's results are written.
   * @param currentMap a (possibly non-empty) map to add {attrName --> attrValue}
   *                   entries to. Outside callers can just pass in an empty map.
   */
  private void mapDepsForComputedDefault(List<String> depAttributes,
      List<Map<String, Object>> mappings, Map<String, Object> currentMap) {
    // Because this method uses exponential time/space on the number of inputs, keep the
    // maximum number of inputs conservatively small.
    Preconditions.checkState(depAttributes.size() <= 2);

    if (depAttributes.isEmpty()) {
      // Recursive base case: store whatever's already been populated in currentMap.
      mappings.add(currentMap);
      return;
    }

    // Take the first attribute in the dependency list and iterate over all its values. For each
    // value x, copy currentMap with the additional entry { firstAttrName: x }, then feed
    // this recursively into a subcall over all remaining dependencies. This recursively
    // continues until we run out of values.
    String firstAttribute = depAttributes.get(0);
    for (Object value : visitAttribute(firstAttribute, getAttributeType(firstAttribute))) {
      Map<String, Object> newMap = new HashMap<>();
      newMap.putAll(currentMap);
      newMap.put(firstAttribute, value);
      mapDepsForComputedDefault(depAttributes.subList(1, depAttributes.size()), mappings, newMap);
    }
  }

  /**
   * A custom {@link AttributeMap} that reads attribute values from the given Map. All
   * non-configurable attributes are also readable. Any attempt to read an attribute
   * that's not in one of these two cases triggers an IllegalArgumentException.
   */
  private AttributeMap mapBackedAttributeMap(final Map<String, Object> directMap) {
    final AggregatingAttributeMapper owner = AggregatingAttributeMapper.this;
    return new AttributeMap() {

      @Override
      public <T> T get(String attributeName, Type<T> type) {
        owner.checkType(attributeName, type);
        if (nonconfigurableAttributes.contains(attributeName)) {
          return owner.get(attributeName, type);
        }
        if (!directMap.containsKey(attributeName)) {
          throw new IllegalArgumentException("attribute \"" + attributeName
              + "\" isn't available in this computed default context");
        }
        return type.cast(directMap.get(attributeName));
      }

      @Override public String getName() { return owner.getName(); }
      @Override public Label getLabel() { return owner.getLabel(); }
      @Override public Iterable<String> getAttributeNames() {
        return ImmutableList.<String>builder()
            .addAll(directMap.keySet()).addAll(nonconfigurableAttributes).build();
      }
      @Override
      public void visitLabels(AcceptsLabelAttribute observer) { owner.visitLabels(observer); }
      @Override
      public String getPackageDefaultHdrsCheck() { return owner.getPackageDefaultHdrsCheck(); }
      @Override
      public Boolean getPackageDefaultObsolete() { return owner.getPackageDefaultObsolete(); }
      @Override
      public Boolean getPackageDefaultTestOnly() { return owner.getPackageDefaultTestOnly(); }
      @Override
      public String getPackageDefaultDeprecation() { return owner.getPackageDefaultDeprecation(); }
      @Override
      public ImmutableList<String> getPackageDefaultCopts() {
        return owner.getPackageDefaultCopts();
      }
      @Nullable @Override
      public Type<?> getAttributeType(String attrName) { return owner.getAttributeType(attrName); }
      @Nullable @Override  public Attribute getAttributeDefinition(String attrName) {
        return owner.getAttributeDefinition(attrName);
      }
      @Override public boolean isAttributeValueExplicitlySpecified(String attributeName) {
        return owner.isAttributeValueExplicitlySpecified(attributeName);
      }
      @Override
      public boolean has(String attrName, Type<?> type) { return owner.has(attrName, type); }
    };
  }
}
