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
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.syntax.Label;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * {@link AttributeMap} implementation that provides the ability to retrieve <i>all possible</i>
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
   *
   * <p>Note that we directly parse the selectors rather than just calling {@link #visitAttribute}
   * to iterate over all possible values. That's because {@link #visitAttribute} can grow
   * exponentially with respect to the number of selects (e.g. if an attribute uses three selects
   * with three conditions each, it can take nine possible values). So we want to avoid that code
   * path whenever actual value iteration isn't specifically needed.
   */
  @Override
  protected void visitLabels(Attribute attribute, AcceptsLabelAttribute observer) {
    visitLabels(attribute, true, observer);
  }

  private void visitLabels(Attribute attribute, boolean includeSelectKeys,
    AcceptsLabelAttribute observer) {
    Type<?> type = attribute.getType();
    Type.SelectorList<?> selectorList = getSelectorList(attribute.getName(), type);
    if (selectorList == null) {
      if (getComputedDefault(attribute.getName(), attribute.getType()) != null) {
        // Computed defaults are a special pain: we have no choice but to iterate through their
        // (computed) values and look for labels.
        for (Object value : visitAttribute(attribute.getName(), attribute.getType())) {
          if (value != null) {
            for (Label label : type.getLabels(value)) {
              observer.acceptLabelAttribute(label, attribute);
            }
          }
        }
      } else {
        super.visitLabels(attribute, observer);
      }
    } else {
      for (Type.Selector<?> selector : selectorList.getSelectors()) {
        for (Map.Entry<Label, ?> selectorEntry : selector.getEntries().entrySet()) {
          if (includeSelectKeys && !Type.Selector.isReservedLabel(selectorEntry.getKey())) {
            observer.acceptLabelAttribute(selectorEntry.getKey(), attribute);
          }
          for (Label value : type.getLabels(selectorEntry.getValue())) {
            observer.acceptLabelAttribute(value, attribute);
          }
        }
      }
    }
  }

  /**
   * Returns all labels reachable via the given attribute. If a label is listed multiple times,
   * each instance appears in the returned list.
   *
   * @param includeSelectKeys whether to include config_setting keys for configurable attributes
   */
  public List<Label> getReachableLabels(String attributeName, boolean includeSelectKeys) {
    final ImmutableList.Builder<Label> builder = ImmutableList.builder();
    visitLabels(getAttributeDefinition(attributeName), includeSelectKeys,
        new AcceptsLabelAttribute() {
          @Override
          public void acceptLabelAttribute(Label label, Attribute attribute) {
            builder.add(label);
          }
        });
    return builder.build();
  }

  /**
   * Returns the labels that might appear multiple times in the same attribute value.
   */
  public Set<Label> checkForDuplicateLabels(Attribute attribute) {
    String attrName = attribute.getName();
    Type<?> attrType = attribute.getType();

    Type.SelectorList<?> selectorList = getSelectorList(attribute.getName(), attrType);
    if (selectorList == null || selectorList.getSelectors().size() == 1) {
      // Three possible scenarios:
      //  1) Plain old attribute (no selects). Without selects, visitAttribute runs efficiently.
      //  2) Computed default, possibly depending on other attributes using select. In this case,
      //     visitAttribute might be inefficient. But we have no choice but to iterate over all
      //     possible values (since we have to compute them), so we take the efficiency hit.
      //  3) "attr = select({...})". With just a single select, visitAttribute runs efficiently.
      ImmutableSet.Builder<Label> duplicates = ImmutableSet.builder();
      for (Object value : visitAttribute(attrName, attrType)) {
        if (value != null) {
          duplicates.addAll(CollectionUtils.duplicatedElementsOf(
              ImmutableList.copyOf(attrType.getLabels(value))));
        }
      }
      return duplicates.build();
    } else {
      // Multiple selects concatenated together. It's expensive to iterate over every possible
      // value, so instead collect all labels across all the selects and check for duplicates.
      // This is overly strict, since this counts duplicates across values. We can presumably
      // relax this if necessary, but doing so would incur the value iteration expense this
      // code path avoids.
      return CollectionUtils.duplicatedElementsOf(getReachableLabels(attrName, false));
    }
  }

  /**
   * Returns a list of all possible values an attribute can take for this rule.
   *
   * <p>Note that when an attribute uses multiple selects, it can potentially take on many
   * values. So be cautious about unnecessarily relying on this method.
   */
  public <T> Iterable<T> visitAttribute(String attributeName, Type<T> type) {
    // If this attribute value is configurable, visit all possible values.
    Type.SelectorList<T> selectorList = getSelectorList(attributeName, type);
    if (selectorList != null) {
      ImmutableList.Builder<T> builder = ImmutableList.builder();
      visitConfigurableAttribute(selectorList.getSelectors(), new BoundSelectorPaths(), type,
          null, builder);
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
   * Determines all possible values a configurable attribute can take. Do not call this method
   * unless really necessary (see TODO comment inside).
   *
   * @param selectors the selectors that make up this attribute assignment (in order)
   * @param boundSelectorPaths paths that have already been chosen from previous selectors in an
   *     earlier recursive call of this method. For example, given
   *     <pre>cmd = select({':a': 'w', ':b': 'x'}) + select({':a': 'y', ':b': 'z'})</pre>
   *     the only possible values for <code>cmd</code> are <code>"wy"</code> and <code>"xz"</code>.
   *     This is because the selects have the same conditions, so whatever matches the first also
   *     matches the second. Note that this doesn't work for selects with overlapping but
   *     <i>different</i> key sets. That's because of key specialization (see
   *     {@link com.google.devtools.build.lib.analysis.ConfiguredAttributeMapper} - if the
   *     second select also included a condition <code>':c'</code> that includes both the flags
   *     in <code>':a'</code> and <code>':b'</code>, <code>':c'</code> would be chosen over
   *     them both.
   * @param type the type of this attribute
   * @param currentValueSoFar the partial value produced so far from earlier calls to this method
   * @param valuesBuilder output container for full values this attribute can take
   */
  private <T> void visitConfigurableAttribute(List<Type.Selector<T>> selectors,
      BoundSelectorPaths boundSelectorPaths, Type<T> type, T currentValueSoFar,
      ImmutableList.Builder<T> valuesBuilder) {
    // TODO(bazel-team): minimize or eliminate uses of this interface. It necessarily grows
    // exponentially with the number of selects in the attribute. Is that always necessary?
    // For example, dependency resolution just needs to know every possible label an attribute
    // might reference, but it doesn't need to know the exact combination of labels that make
    // up a value. This may be even less important for non-label values (e.g. strings), which
    // have no impact on the dependency structure.

    if (selectors.isEmpty()) {
      valuesBuilder.add(Preconditions.checkNotNull(currentValueSoFar));
    } else {
      Type.Selector<T> firstSelector = selectors.get(0);
      List<Type.Selector<T>> remainingSelectors = selectors.subList(1, selectors.size());

      Map<Label, T> firstSelectorEntries = firstSelector.getEntries();
      Label boundKey = boundSelectorPaths.getChosenKey(firstSelectorEntries.keySet());
      if (boundKey != null) {
        // If we've already followed some path from a previous selector with the same exact
        // conditions as this one, we only need to visit that path (since the same key will
        // match both selectors).
        T boundValue = firstSelectorEntries.get(boundKey);
        visitConfigurableAttribute(remainingSelectors, boundSelectorPaths, type,
                    currentValueSoFar == null
                        ? boundValue
                        : type.concat(ImmutableList.of(currentValueSoFar, boundValue)),
                    valuesBuilder);
      } else {
        // Otherwise, we need to iterate over all possible paths.
        for (Map.Entry<Label, T> selectorBranch : firstSelectorEntries.entrySet()) {
          // Bind this particular path for later selectors using the same conditions.
          boundSelectorPaths.bind(firstSelectorEntries.keySet(), selectorBranch.getKey());
          visitConfigurableAttribute(remainingSelectors, boundSelectorPaths, type,
              currentValueSoFar == null
                  ? selectorBranch.getValue()
                  : type.concat(ImmutableList.of(currentValueSoFar, selectorBranch.getValue())),
              valuesBuilder);
          // Unbind the path (so when we pop back up the recursive stack we can rebind it to new
          // values if we visit this selector again).
          boundSelectorPaths.unbind(firstSelectorEntries.keySet());
        }
      }
    }
  }

  /**
   * Helper class for {@link #visitConfigurableAttribute}. See that method's comments for more
   * details.
   */
  private static class BoundSelectorPaths {
    private final Map<Set<Label>, Label> bindings = new HashMap<>();

    /**
     * Binds the given config key set to the specified path. There should be no previous binding
     * for this key set.
     */
    public void bind(Set<Label> allKeys, Label chosenKey) {
      Preconditions.checkState(allKeys.contains(chosenKey));
      Verify.verify(bindings.put(allKeys, chosenKey) == null);
    }

    /**
     * Unbinds the given config key set.
     */
    public void unbind(Set<Label> allKeys) {
      Verify.verifyNotNull(bindings.remove(allKeys));
    }

    /**
     * Returns the key this config key set is bound to or null if no binding.
     */
    public Label getChosenKey(Set<Label> allKeys) {
      return bindings.get(allKeys);
    }
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
