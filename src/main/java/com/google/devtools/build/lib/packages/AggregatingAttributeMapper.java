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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.packages.Attribute.ComputationLimiter;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.BuildType.Selector;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.packages.Type.ListType;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/**
 * {@link AttributeMap} implementation that provides the ability to retrieve <i>all possible</i>
 * values an attribute might take.
 */
public class AggregatingAttributeMapper extends AbstractAttributeMapper {

  private AggregatingAttributeMapper(Rule rule) {
    super(rule);
  }

  public static AggregatingAttributeMapper of(Rule rule) {
    return new AggregatingAttributeMapper(rule);
  }

  /**
   * Returns all of this rule's attributes that are non-configurable. These are unconditionally
   * available to computed defaults no matter what dependencies they've declared.
   */
  private List<String> getNonConfigurableAttributes() {
    return rule.getRuleClassObject().getNonConfigurableAttributes();
  }

  /**
   * Override that also visits the rule's configurable attribute keys (which are themselves labels).
   *
   * <p>This method directly parses each selector, vs. calling {@link #visitAttribute} to iterate
   * over all possible values. The latter has dangerous efficiency consequences, as discussed in
   * {@link #visitAttribute}'s documentation. So we want to avoid that code path when possible.
   */
  @Override
  <T> void visitLabels(Attribute attribute, Type<T> type, Type.LabelVisitor visitor) {
    visitLabels(attribute, type, /*includeSelectKeys=*/ true, visitor);
  }

  @SuppressWarnings("unchecked")
  private <T> void visitLabels(
      Attribute attribute, Type<T> type, boolean includeSelectKeys, Type.LabelVisitor visitor) {
    String name = attribute.getName();

    // The only way for LabelClass.NONE to contain labels is in select keys.
    if (type.getLabelClass() == LabelClass.NONE) {
      if (includeSelectKeys && attribute.isConfigurable()) {
        SelectorList<T> selectorList = getSelectorList(name, type);
        if (selectorList != null) {
          visitLabelsInSelect(
              selectorList,
              attribute,
              type,
              visitor,
              /*includeKeys=*/ true,
              /*includeValues=*/ false);
        }
      }
      return;
    }

    Object rawVal = rule.getAttr(name, type);
    if (rawVal instanceof SelectorList) {
      visitLabelsInSelect(
          (SelectorList<T>) rawVal,
          attribute,
          type,
          visitor,
          includeSelectKeys,
          /*includeValues=*/ true);
    } else if (rawVal instanceof ComputedDefault) {
      // Computed defaults are a special pain: we have no choice but to iterate through their
      // (computed) values and look for labels.
      for (T value : ((ComputedDefault) rawVal).getPossibleValues(type, rule)) {
        if (value != null) {
          type.visitLabels(visitor, value, attribute);
        }
      }
    } else {
      T value = getFromRawAttributeValue(rawVal, name, type);
      if (value != null) {
        type.visitLabels(visitor, value, attribute);
      }
    }
  }

  private static <T> void visitLabelsInSelect(
      SelectorList<T> selectorList,
      Attribute attribute,
      Type<T> type,
      Type.LabelVisitor visitor,
      boolean includeKeys,
      boolean includeValues) {
    for (Selector<T> selector : selectorList.getSelectors()) {
      for (Map.Entry<Label, T> selectorEntry : selector.getEntries().entrySet()) {
        if (includeKeys && !Selector.isReservedLabel(selectorEntry.getKey())) {
          visitor.visit(selectorEntry.getKey(), attribute);
        }
        if (includeValues) {
          T value =
              selector.isValueSet(selectorEntry.getKey())
                  ? selectorEntry.getValue()
                  : type.cast(attribute.getDefaultValue(null));
          type.visitLabels(visitor, value, attribute);
        }
      }
    }
  }

  /**
   * Returns all labels reachable via the given attribute, with duplicate instances removed.
   *
   * <p>Use this interface over {@link #visitAttribute} whenever possible, since the latter has
   * efficiency problems discussed in that method's documentation.
   *
   * @param includeSelectKeys whether to include config_setting keys for configurable attributes
   */
  public ImmutableSet<Label> getReachableLabels(String attributeName, boolean includeSelectKeys) {
    Attribute attribute = getAttributeDefinition(attributeName);
    ImmutableSet.Builder<Label> builder = ImmutableSet.builder();
    visitLabels(
        attribute, attribute.getType(), includeSelectKeys, (label, attr) -> builder.add(label));
    return builder.build();
  }

  /** Returns the labels that appear multiple times in the same attribute value. */
  @SuppressWarnings("unchecked")
  public Set<Label> checkForDuplicateLabels(Attribute attribute) {
    Type<List<Label>> attrType = BuildType.LABEL_LIST;
    checkArgument(attribute.getType() == attrType, "Not a label list type: %s", attribute);
    String attrName = attribute.getName();
    Object rawVal = rule.getAttr(attrName, attrType);

    // Plain old attribute (no selects).
    if (!(rawVal instanceof SelectorList)) {
      return checkForDuplicateLabels(
          visitRawNonConfigurableAttributeValue(rawVal, attrName, attrType));
    }

    List<Selector<List<Label>>> selectors = ((SelectorList<List<Label>>) rawVal).getSelectors();

    // "attr = select({...})" with just a single select.
    if (selectors.size() == 1) {
      return checkForDuplicateLabels(selectors.get(0).getEntries().values());
    }

    // Multiple selects concatenated together. It's expensive to iterate over every possible
    // permutation of values, so instead check for duplicates within a single select branch while
    // also collecting all labels for a cross-select duplicate check at the end. This is overly
    // strict, since this counts values present in mutually exclusive select branches. We can
    // presumably relax this if necessary, but doing so would incur some of the expense this code
    // path avoids.
    ImmutableSet.Builder<Label> duplicates = null;
    List<Label> combinedLabels = new ArrayList<>(); // Labels that appear across all selectors.
    for (Selector<List<Label>> selector : selectors) {
      // Labels within a single selector. It's okay for there to be duplicates as long as
      // they're in different selector paths (since only one path can actually get chosen).
      Set<Label> selectorLabels = new LinkedHashSet<>();
      for (List<Label> labelsInSelectorValue : selector.getEntries().values()) {
        // Duplicates within a single select branch are not okay.
        duplicates = addDuplicateLabels(duplicates, labelsInSelectorValue);
        selectorLabels.addAll(labelsInSelectorValue);
      }
      combinedLabels.addAll(selectorLabels);
    }
    duplicates = addDuplicateLabels(duplicates, combinedLabels);

    return duplicates == null ? ImmutableSet.of() : duplicates.build();
  }

  private static Set<Label> checkForDuplicateLabels(Collection<List<Label>> possibleLabels) {
    switch (possibleLabels.size()) {
      case 0:
        return ImmutableSet.of();
      case 1:
        List<Label> onlyPossibility =
            possibleLabels instanceof List
                ? ((List<List<Label>>) possibleLabels).get(0) // Avoid overhead of list iterator.
                : possibleLabels.iterator().next();
        return CollectionUtils.duplicatedElementsOf(onlyPossibility);
      default:
        ImmutableSet.Builder<Label> duplicates = null;
        for (List<Label> labels : possibleLabels) {
          duplicates = addDuplicateLabels(duplicates, labels);
        }
        return duplicates == null ? ImmutableSet.of() : duplicates.build();
    }
  }

  private static ImmutableSet.Builder<Label> addDuplicateLabels(
      @Nullable ImmutableSet.Builder<Label> builder, List<Label> labels) {
    Set<Label> duplicates = CollectionUtils.duplicatedElementsOf(labels);
    if (duplicates.isEmpty()) {
      return builder;
    }
    if (builder == null) {
      builder = ImmutableSet.builder();
    }
    return builder.addAll(duplicates);
  }

  /**
   * If the attribute is a selector list of list type, then this method returns a list with number
   * of elements equal to the number of select statements in the selector list. Each element of this
   * list is equal to concatenating every possible attribute value in a single select statement.
   * The conditions themselves in the select statements are completely ignored. Returns {@code null}
   * if the attribute isn't of the desired format.
   *
   * As an example, if we have select({a: ["a"], b: ["a", "b"]}) + select({a: ["c", "d"], c: ["e"])
   * The output will be [["a", "a", "b"], ["c", "d", "e"]]. The idea behind this structure is that
   * at least some of the structure in the original selector list is preserved and we know any
   * possible attribute value is the result of concatenating some sublist of each element.
   */
  @Nullable
  public <T> Iterable<T> getConcatenatedSelectorListsOfListType(
      String attributeName, Type<T> type) {
    SelectorList<T> selectorList = getSelectorList(attributeName, type);
    if (selectorList != null && type instanceof ListType) {
      List<T> selectList = new ArrayList<>();

      for (Selector<T> selector : selectorList.getSelectors()) {
        selectList.add(type.concat(selector.getEntries().values()));
      }
      return ImmutableList.copyOf(selectList);
    }
    return null;
  }

  /**
   * Returns a list of all possible values an attribute can take for this rule.
   *
   * <p>If the attribute's value is a simple value, then this returns a singleton list of that
   * value.
   *
   * <p>If the attribute's value is an expression containing one or many {@code select(...)}
   * expressions, then this returns a list of all values that expression may evaluate to. This is
   * dangerous because it's easy to write attributes with an exponential number of possible values:
   *
   * <pre>
   *   foo = select({a: 1, b: 2} + select({c: 3, d: 4}) + select({e: 5, f: 6})
   * </pre>
   *
   * <p>Possible values: <code>[135, 136, 145, 146, 235, 236, 245, 246]</code> (i.e. 2^3).
   *
   * <p>This is true not just for attributes with multiple selects, but also {@link
   * Attribute.ComputedDefault}s depending on such attributes.
   *
   * <p>If the attribute does not have an explicit value for this rule, and the rule provides a
   * computed default, the computed default function is evaluated given the rule's other attribute
   * values as inputs and the output is returned in a singleton list.
   *
   * <p>If the attribute does not have an explicit value for this rule, and the rule provides a
   * computed default, and the computed default function depends on other attributes whose values
   * contain {@code select(...)} expressions, then the computed default function is evaluated for
   * every possible combination of input values, and the list of outputs is returned.
   *
   * <p><b>EFFICIENCY WARNING:</b> Do not use this method unless you really need every single value
   * the attribute might take.
   *
   * <p>More often than not, calling code doesn't really need every value, but really just wants to
   * know, e.g., which labels might appear in a dependency list. For such cases, merging methods
   * like {@link #getReachableLabels} work just as well without the efficiency hit. Use those
   * whenever possible.
   */
  public <T> Iterable<T> visitAttribute(String attributeName, Type<T> type) {
    return visitAttribute(attributeName, type, /*mayTreatMultipleAsNone=*/ false);
  }

  /**
   * Specialization of {@link #visitAttribute(String, Type)} for query output formatters which need
   * one attribute value or none at all. Should be used with the same care as its sibling method.
   *
   * @param mayTreatMultipleAsNone signals if attribute-value computation <b>may</b> be aborted if
   *     more than one possible value is encountered. This parameter is respected on a best-effort
   *     basis - multiple values may still be returned if an unoptimized code path is visited.
   */
  @SuppressWarnings("unchecked")
  public <T> Iterable<T> visitAttribute(
      String attributeName, Type<T> type, boolean mayTreatMultipleAsNone) {
    Object rawVal = rule.getAttr(attributeName, type);

    // If this attribute value is configurable, visit all possible values.
    if (rawVal instanceof SelectorList) {
      return getAllValues(((SelectorList<T>) rawVal).getSelectors(), type, mayTreatMultipleAsNone);
    }

    return visitRawNonConfigurableAttributeValue(rawVal, attributeName, type);
  }

  private <T> List<T> visitRawNonConfigurableAttributeValue(
      Object rawVal, String attributeName, Type<T> type) {
    // If this attribute is a computed default, feed it all possible value combinations of
    // its declared dependencies and return all computed results. For example, if this default
    // uses attributes x and y, x can configurably be x1 or x2, and y can configurably be y1
    // or y1, then compute default values for the (x1,y1), (x1,y2), (x2,y1), and (x2,y2) cases.
    if (rawVal instanceof Attribute.ComputedDefault) {
      return ((Attribute.ComputedDefault) rawVal).getPossibleValues(type, rule);
    }

    if ("visibility".equals(attributeName) && type.equals(BuildType.NODEP_LABEL_LIST)) {
      // This special case for the visibility attribute is needed because its value is replaced
      // with an empty list during package loading if it is public or private in order not to visit
      // the package called 'visibility'.
      return ImmutableList.of(type.cast(rule.getVisibility().getDeclaredLabels()));
    }

    // For any other attribute, just return its direct value.
    T value = getFromRawAttributeValue(rawVal, attributeName, type);
    return value == null ? ImmutableList.of() : ImmutableList.of(value);
  }

  /**
   * Given a list of attributes, creates an {attrName -> attrValue} map for every possible
   * combination of those attributes' values and returns a list of all the maps.
   *
   * <p>For example, given attributes x and y, which respectively have possible values x1, x2 and
   * y1, y2, this returns:
   *
   * <pre>
   *   [
   *    {x: x1, y: y1},
   *    {x: x1, y: y2},
   *    {x: x2, y: y1},
   *    {x: x2, y: y2}
   *   ]
   * </pre>
   *
   * <p>The work done by this method may be limited by providing a {@link ComputationLimiter} that
   * throws if too much work is attempted.
   */
  <TException extends Exception> List<Map<String, Object>> visitAttributes(
      List<String> attributes, ComputationLimiter<TException> limiter) throws TException {
    List<Map<String, Object>> depMaps = new LinkedList<>();
    AtomicInteger combinationsSoFar = new AtomicInteger(0);
    visitAttributesInner(
        attributes,
        depMaps,
        Maps.newHashMapWithExpectedSize(attributes.size()),
        combinationsSoFar,
        limiter);
    return depMaps;
  }

  /**
   * A recursive function used in the implementation of {@link #visitAttributes}.
   *
   * @param attributes a list of attributes that are yet to be visited.
   * @param mappings a mutable list of {attrName --> attrValue} maps collected so far. This method
   *     will add newly discovered maps to the list.
   * @param currentMap {attrName --> attrValue} assignments accumulated so far, not including those
   *     in {@code attributes}. This map may be mutated and as such must be copied if we wish to
   *     preserve its state, such as in the base case.
   * @param combinationsSoFar a counter for all previously processed combinations of possible
   *     values.
   * @param limiter a strategy to limit the work done by invocations of this method.
   */
  private <TException extends Exception> void visitAttributesInner(
      List<String> attributes,
      List<Map<String, Object>> mappings,
      Map<String, Object> currentMap,
      AtomicInteger combinationsSoFar,
      ComputationLimiter<TException> limiter)
      throws TException {
    if (attributes.isEmpty()) {
      // Because this method uses exponential time/space on the number of inputs, we may limit
      // the total number of method calls.
      limiter.onComputationCount(combinationsSoFar.incrementAndGet());
      // Recursive base case: snapshot and store whatever's already been populated in currentMap.
      mappings.add(new HashMap<>(currentMap));
      return;
    }

    // Take the first attribute in the dependency list and iterate over all its values. For each
    // value x, update currentMap with the additional entry { firstAttrName: x }, then feed
    // this recursively into a subcall over all remaining dependencies. This recursively
    // continues until we run out of values.
    String currentAttribute = attributes.get(0);
    Iterable<?> firstAttributePossibleValues =
        visitAttribute(currentAttribute, getAttributeType(currentAttribute));
    List<String> restOfAttrs = attributes.subList(1, attributes.size());
    for (Object value : firstAttributePossibleValues) {
      // Overwrite each time.
      currentMap.put(currentAttribute, value);
      visitAttributesInner(restOfAttrs, mappings, currentMap, combinationsSoFar, limiter);
    }
  }

  /**
   * Returns an {@link AttributeMap} that delegates to {@code AggregatingAttributeMapper.this}
   * except for {@link #get} calls for attributes that are configurable. In that case, the {@link
   * AttributeMap} looks up an attribute's value in {@code directMap}. Any attempt to {@link #get} a
   * configurable attribute that's not in {@code directMap} causes an {@link
   * IllegalArgumentException} to be thrown.
   */
  AttributeMap createMapBackedAttributeMap(final Map<String, Object> directMap) {
    final AggregatingAttributeMapper owner = AggregatingAttributeMapper.this;
    return new AttributeMap() {

      @Override
      public <T> T get(String attributeName, Type<T> type) {
        owner.checkType(attributeName, type);
        if (getNonConfigurableAttributes().contains(attributeName)) {
          return owner.get(attributeName, type);
        }
        if (!directMap.containsKey(attributeName)) {
          throw new IllegalArgumentException(
              "attribute \""
                  + attributeName
                  + "\" isn't available in this computed default context");
        }
        return type.cast(directMap.get(attributeName));
      }

      @Override
      public boolean isConfigurable(String attributeName) {
        return owner.isConfigurable(attributeName);
      }

      @Override
      public String getName() {
        return owner.getName();
      }

      @Override
      public Label getLabel() {
        return owner.getLabel();
      }

      @Override
      public String getRuleClassName() {
        return owner.getRuleClassName();
      }

      @Override
      public Iterable<String> getAttributeNames() {
        return ImmutableList.<String>builder()
            .addAll(directMap.keySet())
            .addAll(getNonConfigurableAttributes())
            .build();
      }

      @Override
      public Collection<DepEdge> visitLabels() {
        return owner.visitLabels();
      }

      @Override
      public Collection<DepEdge> visitLabels(Attribute attribute) {
        return owner.visitLabels(attribute);
      }

      @Override
      public String getPackageDefaultHdrsCheck() {
        return owner.getPackageDefaultHdrsCheck();
      }

      @Override
      public Boolean getPackageDefaultTestOnly() {
        return owner.getPackageDefaultTestOnly();
      }

      @Override
      public String getPackageDefaultDeprecation() {
        return owner.getPackageDefaultDeprecation();
      }

      @Override
      public ImmutableList<String> getPackageDefaultCopts() {
        return owner.getPackageDefaultCopts();
      }

      @Nullable
      @Override
      public Type<?> getAttributeType(String attrName) {
        return owner.getAttributeType(attrName);
      }

      @Nullable
      @Override
      public Attribute getAttributeDefinition(String attrName) {
        return owner.getAttributeDefinition(attrName);
      }

      @Override
      public boolean isAttributeValueExplicitlySpecified(String attributeName) {
        return owner.isAttributeValueExplicitlySpecified(attributeName);
      }

      @Override
      public boolean has(String attrName) {
        return owner.has(attrName);
      }

      @Override
      public <T> boolean has(String attrName, Type<T> type) {
        return owner.has(attrName, type);
      }
    };
  }

  /**
   * Helper class for {@link #getAllValues}. Represents a node in the logical DAG of combinations of
   * {@link Selector}s' values.
   */
  private static class ConfigurableAttrVisitationNode<T> {
    /** Offset into the list of selectors being combined. */
    private final int offset;
    /** Key of the selector taken. */
    private final Label boundKey;
    /** Accumulated value through this node. */
    private final T valueSoFar;

    private ConfigurableAttrVisitationNode(int offset, Label boundKey, T valueSoFar) {
      this.offset = offset;
      this.boundKey = boundKey;
      this.valueSoFar = valueSoFar;
    }
  }

  /**
   * Represents a path previously taken through a previous selector.
   *
   * <p>Used to short-circuit visitation when encountering selectors with <i>equivalent</i> key
   * sets. See uses for details. Note that this optimization is not safe for overlapping but
   * <i>different</i> keysets due to specialization (see {@link ConfiguredAttributeMapper}).
   */
  private static class BoundKeyAndOffset {
    /** Key chosen from associated select. */
    private final Label key;
    /**
     * Offset into the list of selectors where this key was bound. Used to determine when {@link
     * #key} is safe to follow through equivalent selects.
     */
    private final int offset;

    private BoundKeyAndOffset(Label key, int offset) {
      this.key = key;
      this.offset = offset;
    }
  }

  /**
   * Determines all possible values a configurable attribute can take. Do not call this method
   * unless really necessary and avoid all new uses.
   */
  // TODO(bazel-team): minimize or eliminate uses of this interface. It necessarily grows
  // exponentially with the number of selects in the attribute. Is that always necessary?
  // For example, dependency resolution just needs to know every possible label an attribute
  // might reference, but it doesn't need to know the exact combination of labels that make
  // up a value. This may be even less important for non-label values (e.g. strings), which
  // have no impact on the dependency structure.
  private static <T> ImmutableList<T> getAllValues(
      List<Selector<T>> selectors, Type<T> type, boolean mayTreatMultipleAsNone) {
    if (selectors.isEmpty()) {
      return ImmutableList.of();
    }

    if (selectors.size() == 1) {
      // Optimize for common case.
      return selectors.get(0).getEntries().values().stream()
          .filter(Objects::nonNull)
          .collect(ImmutableList.toImmutableList());
    }

    Deque<ConfigurableAttrVisitationNode<T>> nodes = new ArrayDeque<>();
    // Track per selector key set when we started visiting a specific key.
    Map<Set<Label>, BoundKeyAndOffset> boundKeysAndOffsets = new HashMap<>();
    ImmutableList.Builder<T> result = ImmutableList.builder();

    // Seed visitation.
    for (Map.Entry<Label, T> root : selectors.get(0).getEntries().entrySet()) {
      nodes.push(new ConfigurableAttrVisitationNode<>(0, root.getKey(), root.getValue()));
    }

    boolean foundResults = false;
    while (!nodes.isEmpty()) {
      ConfigurableAttrVisitationNode<T> node = nodes.pop();
      int nextOffset = node.offset + 1;
      if (nextOffset >= selectors.size()) {
        // Null values arise when a None is used as the value of a Selector for a type without a
        // default value.
        if (node.valueSoFar != null) {
          if (foundResults && mayTreatMultipleAsNone) {
            // Caller wanted one value or none at all, this is the second, so bail.
            return ImmutableList.of();
          }
          foundResults = true;

          // TODO(gregce): visitAttribute should probably convey that an unset attribute is
          //  possible. Therefore we need to actually handle null values here.
          result.add(node.valueSoFar);
        }
        continue;
      }

      Map<Label, T> nextSelectorEntries = selectors.get(nextOffset).getEntries();
      BoundKeyAndOffset boundKeyAndOffset = boundKeysAndOffsets.get(nextSelectorEntries.keySet());
      if (boundKeyAndOffset != null && boundKeyAndOffset.offset < node.offset) {
        // We've seen this select key set before along this path and chosen this key.
        nodes.push(
            new ConfigurableAttrVisitationNode<>(
                nextOffset,
                boundKeyAndOffset.key,
                concat(type, node.valueSoFar, nextSelectorEntries.get(boundKeyAndOffset.key))));
        continue;
      }

      Set<Label> currentKeys = selectors.get(node.offset).getEntries().keySet();
      // Record that we've descended along node.boundKey starting at this offset.
      boundKeysAndOffsets.put(currentKeys, new BoundKeyAndOffset(node.boundKey, node.offset));

      if (currentKeys.equals(nextSelectorEntries.keySet())) {
        nodes.push(
            new ConfigurableAttrVisitationNode<>(
                nextOffset,
                node.boundKey,
                concat(type, node.valueSoFar, nextSelectorEntries.get(node.boundKey))));
        continue;
      }

      for (Map.Entry<Label, T> entry : nextSelectorEntries.entrySet()) {
        nodes.push(
            new ConfigurableAttrVisitationNode<>(
                nextOffset, entry.getKey(), concat(type, node.valueSoFar, entry.getValue())));
      }
    }

    return result.build();
  }

  private static <T> T concat(Type<T> type, T lhs, T rhs) {
    return type.concat(ImmutableList.of(lhs, rhs));
  }
}
