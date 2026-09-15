// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkPositionIndex;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.Interner;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.Package.Declarations;
import com.google.devtools.build.lib.util.HashCodes;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Represents a rule or macro instance.
 *
 * <p>This encompasses the shared logic between {@link Rule} and {@link MacroInstance}.
 */
public abstract class RuleOrMacroInstance implements DependencyFilter.AttributeInfoProvider {

  private static final String NAME = RuleClass.NAME_ATTRIBUTE.getName();
  static final String GENERATOR_NAME = "generator_name";

  static final String GENERATOR_FUNCTION = "generator_function";
  static final String GENERATOR_LOCATION = "generator_location";

  private static final Object[] EMPTY_OBJECT_ARRAY = new Object[0];
  private static final int[] EMPTY_INT_ARRAY = new int[0];
  private static final BitSet emptyBitSet = new BitSet(0);

  /**
   * Stores attribute values, taking on one of two shapes:
   *
   * <ol>
   *   <li>While the rule or macro instance is mutable, the array length is equal to the number of
   *       attributes. Each array slot holds the attribute value for the corresponding index or null
   *       if not set.
   *   <li>After {@link #freeze}, the array is compacted to store only necessary values. Nulls and
   *       values that match {@link Attribute#getDefaultValue} are omitted to save space. Ordering
   *       of attributes by their index is preserved.
   * </ol>
   */
  private Object[] attrValues;

  /**
   * Holds metadata about attributes, taking on one of two shapes:
   *
   * <ol>
   *   <li>While the rule or macro instance is mutable, contains only a {@link BitSet} tracking
   *       which attribute indices were set explicitly.
   *   <li>After {@link #freeze}, additionally contains an {@linkplain AttributeMetadata#index
   *       index} of which attributes are stored in {@link #attrValues}.
   * </ol>
   */
  private AttributeMetadata attrMetadata;

  final Label label;

  RuleOrMacroInstance(Label label, int attrCount) {
    this.label = checkNotNull(label);
    this.attrValues = new Object[attrCount];
    this.attrMetadata = AttributeMetadata.mutable(attrCount);
  }

  /**
   * Returns true if the subset of this object's fields which are defined in this class equal those
   * of {@code other}. Intended for use by {@code equals()} implementations in subclasses.
   */
  boolean equalsHelper(RuleOrMacroInstance other) {
    return Arrays.equals(attrValues, other.attrValues)
        && attrMetadata.equals(other.attrMetadata)
        && label.equals(other.label);
  }

  /**
   * Returns hash code of the subset of this object's fields which are defined in this class.
   * Intended for use by {@code hashCode()} implementations in subclasses.
   */
  int hashCodeHelper() {
    return label.hashCode()
        + HashCodes.MULTIPLIER
            * (Arrays.hashCode(attrValues) + HashCodes.MULTIPLIER * attrMetadata.hashCode());
  }

  /**
   * Returns the label of the rule or macro instance for error messaging.
   *
   * <p>For symbolic macros, this may not be unique, because macros can create macros that create
   * macros.... that create a single target all with the same name.
   */
  // TODO: steinman - This should be the macro ID, not the label.
  public Label getLabel() {
    return label;
  }

  /** Returns the {@link AttributeProvider} for this rule or macro's parent class. */
  public abstract AttributeProvider getAttributeProvider();

  /**
   * Returns the name part of the label of the target.
   *
   * <p>Equivalent to {@code getLabel().getName()}.
   */
  public String getName() {
    return label.getName();
  }

  /**
   * Returns an (unmodifiable, unordered) collection containing all the Attribute definitions for
   * this kind of rule or macro. (Note, this doesn't include the <i>values</i> of the attributes,
   * merely the schema. Call get[Type]Attr() methods to access the actual values.)
   */
  public Collection<Attribute> getAttributes() {
    return getAttributeProvider().getAttributes();
  }

  /**
   * Returns true iff the {@link AttributeProvider} has an attribute with the given name and type.
   *
   * <p>Note: RuleContext also has isAttrDefined(), which takes Aspects into account. Whenever
   * possible, use RuleContext.isAttrDefined() instead of this method.
   */
  public boolean isAttrDefined(String attrName, Type<?> type) {
    return getAttributeProvider().hasAttr(attrName, type);
  }

  void setAttributeValue(Attribute attribute, Object value, boolean explicit) {
    checkState(!isFrozen(), "Already frozen: %s", this);
    String attrName = attribute.getName();
    if (attrName.equals(NAME)) {
      // Avoid unnecessarily storing the name in attrValues - it's stored in the label.
      return;
    }
    Integer attrIndex = getAttributeProvider().getAttributeIndex(attrName);
    checkArgument(
        attrIndex != null,
        "Attribute %s is not valid for this %s",
        attrName,
        isRuleInstance() ? "rule" : "macro");
    if (explicit) {
      checkState(
          !attrMetadata.getExplicit(attrIndex), "Attribute %s already explicitly set", attrName);
      attrMetadata.setExplicit(attrIndex);
    }
    attrValues[attrIndex] = value;
  }

  /**
   * Returns the value of the given attribute for this rule or macro. Returns null for invalid
   * attributes and default value if attribute was not set.
   *
   * @param attrName the name of the attribute to lookup.
   */
  @Nullable
  public Object getAttr(String attrName) {
    if (attrName.equals(NAME)) {
      return getName();
    }
    Integer attrIndex = getAttributeProvider().getAttributeIndex(attrName);
    return attrIndex == null ? null : getAttrWithIndex(attrIndex);
  }

  /**
   * Returns the value of the given attribute if it has the right type.
   *
   * @throws IllegalArgumentException if the attribute does not have the expected type.
   */
  @Nullable
  public <T> Object getAttr(String attrName, Type<T> type) {
    if (attrName.equals(NAME)) {
      checkAttrType(attrName, type, RuleClass.NAME_ATTRIBUTE);
      return getName();
    }

    Integer index = getAttributeProvider().getAttributeIndex(attrName);
    if (index == null) {
      throw new IllegalArgumentException(
          "No such attribute "
              + attrName
              + " in "
              + getAttributeProvider()
              + (isRuleInstance() ? " rule " : " macro ")
              + label);
    }
    checkAttrType(attrName, type, getAttributeProvider().getAttribute(index));
    return getAttrWithIndex(index);
  }

  /**
   * Returns the value of the attribute with the given index. Returns null, if no such attribute
   * exists OR no value was set.
   */
  @Nullable
  Object getAttrWithIndex(int attrIndex) {
    Object value = getAttrIfStored(attrIndex);
    if (value != null) {
      return value;
    }
    Attribute attr = getAttributeProvider().getAttribute(attrIndex);
    return attr.getDefaultValueUnchecked();
  }

  /**
   * Returns the attribute value at the specified index if stored in this rule or macro, otherwise
   * {@code null}.
   *
   * <p>Unlike {@link #getAttr}, does not fall back to the default value.
   */
  @Nullable
  Object getAttrIfStored(int attrIndex) {
    int attrCount = getAttributeProvider().getAttributeCount();
    checkPositionIndex(attrIndex, attrCount - 1);
    if (!isFrozen()) {
      return attrValues[attrIndex];
    }
    int index = Arrays.binarySearch(attrMetadata.index, attrIndex);
    if (index >= 0) {
      return attrValues[index];
    }
    if (attrMetadata.getExplicit(attrIndex)) {
      Attribute attr = getAttributeProvider().getAttribute(attrIndex);
      return attr.getDefaultValueUnchecked();
    }
    return null;
  }

  /**
   * Returns raw attribute values stored by this rule or macro.
   *
   * <p>The indices of attribute values in the returned list are not guaranteed to be consistent
   * with the other methods of this class. If this is important, which is generally the case, avoid
   * this method.
   *
   * <p>The returned iterable may contain null values. Its {@link Iterable#iterator} is
   * unmodifiable.
   */
  Iterable<Object> getRawAttrValues() {
    return () -> Iterators.forArray(attrValues);
  }

  /** See {@link #isAttributeValueExplicitlySpecified(String)} */
  @Override
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    return isAttributeValueExplicitlySpecified(attribute.getName());
  }

  /**
   * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. This
   * returns true also if the value explicitly specified in the BUILD file is the same as the
   * attribute's default value. In addition, this method return false if the rule or macro has no
   * attribute with the given name.
   */
  public boolean isAttributeValueExplicitlySpecified(String attrName) {
    if (attrName.equals(NAME)) {
      return true;
    }
    if ((attrName.equals(GENERATOR_FUNCTION) || attrName.equals(GENERATOR_LOCATION))
        && isRuleInstance()) {
      return isRuleCreatedInMacro();
    }
    Integer attrIndex = getAttributeProvider().getAttributeIndex(attrName);
    if (attrIndex == null) {
      return false;
    }
    return attrMetadata.getExplicit(attrIndex);
  }

  /* Returns true iff this is a rule instance (v. macro). */
  public abstract boolean isRuleInstance();

  /**
   * Returns whether this is a rule (v. macro) that was created by a legacy or symbolic macro.
   * Always false for macro instances; sometimes true for rules.
   */
  public abstract boolean isRuleCreatedInMacro();

  private void checkAttrType(String attrName, Type<?> requestedType, Attribute attr) {
    if (requestedType != attr.getType()) {
      throw new IllegalArgumentException(
          "Attribute "
              + attrName
              + " is of type "
              + attr.getType()
              + " and not of type "
              + requestedType
              + " in "
              + getAttributeProvider()
              + (isRuleInstance() ? " rule " : " macro ")
              + label);
    }
  }

  /**
   * Returns {@code true} if this rule or macro's attributes are immutable.
   *
   * <p>Frozen instances optimize for space by omitting storage for attribute values that match the
   * {@link Attribute} default. If {@link #getAttrIfStored} returns {@code null}, the value should
   * be taken from either {@link Attribute#getLateBoundDefault} for late-bound defaults or {@link
   * Attribute#getDefaultValue} for all other attributes (including computed defaults).
   *
   * <p>Mutable instances have no such optimization. During rule creation, this allows for
   * distinguishing whether a computed default (which may depend on other unset attributes) is
   * available.
   */
  boolean isFrozen() {
    return attrMetadata.index != null;
  }

  /** Makes this rule or macro's attributes immutable and compacts their representation. */
  void freeze() {
    if (isFrozen()) {
      return;
    }

    AttributeProvider provider = getAttributeProvider();
    int numToStore = 0;
    for (int i = 0; i < attrValues.length; i++) {
      Object value = attrValues[i];
      if (value == null) {
        continue;
      }
      if (value.equals(provider.getAttribute(i).getDefaultValueUnchecked())) {
        attrValues[i] = null;
        continue;
      }
      numToStore++;
    }

    if (numToStore == 0) {
      this.attrValues = EMPTY_OBJECT_ARRAY;
      this.attrMetadata = AttributeMetadata.frozen(EMPTY_INT_ARRAY, attrMetadata.explicit);
    } else {
      Object[] compactValues = new Object[numToStore];
      int[] index = new int[numToStore];

      int destIdx = 0;
      for (int i = 0; i < attrValues.length; i++) {
        Object value = attrValues[i];
        if (value != null) {
          index[destIdx] = i;
          compactValues[destIdx] = value;
          destIdx++;
        }
      }

      this.attrValues = compactValues;
      this.attrMetadata = AttributeMetadata.frozen(index, attrMetadata.explicit);
    }

    // Sanity check to ensure mutable vs frozen is distinguishable.
    checkState(isFrozen(), "Freeze unsuccessful");
  }

  /**
   * Encapsulates attribute metadata (index mapping of non-default attribute values and tracking of
   * which attributes were explicitly set).
   *
   * <p>Target declarations across a codebase frequently follow uniform usage patterns (e.g.,
   * setting the same common subsets of explicit and non-default attributes for a given rule class),
   * so the duplication rate among frozen {@code AttributeMetadata} instances is high. Interning
   * these instances achieves significant heap savings.
   */
  private static final class AttributeMetadata {
    private static final Interner<AttributeMetadata> interner = BlazeInterners.newWeakInterner();

    static AttributeMetadata mutable(int attrCount) {
      return new AttributeMetadata(null, new BitSet(attrCount));
    }

    static AttributeMetadata frozen(int[] index, BitSet explicit) {
      if (explicit.isEmpty()) {
        explicit = emptyBitSet;
      } else if (explicit.size() - explicit.length() >= Long.SIZE) {
        // More words allocated than necessary. Copy to a more compact BitSet.
        BitSet compacted = new BitSet(explicit.length());
        compacted.or(explicit);
        explicit = compacted;
      }
      return interner.intern(new AttributeMetadata(index, explicit));
    }

    /**
     * When mutable, this is {@code null}. When frozen, this is a sorted array of the attribute
     * indices with non-default values. The corresponding values are stored in {@link #attrValues}
     * at the exact same array offset (i.e., {@code attrValues[i]} is the value for attribute index
     * {@code index[i]}).
     */
    @Nullable private final int[] index;

    /** Tracks the indices of attributes which were explicitly specified. */
    private final BitSet explicit;

    AttributeMetadata(@Nullable int[] index, BitSet explicit) {
      this.index = index;
      this.explicit = checkNotNull(explicit);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof AttributeMetadata other)) {
        return false;
      }
      return Arrays.equals(index, other.index) && explicit.equals(other.explicit);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(index) + HashCodes.MULTIPLIER * explicit.hashCode();
    }

    boolean getExplicit(int attrIndex) {
      return explicit.get(attrIndex);
    }

    void setExplicit(int attrIndex) {
      checkState(index == null, "Cannot mutate explicit attributes when frozen");
      explicit.set(attrIndex);
    }
  }

  /**
   * Returns a {@link BuildType.SelectorList} for the given attribute if the attribute is
   * configurable for this rule or macro, null otherwise.
   */
  @Nullable
  @SuppressWarnings("unchecked")
  public <T> BuildType.SelectorList<T> getSelectorList(String attributeName, Type<T> type) {
    Integer index = getAttributeProvider().getAttributeIndex(attributeName);
    if (index == null) {
      return null;
    }
    Object attrValue = getAttrIfStored(index);
    if (!(attrValue instanceof BuildType.SelectorList<?> selectorList)) {
      return null;
    }
    if (selectorList.getOriginalType() != type) {
      throw new IllegalArgumentException(
          "Attribute "
              + attributeName
              + " is not of type "
              + type
              + " in "
              + getAttributeProvider()
              + " rule "
              + label);
    }
    return (BuildType.SelectorList<T>) selectorList;
  }

  /**
   * Retrieves the package's default visibility, or for certain rule classes, injects a different
   * default visibility.
   */
  public abstract RuleVisibility getDefaultVisibility();

  @Nullable
  @SuppressWarnings("unchecked")
  private List<Label> getRawVisibilityLabels() {
    Integer visibilityIndex = getAttributeProvider().getAttributeIndex("visibility");
    if (visibilityIndex == null) {
      return null;
    }
    return (List<Label>) getAttrIfStored(visibilityIndex);
  }

  /**
   * Returns the declared labels of the visibility attribute, or the default visibility if the
   * attribute is not set.
   */
  public List<Label> getVisibilityDeclaredLabels() {
    List<Label> rawLabels = getRawVisibilityLabels();
    return rawLabels != null ? rawLabels : getDefaultVisibility().getDeclaredLabels();
  }

  /** Returns the metadata of the package where this target or macro instance lives. */
  public abstract Package.Metadata getPackageMetadata();

  abstract Declarations getPackageDeclarations();

  /**
   * Returns the innermost symbolic macro that declared this target or macro instance, or null if it
   * was declared outside any symbolic macro (i.e. directly in a BUILD file or only in one or more
   * legacy macros).
   */
  @Nullable
  public abstract MacroInstance getDeclaringMacro();

  @Nullable
  public PackageArgs getPackageArgs() {
    return getPackageDeclarations().getPackageArgs();
  }

  abstract void reportError(String message, TargetDefinitionContext targetDefinitionContext);
}
