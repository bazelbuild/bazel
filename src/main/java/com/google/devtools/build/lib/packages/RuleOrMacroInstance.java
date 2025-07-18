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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkPositionIndex;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package.Declarations;
import com.google.devtools.build.lib.util.HashCodes;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Represents a rule or macro instance.
 *
 * <p>This encompasses the shared logic between {@link Rule} and {@link MacroInstance}.
 */
public abstract class RuleOrMacroInstance implements DependencyFilter.AttributeInfoProvider {

  static final String NAME = RuleClass.NAME_ATTRIBUTE.getName();
  static final String GENERATOR_NAME = "generator_name";

  static final String GENERATOR_FUNCTION = "generator_function";
  static final String GENERATOR_LOCATION = "generator_location";

  private static final int ATTR_SIZE_THRESHOLD = 126;

  /**
   * For {@link Rule}s, the length of this instance's generator name if it is a prefix of its name,
   * otherwise zero. For {@link MacroInstance}s, always zero since they never have a generator name.
   *
   * <p>The generator name of a rule is the {@code name} parameter passed to a legacy macro that
   * instantiates the rule. Most rules instantiated via legacy macro follow this pattern:
   *
   * <pre>{@code
   * def some_macro(name):
   *   some_rule(name = name + '_some_suffix')
   * }</pre>
   *
   * thus resulting in a generator name which is a prefix of the rule name. In such a case, we save
   * memory by storing the length of the generator name instead of the string. Note that this saves
   * memory from both the storage in {@link #attrValues} and the string itself (if it is not
   * otherwise retained). This optimization works because this field does not push the shallow heap
   * cost of {@link Rule} beyond an 8-byte threshold. If it did, this optimization would be a net
   * loss.
   */
  int generatorNamePrefixLength = 0;

  RuleOrMacroInstance(Label label, int attrCount) {
    this.label = checkNotNull(label);
    this.attrValues = new Object[attrCount];
    this.attrBytes = new byte[bitSetSize(attrCount)];
  }

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
   * Holds bits of metadata about attributes, taking on one of three shapes:
   *
   * <ol>
   *   <li>While the rule or macro instance is mutable, contains one bit for each attribute
   *       indicating whether it was explicitly set.
   *   <li>After {@link #freeze} for rules or macros with fewer than 126 attributes (extremely
   *       common case), contains one byte dedicated to each value in the compact representation of
   *       {@link #attrValues}, at corresponding array indices. The first bit indicates whether the
   *       attribute was explicitly set. The remaining 7 bits represent the attribute's index (as
   *       per {@link AttributeProvider#getAttributeIndex}). See {@link #freezeSmall}.
   *   <li>After {@link #freeze} for rules with 126 or more attributes (rare case), contains the
   *       full set of bytes from the mutable representation, followed by the index of each
   *       attribute stored in the compact representation of {@link #attrValues}. Because attribute
   *       indices may require a full byte, there is no room to pack the explicit bit as we do for
   *       the small case. See {@link #freezeLarge}.
   * </ol>
   */
  private byte[] attrBytes;

  Label label;

  /**
   * Returns true if the subset of this object's fields which are defined in this class equal those
   * of {@code other}. Intended for use by {@code equals()} implementations in subclasses.
   */
  protected boolean equalsHelper(RuleOrMacroInstance other) {
    return generatorNamePrefixLength == other.generatorNamePrefixLength
        && Arrays.equals(attrValues, other.attrValues)
        && Arrays.equals(attrBytes, other.attrBytes)
        && Objects.equals(label, other.label);
  }

  /**
   * Returns hash code of the subset of this object's fields which are defined in this class.
   * Intended for use by {@code hashCode()} implementations in subclasses.
   */
  protected int hashCodeHelper() {
    return HashCodes.hashObjects(generatorNamePrefixLength, label)
        + HashCodes.MULTIPLIER
            * (Arrays.hashCode(attrValues) + HashCodes.MULTIPLIER * Arrays.hashCode(attrBytes));
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

  /**
   * Copies attribute values from the given rule or macro instance to this rule or macro instance.
   */
  void copyAttributesFrom(RuleOrMacroInstance ruleOrMacroInstance) {
    Preconditions.checkArgument(
        getAttributeProvider().equals(ruleOrMacroInstance.getAttributeProvider()),
        "Rule class mismatch: (this=%s, given=%s)",
        getAttributeProvider(),
        ruleOrMacroInstance.getAttributeProvider());
    Preconditions.checkArgument(
        ruleOrMacroInstance.isFrozen(), "Not frozen: %s", ruleOrMacroInstance);
    checkState(!isFrozen(), "Already frozen: %s", this);
    this.attrValues = ruleOrMacroInstance.attrValues;
    this.attrBytes = ruleOrMacroInstance.attrBytes;
  }

  void setAttributeValue(Attribute attribute, Object value, boolean explicit) {
    Preconditions.checkState(!isFrozen(), "Already frozen: %s", this);
    String attrName = attribute.getName();
    if (attrName.equals(NAME)) {
      // Avoid unnecessarily storing the name in attrValues - it's stored in the label.
      return;
    }
    if (attrName.equals(GENERATOR_NAME)) {
      String generatorName = (String) value;
      if (getName().startsWith(generatorName)) {
        generatorNamePrefixLength = generatorName.length();
        return;
      }
    }
    Integer attrIndex = getAttributeProvider().getAttributeIndex(attrName);
    Preconditions.checkArgument(
        attrIndex != null,
        "Attribute %s is not valid for this %s",
        attrName,
        isRuleInstance() ? "rule" : "macro");
    if (explicit) {
      checkState(!getExplicitBit(attrIndex), "Attribute %s already explicitly set", attrName);
      setExplicitBit(attrIndex);
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
    return switch (getAttrState()) {
      case MUTABLE -> attrValues[attrIndex];
      case FROZEN_SMALL -> {
        int index = binarySearchAttrBytes(0, attrIndex, 0x7f);
        yield index < 0 ? null : attrValues[index];
      }
      case FROZEN_LARGE -> {
        if (attrBytes.length == 0) {
          yield null;
        }
        int bitSetSize = bitSetSize(attrCount);
        int index = binarySearchAttrBytes(bitSetSize, attrIndex, 0xff);
        yield index < 0 ? null : attrValues[index - bitSetSize];
      }
    };
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
    if ((attrName.equals(GENERATOR_FUNCTION)
            || attrName.equals(GENERATOR_LOCATION)
            || attrName.equals(GENERATOR_NAME))
        && isRuleInstance()) {
      return isRuleCreatedInMacro();
    }
    Integer attrIndex = getAttributeProvider().getAttributeIndex(attrName);
    if (attrIndex == null) {
      return false;
    }
    return switch (getAttrState()) {
      case MUTABLE, FROZEN_LARGE -> getExplicitBit(attrIndex);
      case FROZEN_SMALL -> {
        int index = binarySearchAttrBytes(0, attrIndex, 0x7f);
        yield index >= 0 && (attrBytes[index] & 0x80) != 0;
      }
    };
  }

  /* Returns true iff this is a rule instance (v. macro). */
  public abstract boolean isRuleInstance();

  /**
   * Returns whether this is a rule (v. macro) that was created by a legacy or symbolic macro.
   * Always false for macro instances; sometimes true for rules.
   */
  public abstract boolean isRuleCreatedInMacro();

  /** Returns index into {@link #attrBytes} for {@code attrIndex}, or -1 if not found */
  int binarySearchAttrBytes(int start, int attrIndex, int mask) {
    // Binary search, treating values as unsigned bytes.
    int lo = start;
    int hi = attrBytes.length - 1;
    while (hi >= lo) {
      int mid = (lo + hi) / 2;
      int midAttrIndex = attrBytes[mid] & mask;
      if (midAttrIndex == attrIndex) {
        return mid;
      } else if (midAttrIndex < attrIndex) {
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return -1;
  }

  void checkAttrType(String attrName, Type<?> requestedType, Attribute attr) {
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
   * <p>Frozen instances optimize for space by omitting storage for non-explicit attribute values
   * that match the {@link Attribute} default. If {@link #getAttrIfStored} returns {@code null}, the
   * value should be taken from either {@link Attribute#getLateBoundDefault} for late-bound defaults
   * or {@link Attribute#getDefaultValue} for all other attributes (including computed defaults).
   *
   * <p>Mutable instances have no such optimization. During rule creation, this allows for
   * distinguishing whether a computed default (which may depend on other unset attributes) is
   * available.
   */
  boolean isFrozen() {
    return getAttrState() != AttrState.MUTABLE;
  }

  /** Makes this rule or macro's attributes immutable and compacts their representation. */
  void freeze() {
    if (isFrozen()) {
      return;
    }

    BitSet indicesToStore = new BitSet();
    for (int i = 0; i < attrValues.length; i++) {
      Object value = attrValues[i];
      if (value == null) {
        continue;
      }
      if (!getExplicitBit(i)) {
        Attribute attr = getAttributeProvider().getAttribute(i);
        if (value.equals(attr.getDefaultValueUnchecked())) {
          // Non-explicit value matches the attribute's default. Save space by omitting storage.
          continue;
        }
      }
      indicesToStore.set(i);
    }

    if (getAttributeProvider().getAttributeCount() < ATTR_SIZE_THRESHOLD) {
      freezeSmall(indicesToStore);
    } else {
      freezeLarge(indicesToStore);
    }
    // Sanity check to ensure mutable vs frozen is distinguishable.
    checkState(isFrozen(), "Freeze unsuccessful");
  }

  private void freezeSmall(BitSet indicesToStore) {
    int numToStore = indicesToStore.cardinality();
    Object[] compactValues = new Object[numToStore];
    byte[] compactBytes = new byte[numToStore];

    int attrIndex = 0;
    for (int i = 0; i < numToStore; i++) {
      attrIndex = indicesToStore.nextSetBit(attrIndex);
      byte byteValue = (byte) (0x7f & attrIndex);
      if (getExplicitBit(attrIndex)) {
        byteValue = (byte) (byteValue | 0x80);
      }
      compactBytes[i] = byteValue;
      compactValues[i] = attrValues[attrIndex];
      attrIndex++;
    }

    this.attrValues = compactValues;
    this.attrBytes = compactBytes;
  }

  private void freezeLarge(BitSet indicesToStore) {
    int numToStore = indicesToStore.cardinality();
    int bitSetSize = attrBytes.length;
    Object[] compactValues = new Object[numToStore];
    byte[] compactBytes = Arrays.copyOf(attrBytes, bitSetSize + numToStore);

    int attrIndex = 0;
    for (int i = 0; i < numToStore; i++) {
      attrIndex = indicesToStore.nextSetBit(attrIndex);
      compactBytes[i + bitSetSize] = (byte) attrIndex;
      compactValues[i] = attrValues[attrIndex];
      attrIndex++;
    }

    this.attrValues = compactValues;
    this.attrBytes = compactBytes;
  }

  enum AttrState {
    MUTABLE,
    FROZEN_SMALL,
    FROZEN_LARGE
  }

  AttrState getAttrState() {
    // This check works because the name attribute is never stored, so the compact representation
    // of attrValues will always have length < attr count.
    int attrCount = getAttributeProvider().getAttributeCount();
    if (attrValues.length == attrCount) {
      return AttrState.MUTABLE;
    }
    return attrCount < ATTR_SIZE_THRESHOLD ? AttrState.FROZEN_SMALL : AttrState.FROZEN_LARGE;
  }

  /** Calculates the number of bytes necessary to have an explicit bit for each attribute. */
  private static int bitSetSize(int attrCount) {
    // ceil(attrCount / 8)
    return (attrCount + 7) / 8;
  }

  private boolean getExplicitBit(int attrIndex) {
    int byteIndex = attrIndex / 8;
    int bitIndex = attrIndex % 8;
    byte byteValue = attrBytes[byteIndex];
    return (byteValue & (1 << bitIndex)) != 0;
  }

  private void setExplicitBit(int attrIndex) {
    int byteIndex = attrIndex / 8;
    int bitIndex = attrIndex % 8;
    byte byteValue = attrBytes[byteIndex];
    attrBytes[byteIndex] = (byte) (byteValue | (1 << bitIndex));
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
    if (!(attrValue instanceof BuildType.SelectorList)) {
      return null;
    }
    if (((BuildType.SelectorList<?>) attrValue).getOriginalType() != type) {
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
    return (BuildType.SelectorList<T>) attrValue;
  }

  /**
   * Retrieves the package's default visibility, or for certain rule classes, injects a different
   * default visibility.
   */
  public abstract RuleVisibility getDefaultVisibility();

  /**
   * Implementation of {@link #getRawVisibility} that avoids constructing a {@code RuleVisibility}.
   */
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

  abstract void reportError(String message, EventHandler eventHandler);
}
