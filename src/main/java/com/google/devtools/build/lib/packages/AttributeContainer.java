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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.BitSet;
import java.util.List;
import javax.annotation.Nullable;

/**
 * AttributeContainer holds all attribute values of a rule. In addition, for each one, it records
 * whether it is 'explicit', that is, provided by the function call that instantiated the rule, as
 * opposed to the default value, or a value computed from other attributes.
 *
 * <p>This class provides the lowest-level access to attribute information. The public interface for
 * attribute access is the rule itself. For a higher level abstraction use {@link AttributeMap}
 * instead.
 */
abstract class AttributeContainer {

  /**
   * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. In
   * addition, this method returns false if the rule has no attribute with the given index.
   */
  abstract boolean isAttributeValueExplicitlySpecified(int attrIndex);

  /**
   * Returns the value of the attribute with index attrIndex. Returns null if the attribute is not
   * set in the container.
   */
  @Nullable
  abstract Object getAttributeValue(int attrIndex);

  /**
   * Returns attribute values as tracked by this instance. The indices of attribute values in the
   * returned list are not guaranteed to be consistent with the other methods of this class. If this
   * is important, which is generally the case, avoid this method.
   */
  abstract List<Object> getRawAttributeValues();

  /**
   * Updates the value of the attribute.
   *
   * @param attrIndex the index of attribute whose value to update. Assumed to be valid index.
   * @param value the value to set
   * @param explicit was this explicitly set in the BUILD file.
   */
  abstract void setAttributeValue(int attrIndex, Object value, boolean explicit);

  /** Returns a frozen AttributeContainer with the same attributes, and a compact representation. */
  @CheckReturnValue
  abstract AttributeContainer freeze(Rule rule);

  /**
   * Returns {@code true} if this container is immutable.
   *
   * <p>Frozen containers optimize for space by omitting storage for non-explicit attribute values
   * that match the {@link Attribute} default. If {@link #getAttributeValue} returns {@code null},
   * the value should be taken from {@link Attribute#getDefaultValue}, even for computed defaults.
   *
   * <p>Mutable containers have no such optimization. During rule creation, this allows for
   * distinguishing whether a computed default (which may depend on other unset attributes) is
   * available.
   */
  abstract boolean isFrozen();

  /** Returns an AttributeContainer for holding attributes of the given rule class. */
  static AttributeContainer newMutableInstance(RuleClass ruleClass) {
    int attrCount = ruleClass.getAttributeCount();
    Preconditions.checkArgument(attrCount < 254);
    return new Mutable(attrCount);
  }

  /** An AttributeContainer to which attributes may be added. */
  private static final class Mutable extends AttributeContainer {

    // Sparsely populated array of values, indexed by Attribute.index.
    final Object[] values;
    final BitSet explicitIndices = new BitSet();

    Mutable(int maxAttrCount) {
      values = new Object[maxAttrCount];
    }

    @Override
    public boolean isAttributeValueExplicitlySpecified(int attrIndex) {
      return (attrIndex >= 0) && explicitIndices.get(attrIndex);
    }

    /**
     * Returns the value of the attribute with index attrIndex. Returns null if the attribute is not
     * yet been set.
     */
    @Override
    @Nullable
    Object getAttributeValue(int attrIndex) {
      return values[attrIndex];
    }

    /**
     * Updates the value of the attribute.
     *
     * @param attrIndex the index of the attribute whose value to update.
     * @param value the value to set
     * @param explicit was this explicitly set in the BUILD file.
     */
    @Override
    void setAttributeValue(int attrIndex, Object value, boolean explicit) {
      if (attrIndex < 0 || attrIndex >= values.length) {
        throw new IllegalArgumentException(
            "attribute with index " + attrIndex + " is not valid for rule");
      }
      if (!explicit && explicitIndices.get(attrIndex)) {
        throw new IllegalArgumentException(
            "attribute with index " + attrIndex + " already explicitly set");
      }
      values[attrIndex] = value;
      if (explicit) {
        explicitIndices.set(attrIndex);
      }
    }

    @Override
    public AttributeContainer freeze(Rule rule) {
      BitSet indicesToStore = new BitSet();
      RuleClass ruleClass = rule.getRuleClassObject();

      for (int i = 0; i < values.length; i++) {
        Object value = values[i];
        if (value == null) {
          continue;
        }
        if (!explicitIndices.get(i)) {
          Attribute attr = ruleClass.getAttribute(i);
          Object defaultValue = attr.getDefaultValue(attr.hasComputedDefault() ? rule : null);
          if (value.equals(defaultValue)) {
            // Non-explicit value matches the attribute's default. Save space by omitting storage.
            continue;
          }
        }
        indicesToStore.set(i);
      }

      return values.length < 126
          ? new Small(values, explicitIndices, indicesToStore)
          : new Large(values, explicitIndices, indicesToStore);
    }

    @Override
    public boolean isFrozen() {
      return false;
    }

    @Override
    List<Object> getRawAttributeValues() {
      // Mutable copy since ImmutableList doesn't support null.
      return Lists.newArrayList(values);
    }
  }

  /** Frozen AttributeContainer based on compact array with indirect indexing. */
  private abstract static class Frozen extends AttributeContainer {

    @Override
    final void setAttributeValue(int attrIndex, Object value, boolean explicit) {
      throw new UnsupportedOperationException(
          "Readonly implementations do not support setAttributeValue");
    }

    @Override
    final AttributeContainer freeze(Rule rule) {
      return this;
    }

    @Override
    final boolean isFrozen() {
      return true;
    }
  }

  /** Returns index into state array for attrIndex, or -1 if not found */
  private static int getStateIndex(byte[] state, int start, int attrIndex, int mask) {
    // Binary search, treating values as unsigned bytes.
    int lo = start;
    int hi = state.length - 1;
    while (hi >= lo) {
      int mid = (lo + hi) / 2;
      int midAttrIndex = state[mid] & mask;
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

  /**
   * A frozen implementation of AttributeContainer which supports RuleClasses with up to 126
   * attributes.
   */
  @VisibleForTesting
  static final class Small extends Frozen {

    private final int maxAttrCount;

    // Conceptually an AttributeContainer is an unordered set of triples
    // (attribute, value, explicit).
    // - attribute is represented internally as its index attrIndex.
    //   cheaply convertible to name and Attribute classes via RuleClass.
    // - value is an opaque object, stored in the values array
    // - explicit is a boolean which tracks if the attribute was
    //   explicitly set in the BUILD file.

    // Attribute values stored in attrIndex order.
    private final Object[] values;

    // The 'value' and 'explicit' components are encoded in the same byte.
    // Since this class only supports ruleClass with < 126 attributes,
    // state[i] encodes the 'value' index in the 7 lower bits and 'explicit' in the top bit.
    // This is the common case.
    private final byte[] state;

    // Useful Terminology for reading the code.
    //  - attrIndex: an integer associated with a legal attribute of the ruleClass.
    //    RuleClass owns the mapping between attribute, its name, and attributeIndex.
    //  - stateIndex: an index into the state[] array.
    //  - valueIndex: an index into the attributeValues array.

    /**
     * Creates a container for a rule of the given rule class. Assumes attrIndex < 126 always.
     *
     * @param attrValues values for all attributes, null values are considered unset
     * @param explicitIndices holds explicit bit for each attribute index
     * @param indicesToStore attribute indices for values that need to be stored, i.e., they were
     *     explicitly set and/or differ from the attribute's default value
     */
    private Small(Object[] attrValues, BitSet explicitIndices, BitSet indicesToStore) {
      this.maxAttrCount = attrValues.length;
      int numToStore = indicesToStore.cardinality();
      this.values = new Object[numToStore];
      this.state = new byte[numToStore];

      int attrIndex = 0;
      for (int i = 0; i < numToStore; i++) {
        attrIndex = indicesToStore.nextSetBit(attrIndex);
        byte stateValue = (byte) (0x7f & attrIndex);
        if (explicitIndices.get(attrIndex)) {
          stateValue = (byte) (stateValue | 0x80);
        }
        state[i] = stateValue;
        values[i] = attrValues[attrIndex];
        attrIndex++;
      }
    }

    /**
     * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. In
     * addition, this method return false if the rule has no attribute with the given name.
     */
    @Override
    boolean isAttributeValueExplicitlySpecified(int attrIndex) {
      if (attrIndex < 0) {
        return false;
      }
      int stateIndex = getStateIndex(state, 0, attrIndex, 0x7f);
      return stateIndex >= 0 && (state[stateIndex] & 0x80) != 0;
    }

    @Nullable
    @Override
    Object getAttributeValue(int attrIndex) {
      Preconditions.checkArgument(attrIndex >= 0);
      if (attrIndex >= maxAttrCount) {
        throw new IndexOutOfBoundsException(
            "Maximum valid attrIndex is " + (maxAttrCount - 1) + ". Given " + attrIndex);
      }
      int stateIndex = getStateIndex(state, 0, attrIndex, 0x7f);
      return stateIndex < 0 ? null : values[stateIndex];
    }

    @Override
    List<Object> getRawAttributeValues() {
      // Mutable copy since ImmutableList doesn't support null.
      return Lists.newArrayList(values);
    }
  }

  /**
   * A frozen implementation of AttributeContainer which supports RuleClasses with up to 254
   * attributes.
   */
  @VisibleForTesting
  static final class Large extends Frozen {

    private final int maxAttrCount;

    // Conceptually an AttributeContainer is an unordered set of triples
    // (attribute, value, explicit).
    // - attribute is represented internally as its index attrIndex.
    //   cheaply convertible to name and Attribute classes via RuleClass.
    // - value is an opaque object, stored in the values array
    // - explicit is a boolean which tracks if the attribute was
    //   explicitly set in the BUILD file.

    // Attribute values stored in attrIndex order.
    private final Object[] values;

    // P = ceil(ruleClass.attributeCount()/8)
    // The first P bytes store the explicit bits, while the remaining bytes store attrIndex.
    //
    // NOTE: We could potentially shave off a few bytes by using P=ceil(values.length/8)
    // But
    // - it leads to higher CPU cost
    // - actual memory savings may not be much since memory is allocated in blocks of 8 bytes and
    //   the savings is at most 8 bytes.
    // - this implementation is used only if ruleClass supports > 126 attributes (very rare).
    private final byte[] state;

    // Useful Terminology for reading the code.
    //  - attrIndex: an integer associated with a legal attribute of the ruleClass.
    //    RuleClass owns the mapping between attribute, its name, and attributeIndex.
    //  - stateIndex: an index into the state[] array.
    //  - valueIndex: an index into the attributeValues array.

    /** Calculates the number of bytes necessary to have an explicit bit for each attribute. */
    private static int prefixSize(int attrCount) {
      // ceil(max attributes / 8)
      return (attrCount + 7) / 8;
    }

    /**
     * Sets the explicit bit for {@code attrIndex} in the byte array. Assumes {@code attrIndex} is a
     * valid index.
     */
    private static void setExplicitBit(byte[] bytes, int attrIndex) {
      int byteIndex = attrIndex / 8;
      int bitIndex = attrIndex % 8;
      byte byteValue = bytes[byteIndex];
      bytes[byteIndex] = (byte) (byteValue | (1 << bitIndex));
    }

    /**
     * Gets the explicit bit for {@code attrIndex} in the byte array. Assumes {@code attrIndex} is a
     * valid index.
     */
    private static boolean getExplicitBit(byte[] bytes, int attrIndex) {
      int byteIndex = attrIndex / 8;
      int bitIndex = attrIndex % 8;
      byte byteValue = bytes[byteIndex];
      return (byteValue & (1 << bitIndex)) != 0;
    }

    /**
     * Creates a container for a rule of the given rule class. Assumes maxAttrCount < 254
     *
     * @param attrValues values for all attributes, null values are considered unset.
     * @param explicitIndices holds explicit bit for each attribute index
     * @param indicesToStore attribute indices for values that need to be stored, i.e. they were
     *     explicitly set and/or differ from the attribute's default value
     */
    private Large(Object[] attrValues, BitSet explicitIndices, BitSet indicesToStore) {
      this.maxAttrCount = attrValues.length;
      int numToStore = indicesToStore.cardinality();
      int p = prefixSize(maxAttrCount);
      this.values = new Object[numToStore];
      this.state = new byte[p + numToStore];

      int attrIndex = 0;
      for (int i = 0; i < numToStore; i++) {
        attrIndex = indicesToStore.nextSetBit(attrIndex);
        if (explicitIndices.get(attrIndex)) {
          setExplicitBit(state, attrIndex);
        }
        state[i + p] = (byte) attrIndex;
        values[i] = attrValues[attrIndex];
        attrIndex++;
      }
    }

    /**
     * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. In
     * addition, this method return false if the rule has no attribute with the given name.
     */
    @Override
    boolean isAttributeValueExplicitlySpecified(int attrIndex) {
      return (attrIndex >= 0) && getExplicitBit(state, attrIndex);
    }

    @Nullable
    @Override
    Object getAttributeValue(int attrIndex) {
      Preconditions.checkArgument(attrIndex >= 0);
      if (state.length == 0) {
        return null;
      }
      if (attrIndex >= maxAttrCount) {
        throw new IndexOutOfBoundsException(
            "Maximum valid attrIndex is " + (maxAttrCount - 1) + ". Given " + attrIndex);
      }
      int p = prefixSize(maxAttrCount);
      int stateIndex = getStateIndex(state, p, attrIndex, 0xff);
      return stateIndex < 0 ? null : values[stateIndex - p];
    }

    @Override
    List<Object> getRawAttributeValues() {
      // Mutable copy since ImmutableList doesn't support null.
      return Lists.newArrayList(values);
    }
  }
}
