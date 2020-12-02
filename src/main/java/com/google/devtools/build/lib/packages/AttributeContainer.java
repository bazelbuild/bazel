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
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.BitSet;
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
   * Updates the value of the attribute.
   *
   * @param attrIndex the index of attribute whose value to update. Assumed to be valid index.
   * @param value the value to set
   * @param explicit was this explicitly set in the BUILD file.
   */
  abstract void setAttributeValue(int attrIndex, Object value, boolean explicit);

  /** Returns a frozen AttributeContainer with the same attributes, and a compact representation. */
  @CheckReturnValue
  abstract AttributeContainer freeze();

  /** Returns an AttributeContainer for holding attributes of the given rule class. */
  static AttributeContainer newMutableInstance(RuleClass ruleClass) {
    int attrCount = ruleClass.getAttributeCount();
    Preconditions.checkArgument(attrCount < 254);
    return new Mutable(attrCount);
  }

  /** An AttributeContainer to which attributes may be added. */
  static final class Mutable extends AttributeContainer {

    // Sparsely populated array of values, indexed by Attribute.index.
    final Object[] values;
    final BitSet explicitAttrs = new BitSet();

    @VisibleForTesting
    Mutable(int maxAttrCount) {
      values = new Object[maxAttrCount];
    }

    @Override
    public boolean isAttributeValueExplicitlySpecified(int attrIndex) {
      return (attrIndex >= 0) && explicitAttrs.get(attrIndex);
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
      if (!explicit && explicitAttrs.get(attrIndex)) {
        throw new IllegalArgumentException(
            "attribute with index " + attrIndex + " already explicitly set");
      }
      values[attrIndex] = value;
      if (explicit) {
        explicitAttrs.set(attrIndex);
      }
    }

    @Override
    public AttributeContainer freeze() {
      if (values.length < 126) {
        return new Small(values, explicitAttrs);
      } else {
        return new Large(values, explicitAttrs);
      }
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
    final AttributeContainer freeze() {
      return this;
    }
  }

  private static final byte[] EMPTY_STATE = {};
  private static final Object[] EMPTY_VALUES = {};

  /** Returns number of non-null values. */
  private static int nonNullCount(Object[] attrValues) {
    // Pre-allocate longer array.
    int numSet = 0;
    for (Object val : attrValues) {
      if (val != null) {
        numSet++;
      }
    }
    return numSet;
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
    // state[i] encodes the the 'value' index in the 7 lower bits and 'explicit' in the top bit.
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
     * @param attrValues values for all attributes, null values are considered unset.
     * @param explicitAttrs holds explicit bit for each attribute index
     */
    private Small(Object[] attrValues, BitSet explicitAttrs) {
      maxAttrCount = attrValues.length;
      int numSet = nonNullCount(attrValues);
      if (numSet == 0) {
        this.values = EMPTY_VALUES;
        this.state = EMPTY_STATE;
        return;
      }
      values = new Object[numSet];
      state = new byte[numSet];
      int index = 0;
      int attrIndex = -1;
      for (Object attrValue : attrValues) {
        attrIndex++;
        if (attrValue == null) {
          continue;
        }
        byte stateValue = (byte) (0x7f & attrIndex);
        if (explicitAttrs.get(attrIndex)) {
          stateValue = (byte) (stateValue | 0x80);
        }
        state[index] = stateValue;
        values[index] = attrValue;
        index += 1;
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

    private static int prefixSize(int attrCount) {
      // ceil(max attributes / 8)
      return (attrCount + 7) >> 3;
    }

    /** Set the specified bit in the byte array. Assumes bitIndex is a valid index. */
    private static void setBit(byte[] bits, int bitIndex) {
      int idx = (bitIndex + 1);
      int explicitByte = bits[idx >> 3];
      byte mask = (byte) (1 << (idx & 0x07));
      bits[idx >> 3] = (byte) (explicitByte | mask);
    }

    /** Get the specified bit in the byte array. Assumes bitIndex is a valid index. */
    private static boolean getBit(byte[] bits, int bitIndex) {
      int idx = (bitIndex + 1);
      int explicitByte = bits[idx >> 3];
      int mask = (byte) (1 << (idx & 0x07));
      return (explicitByte & mask) != 0;
    }

    /**
     * Creates a container for a rule of the given rule class. Assumes maxAttrCount < 254
     *
     * @param attrValues values for all attributes, null values are considered unset.
     * @param explicitAttrs holds explicit bit for each attribute index
     */
    private Large(Object[] attrValues, BitSet explicitAttrs) {
      this.maxAttrCount = attrValues.length;
      int numSet = nonNullCount(attrValues);
      if (numSet == 0) {
        this.values = EMPTY_VALUES;
        this.state = EMPTY_STATE;
        return;
      }
      int p = prefixSize(maxAttrCount);
      values = new Object[numSet];
      state = new byte[p + numSet];
      int index = 0;
      int attrIndex = -1;
      for (Object attrValue : attrValues) {
        attrIndex++;
        if (attrValue == null) {
          continue;
        }
        if (explicitAttrs.get(attrIndex)) {
          setBit(state, attrIndex);
        }
        state[index + p] = (byte) attrIndex;
        values[index] = attrValue;
        index += 1;
      }
    }

    /**
     * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. In
     * addition, this method return false if the rule has no attribute with the given name.
     */
    @Override
    boolean isAttributeValueExplicitlySpecified(int attrIndex) {
      return (attrIndex >= 0) && getBit(state, attrIndex);
    }

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
  }
}
