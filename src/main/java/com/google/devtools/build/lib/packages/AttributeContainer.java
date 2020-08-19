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

import com.google.common.base.Preconditions;
import java.util.Arrays;

/**
 * Provides attribute setting and retrieval for a Rule. In particular, it can be consumed by
 * independent {@link AttributeMap} instances that can apply varying kinds of logic for determining
 * the "value" of an attribute. For example, a configurable attribute's "value" may be a { config
 * --> value } dictionary or a configuration-bound lookup on that dictionary, depending on the
 * context in which it's requested.
 *
 * <p>This class provides the lowest-level access to attribute information. It is *not* intended to
 * be a robust public interface, but rather just an input to {@link AttributeMap} instances. Use
 * those instances for all domain-level attribute access.
 */
// TODO(adonovan): eliminate this class.
// All uses of this calls come from Rule or WorkspaceFactory. Perhaps we can eliminate
// the WorkspaceFactory.setParent hack (removing the WorkspaceFactory usage) and inline
// AttributeContainer in Rule?
abstract class AttributeContainer {

  protected final RuleClass ruleClass;

  // Conceptually an AttributeContainer is an unordered set of triples
  // (attribute, value, explicit).
  // - attribute is represented internally as its index attrIndex.
  //   cheaply convertible to name and Attribute classes via RuleClass.
  // - value is an opaque object, stored in the values array
  // - explicit is a boolean which tracks if the attribute was
  //   explicitly set in the BUILD file.
  //
  // The representations used are sparse representations, so space usage is proportional
  // to the number of set attributes and not the number of potential attributes.
  // We use two representations for this information and pick between them based
  // on the maximum number of attributes in the rule.
  //
  // Potential optimizations for later:
  // - Add a freeze() method which denotes that the instance becomes unmodifiable.
  //   - Sort internal arrays, so we can use binary search.
  //   - In general many parts of package/rule could probably benefit from
  //     having a freeze method, which gets triggered in packageBuilder.finishBuild()
  // - generator_* attributes are inferred from callstack and not explicitly stored.
  // - Have an attributeContainer at the package level to store attributes inferred from
  //   package(), and do NOT set them at the rule level unnecessarily.
  // - Once the above are done, could we infer "explicit" bit from the following:
  //   the existence of value, the type of attribute (late bound, synthetic, hidden),
  //   whether the attribute can be (and is) set at package level.

  // Attribute values in an arbitrary order (actually the order they were added).
  private Object[] values;

  // The 'value' and 'explicit' components are both encoded in the state byte array,
  // though the representation varies based on N = ruleClass.getAttributeCount().
  // - For small rules (N < 126), state[i] encodes the the 'value' index
  //   in the 7 lower bits and 'explicit' in the top bit. This is the common case.
  // - For large rules, the 'explicit' bits are packed into the initial N/8 bytes
  //   and the 'value' indices occupy the remaining bytes.
  // - A value of 0 indicates an unused entry.
  protected byte[] state;

  // Useful Terminology for reading the code.
  //  - attrIndex: an integer associated with a legal attribute of the ruleClass.
  //    RuleClass owns the mapping between attribute, its name, and attributeIndex.
  //  - stateIndex: an index into the state[] array.
  //    value of 0 represents unused entries,
  //  - valueIndex: an index into the attributeValues array.
  //    unused entries have null (reverse not necessarily true, but probably is)
  //    legal to use only if corresponding stateIndex > 0

  private static final byte[] EMPTY_STATE = {};
  private static final Object[] EMPTY_VALUES = {};

  // Grow an array by this factor.
  private static final float GROWTH_FACTOR = 1.2f;

  /**
   * Creates a container for a rule of the given rule class. private constructor forces all
   * subclasses to be implemented in this file.
   */
  private AttributeContainer(RuleClass ruleClass) {
    this.ruleClass = ruleClass;
    this.values = EMPTY_VALUES;
    this.state = EMPTY_STATE;
    // subclasses should initialize storage for explicit bits.
  }

  /**
   * Returns one of the following...
   *
   * <ul>
   *   <li>the index in state which represents the given attrIndex.
   *   <li>OR the location where it could be added
   *   <li>OR -1 if state is full. The first two can be distinguished by seeing if the value of
   *       state at that location is 0. See {@link #getValueIndex}.
   * </ul>
   */
  protected abstract int getStateIndex(int attrIndex);

  /** Update value at state[stateIndex] to point to attrIndex. */
  protected abstract void setStateValue(int stateIndex, int attrIndex);

  /** Returns the valueIndex stored for the entry at given stateIndex. Returns -1 if not found. */
  private int getValueIndex(int stateIndex) {
    if (stateIndex < 0 || state[stateIndex] == 0) {
      return -1;
    } else {
      return stateIndex;
    }
  }

  private void setValue(int attrIndex, Object value) {
    int stateIndex = getStateIndex(attrIndex);
    int valueIndex = getValueIndex(stateIndex);
    if (valueIndex >= 0) {
      // overwrite existing value.
      values[stateIndex] = value;
      return;
    }
    if (stateIndex < 0) {
      // Logically stateIndex should be at the end of physical array.
      stateIndex = state.length;
    }
    // grow state[] if needed
    if (state.length <= stateIndex) {
      int newLength = Math.max((int) (state.length * GROWTH_FACTOR), stateIndex + 1);
      // Round up to multiple of 8 (so it takes multiple of 8 bytes)
      newLength = (newLength + 7) & ~7;
      state = Arrays.copyOf(state, newLength);
    }

    setStateValue(stateIndex, attrIndex);
    valueIndex = stateIndex; // since state and attributeValues are parallel arrays.
    // grow values[] if needed.
    if (values.length <= valueIndex) {
      int newLength = Math.max((int) (values.length * GROWTH_FACTOR), valueIndex + 1);
      // Round up to multiple of 2 (so it takes multiple of 8 bytes).
      newLength = (newLength + 1) & ~1;
      values = Arrays.copyOf(values, newLength);
    }
    values[valueIndex] = value;
  }

  /** Returns the explicit bit for attrIndex. */
  protected abstract boolean getExplicit(int attrIndex);

  /** Sets the explicit bit for attrIndex. */
  protected abstract void setExplicit(int attrIndex);

  /** See {@link #isAttributeValueExplicitlySpecified(String)}. */
  boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    return isAttributeValueExplicitlySpecified(attribute.getName());
  }

  /**
   * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. In
   * addition, this method return false if the rule has no attribute with the given name.
   */
  boolean isAttributeValueExplicitlySpecified(String attributeName) {
    Integer attrIndex = ruleClass.getAttributeIndex(attributeName);
    return attrIndex != null && getExplicit(attrIndex);
  }

  /**
   * Returns the value of the attribute with index attrIndex. Returns null if the attribute is not
   * set in the container.
   */
  Object getAttributeValue(int attrIndex) {
    Preconditions.checkArgument(attrIndex >= 0);
    int valueIndex = getValueIndex(getStateIndex(attrIndex));
    return valueIndex < 0 ? null : values[valueIndex];
  }

  /**
   * Updates the value of the attribute.
   *
   * @param attribute the attribute whose value to update.
   * @param value the value to set
   * @param explicit was this explicitly set in the BUILD file.
   */
  void setAttributeValue(Attribute attribute, Object value, boolean explicit) {
    String name = attribute.getName();
    Integer attrIndex = ruleClass.getAttributeIndex(name);
    if (!explicit && getExplicit(attrIndex)) {
      throw new IllegalArgumentException("attribute " + name + " already explicitly set");
    }
    setValue(attrIndex, value);
    if (explicit) {
      setExplicit(attrIndex);
    }
  }

  /**
   * Concrete implementation which supports upto 126 attributes.
   * <li>The state[] and attributeValues[] array are parallel to each other, i.e. state[i] <-->
   *     attributeValues[i] correspond to each other.
   * <li>The explicit bits are stored in the most-significant-bit of state[i].
   * <li>This if state[i] represents attrIndex, the value stored will be
   *
   *     <pre>(attrIndex + 1) + (explicit ? 128 : 0)</pre>
   *
   *     The +1 ensures the value is non-zero as 0 represents unused.
   */
  private static final class SmallAttributeContainer extends AttributeContainer {

    private SmallAttributeContainer(RuleClass ruleClass) {
      super(ruleClass);
    }

    /**
     * Returns one of the following...
     * <li>the index in state which represents the given attrIndex.
     * <li>OR the location where it could be added
     * <li>OR -1 if state is full. The first two can be distinguished by seeing if the value of
     *     state at that location is 0. See {@link #getValueIndex}.
     */
    @Override
    protected final int getStateIndex(int attrIndex) {
      for (int i = 0; i < state.length; i += 1) {
        if (state[i] == 0) {
          // reached logical end of array.
          return i;
        }
        // Interpret the bottom 7 bits as the attrIndex and subtract 1.
        if ((0x7f & state[i]) - 1 == attrIndex) {
          // Found the entry for attrIndex.
          return i;
        }
      }
      return -1;
    }

    @Override
    protected void setStateValue(int stateIndex, int attrIndex) {
      // Update bottom 7 bits to (attrIndex+1) and preserve MSB.
      byte bottom = (byte) (attrIndex + 1);
      byte top = (byte) (state[stateIndex] & 0x80);
      state[stateIndex] = (byte) (top | bottom);
    }

    @Override
    protected boolean getExplicit(int attrIndex) {
      int stateIndex = getStateIndex(attrIndex);
      if ((stateIndex < 0) || (state[stateIndex] == 0)) {
        // No value stored, so cannot be explicit.
        return false;
      }
      // Check MSB of state[stateIndex]
      return (state[stateIndex] & 0x80) != 0;
    }

    @Override
    protected void setExplicit(int attrIndex) {
      int stateIndex = getStateIndex(attrIndex);
      if ((stateIndex < 0) || (state[stateIndex] == 0)) {
        // No value stored.
        throw new IllegalStateException("Cannot set explicit bit before storing value.");
      }
      // Set the high bit.
      state[stateIndex] = (byte) (state[stateIndex] | 0x80);
    }
  }

  /**
   * Implementation where the explicit bits are stored in a prefix of the state[] array.
   *
   * <p>Take P=ceil(ruleClass.getAttributeCount()/8)
   * <li>If state[i+P] has value (attrIndex+1), the corresponding value is stored in
   *     attributeValues[i]
   * <li>The first P bytes will be used to store the explicit bits. In particular, the explicitBit
   *     for attrIndex=K-1 is stored in the K'th bit (viewing the first P bytes as a sequence of 8*P
   *     bits).
   *
   *     <p>Conceptually, we inlined the subset of features of BitSet we need and used a prefix of
   *     the state[] array to store that information.
   */
  private static final class ExplicitAttributeContainer extends AttributeContainer {

    private ExplicitAttributeContainer(RuleClass ruleClass) {
      super(ruleClass);
      // Pre-allocate space for the explicit bits.
      state = new byte[prefixSize(ruleClass.getAttributeCount())];
    }

    private static int prefixSize(int numAttributes) {
      // ceil(numAttributes / 8)
      return (numAttributes + 7) >> 3;
    }

    /**
     * Returns one of the following...
     * <li>the index in state which represents the given attrIndex.
     * <li>OR the location where it could be added
     * <li>OR -1 if state is full. The first two can be distinguished by seeing if the value of
     *     state at that location is 0. See {@link #getValueIndex}.
     */
    @Override
    protected final int getStateIndex(int attrIndex) {
      int p = prefixSize(ruleClass.getAttributeCount());
      for (int i = p; i < state.length; i += 1) {
        if (state[i] == 0) {
          // reached logical end of array.
          return i;
        }
        // Interpret byte as an integer and subtract 1.
        if ((0xff & state[i]) - 1 == attrIndex) {
          // Found the entry for attrIndex.
          return i;
        }
      }
      return -1;
    }

    @Override
    protected void setStateValue(int stateIndex, int attrIndex) {
      // This value would be > prefixSize(ruleClass.getAttributeCount())
      state[stateIndex] = (byte) (attrIndex + 1);
    }

    @Override
    protected boolean getExplicit(int attrIndex) {
      int idx = (attrIndex + 1); // The bit to look for at the start of state[].
      byte explicitByte = state[idx >> 3];
      byte mask = (byte) (1 << (idx & 0x07));
      return (explicitByte & mask) != 0;
    }

    // Sets the given bit in the value and returns new value.
    @Override
    protected void setExplicit(int attrIndex) {
      int idx = (attrIndex + 1); // The bit to look for at the start of state[].
      byte explicitByte = state[idx >> 3];
      byte mask = (byte) (1 << (idx & 0x07));
      state[idx >> 3] = (byte) (explicitByte | mask);
    }
  }

  /** Returns an AttributeContainer for holding attributes of the given rule class. */
  static AttributeContainer newInstance(RuleClass ruleClass) {
    int numAttributes = ruleClass.getAttributeCount();
    if (numAttributes <= 126) {
      return new SmallAttributeContainer(ruleClass);
    } else if (numAttributes <= 254) {
      return new ExplicitAttributeContainer(ruleClass);
    } else {
      // If we run into this, add a new implementation of AttributeContainer where byte is replaced
      // with
      // short.
      throw new AssertionError("can't pack " + numAttributes + " rule indices into bytes");
    }
  }
}
