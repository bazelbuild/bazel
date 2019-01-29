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

import com.google.devtools.build.lib.events.Location;
import java.util.Arrays;
import javax.annotation.Nullable;

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

  // Holds two lists of attribute indices.
  // The first byte gives the length of the first list.
  // The first list records which attributes were set explicitly in the BUILD file.
  // The second list ends at the end of the array.
  // The second list records which attributes have Locations, in reverse order
  // from the attributeLocations array.
  // Between the lists there may be unused zero bytes (zeros are forbidden within each list).
  private byte[] state;

  // Attribute locations, packed:
  private Location[] attributeLocations;


  /**
   * Create a container for a rule of the given rule class.
   */
  public AttributeContainer(RuleClass ruleClass) {
   this(ruleClass, EMPTY_LOCATIONS);
  }

  AttributeContainer(RuleClass ruleClass, Location[] locations) {
    int n = ruleClass.getAttributeCount();
    if (n > 254) {
      // We reserve the zero byte as a hole/sentinel inside state[].
      // If you hit this limit, replace byte with char and remove the masking with 0xff.
      throw new AssertionError("can't pack " + n + " rule indices into bytes");
    }
    this.ruleClass = ruleClass;
    this.attributeValues = new Object[n];
    this.state = EMPTY_STATE;
    this.attributeLocations = locations;
  }

  private static final byte[] EMPTY_STATE = {0};
  private static final Location[] EMPTY_LOCATIONS = {};

  /**
   * Returns an attribute value by name, or null on no match.
   */
  @Nullable
  public Object getAttr(String attrName) {
    Integer idx = ruleClass.getAttributeIndex(attrName);
    return idx != null ? attributeValues[idx] : null;
  }

  /**
   * See {@link #isAttributeValueExplicitlySpecified(String)}.
   */
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    return isAttributeValueExplicitlySpecified(attribute.getName());
  }

  /**
   * Returns true iff the value of the specified attribute is explicitly set in the BUILD file. This
   * returns true also if the value explicitly specified in the BUILD file is the same as the
   * attribute's default value. In addition, this method return false if the rule has no attribute
   * with the given name.
   */
  public boolean isAttributeValueExplicitlySpecified(String attributeName) {
    Integer idx = ruleClass.getAttributeIndex(attributeName);
    return idx != null && getExplicit(idx);
  }

 /**
  * Returns the number of elements of state[] currently used to store
  * indices of "explicitly set" attributes.
  */
  private int explicitCount() {
    return 0xff & state[0];
  }

  private boolean getExplicit(int index) {
    int n = explicitCount();
    for (int i = 1; i <= n; ++i) {
      if ((0xff & state[i]) == index + 1) {
        return true;
      }
    }
    return false;
  }

  private int getLocationIndex(int index) {
    int n = explicitCount();
    for (int i = state.length - 1; i > n; --i) {
      if ((0xff & state[i]) == index + 1) {
        return state.length - 1 - i;
      }
    }
    return -1;
  }

  private void setExplicit(int index) {
    if (getExplicit(index)) {
      return;
    }
    ensureSpace();
    int n = explicitCount() + 1;
    state[0] = (byte) n;
    state[n] = (byte) (index + 1);
  }

  private int addLocationIndex(int index) {
    ensureSpace();
    for (int i = state.length - 1; ; --i) {
      if (i <= explicitCount()) {
        throw new AssertionError("ensureSpace() did not insert a zero");
      }
      if (state[i] == 0) {
        state[i] = (byte) (index + 1);
        return state.length - 1 - i;
      }
    }
  }

  /**
   * Ensures that the state[n] byte is equal to the sentinel value 0, so there is room for another
   * attribute's explicit bit to be set, or for an attribute's location to be set.
   */
  private void ensureSpace() {
    int n = explicitCount() + 1;
    if (n < state.length && state[n] == 0) {
      return;
    }
    // Grow up to the next multiple of eight bytes, as the object will be
    // aligned to eight bytes anyway.  Insert zeros between the two lists.
    byte[] newState = new byte[(state.length | 7) + 1];
    // Copy stored explicit attributes to the beginning of the array.
    System.arraycopy(state, 0, newState, 0, n);
    // Copy stored attribute locations to the *end* of the array.
    int oldLocations = state.length - n;
    System.arraycopy(state, n, newState, newState.length - oldLocations, oldLocations);
    state = newState;
  }

  /**
   * Returns the location of the attribute definition for this rule, or null if not found.
   */
  public Location getAttributeLocation(String attrName) {
    Integer idx = ruleClass.getAttributeIndex(attrName);
    int locationIndex = idx != null ? getLocationIndex(idx) : -1;
    return locationIndex >= 0 ? attributeLocations[locationIndex] : null;
  }

  Object getAttributeValue(int index) {
    return attributeValues[index];
  }

  void setAttributeValue(Attribute attribute, Object value, boolean explicit) {
    String name = attribute.getName();
    Integer index = ruleClass.getAttributeIndex(name);
    if (!explicit && getExplicit(index)) {
      throw new IllegalArgumentException("attribute " + name + " already explicitly set");
    }
    attributeValues[index] = value;
    if (explicit) {
      setExplicit(index);
    }
  }

  // This sets the attribute "explicitly" as if it came from the BUILD file.
  // At present, the sole use of this is for the test_suite.$implicit_tests
  // attribute, which is synthesized during package loading.  We do want to
  // consider that "explicitly set" so that it appears in query output.
  void setAttributeValueByName(String attrName, Object value) {
    setAttributeValue(ruleClass.getAttributeByName(attrName), value, true);
  }

  void setAttributeLocation(int attrIndex, Location location) {
    int locationIndex = getLocationIndex(attrIndex);
    if (locationIndex >= 0) {
      throw new IllegalArgumentException("already have a location for attribute "
          + ruleClass.getAttribute(attrIndex).getName() + ": " + attributeLocations[locationIndex]);
    }
    locationIndex = addLocationIndex(attrIndex);
    if (locationIndex >= attributeLocations.length) {
      // Grow by two references, as the object will be aligned to eight bytes anyway.
      attributeLocations = Arrays.copyOf(attributeLocations, attributeLocations.length + 2);
    }
    attributeLocations[locationIndex] = location;
  }
}
