// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;

/**
 * Container for a pending operation on the reverse deps set. We use subclasses to save 8 bytes of
 * memory instead of keeping a field in this class, and we store {@link Op#CHECK} or {@link Op#ADD}
 * operations as the bare {@link SkyKey} in order to save the wrapper object in that case.
 *
 * <p>When a list of {@link KeyToConsolidate} operations is processed, each operation is performed
 * in order. Operations on a done or freshly evaluating node entry are straightforward: they apply
 * to the entry's reverse deps. Operations on a re-evaluating node entry have a double meaning: they
 * will eventually be applied to the node entry's existing reverse deps, just as for a done node
 * entry, but they are also used to track the entries that declared/redeclared a reverse dep on this
 * entry during this evaluation (and will thus need to be signaled when this entry finishes
 * evaluating).
 */
public abstract class KeyToConsolidate {
  enum Op {
    /**
     * If the entry is re-evaluating, assert that the reverse dep is already present in the set of
     * reverse deps and add this reverse dep to the set of reverse deps to signal when this entry is
     * done. If the entry is already done, do nothing.
     */
    CHECK,
    /**
     * Add the reverse dep to the set of reverse deps and assert it was not already present. If the
     * entry is re-evaluating, add this reverse dep to the set of reverse deps to signal when this
     * entry is done.
     */
    ADD,
    /**
     * Remove the reverse dep from the set of reverse deps and assert it was present. If the entry
     * is re-evaluating, also remove the reverse dep from the set of reverse deps to signal when
     * this entry is done, and assert that it was present.
     */
    REMOVE,
    /**
     * The same as {@link #REMOVE}, except that if the entry is re-evaluating, we assert that the
     * set of reverse deps to signal did <i>not</i> contain this reverse dep.
     */
    REMOVE_OLD
  }

  private static final Interner<KeyToConsolidate> consolidateInterner =
      BlazeInterners.newWeakInterner();

  private final SkyKey key;

  /** Do not call directly -- use the {@link #create} static method instead. */
  private KeyToConsolidate(SkyKey key) {
    this.key = key;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("key", key).toString();
  }

  /**
   * Gets which operation was delayed for the given object, created using {@link #create}. The same
   * {@code opToStoreBare} passed in to {@link #create} should be passed in here.
   */
  static Op op(Object obj, Op opToStoreBare) {
    if (obj instanceof SkyKey) {
      return opToStoreBare;
    }
    if (obj instanceof KeyToAdd) {
      return Op.ADD;
    }
    if (obj instanceof KeyToCheck) {
      return Op.CHECK;
    }
    if (obj instanceof KeyToRemove) {
      return Op.REMOVE;
    }
    if (obj instanceof KeyToRemoveOld) {
      return Op.REMOVE_OLD;
    }
    throw new IllegalStateException(
        "Unknown object type: " + obj + ", " + opToStoreBare + ", " + obj.getClass());
  }

  /** Gets the key whose operation was delayed for the given object. */
  static SkyKey key(Object obj) {
    if (obj instanceof SkyKey) {
      return (SkyKey) obj;
    }
    Preconditions.checkState(obj instanceof KeyToConsolidate, obj);
    return ((KeyToConsolidate) obj).key;
  }

  /**
   * Creates a new operation, encoding the operation {@code op} with reverse dep {@code key}. To
   * save memory, the caller should specify the most common operation expected as {@code
   * opToStoreBare}. That operation will be encoded as the raw {@code key}, saving the memory of an
   * object wrapper. Whatever {@code opToStoreBare} is set to here, the same value must be passed in
   * to {@link #op} when decoding an operation emitted by this method.
   */
  static Object create(SkyKey key, Op op, IncrementalInMemoryNodeEntry entry) {
    if (op == ReverseDepsUtility.getOpToStoreBare(entry)) {
      return key;
    }
    switch (op) {
      case CHECK:
        return consolidateInterner.intern(new KeyToCheck(key));
      case REMOVE:
        return consolidateInterner.intern(new KeyToRemove(key));
      case REMOVE_OLD:
        return consolidateInterner.intern(new KeyToRemoveOld(key));
      case ADD:
        return consolidateInterner.intern(new KeyToAdd(key));
      default:
        throw new IllegalStateException(op + ", " + key);
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == null) {
      return false;
    }
    return this.getClass() == obj.getClass() && this.key.equals(((KeyToConsolidate) obj).key);
  }

  protected int keyHashCode() {
    return key.hashCode();
  }

  @Override
  public int hashCode() {
    // Overridden in subclasses.
    throw new UnsupportedOperationException(key.toString());
  }

  private static final class KeyToAdd extends KeyToConsolidate {
    KeyToAdd(SkyKey key) {
      super(key);
    }

    @Override
    public int hashCode() {
      return keyHashCode();
    }
  }

  private static final class KeyToCheck extends KeyToConsolidate {
    KeyToCheck(SkyKey key) {
      super(key);
    }

    @Override
    public int hashCode() {
      return 31 + 43 * keyHashCode();
    }
  }

  private static final class KeyToRemove extends KeyToConsolidate {
    KeyToRemove(SkyKey key) {
      super(key);
    }

    @Override
    public int hashCode() {
      return 42 + 37 * keyHashCode();
    }
  }

  private static final class KeyToRemoveOld extends KeyToConsolidate {
    KeyToRemoveOld(SkyKey key) {
      super(key);
    }

    @Override
    public int hashCode() {
      return 93 + 37 * keyHashCode();
    }
  }
}
