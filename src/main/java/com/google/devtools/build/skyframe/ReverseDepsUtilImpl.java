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
package com.google.devtools.build.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A utility class that allows us to keep the reverse dependencies as an array list instead of a
 * set. This is more memory-efficient. At the same time it allows us to group the removals and
 * uniqueness checks so that it also performs well.
 *
 * <p>The reason for making this a separate class is to share non-trivial code between BuildingState
 * and NodeEntry. We could simply make those two classes extend this class instead, but we would be
 * less memory-efficient since object memory alignment does not cross classes (you would have two
 * memory alignments, one for the base class and one for the extended one).
 *
 */
public abstract class ReverseDepsUtilImpl<T> implements ReverseDepsUtil<T> {

  static final int MAYBE_CHECK_THRESHOLD = 10;

  private static final Interner<KeyToConsolidate> consolidateInterner =
      BlazeInterners.newWeakInterner();

  abstract void setReverseDepsObject(T container, Object object);

  abstract void setDataToConsolidate(T container, @Nullable List<Object> dataToConsolidate);

  abstract Object getReverseDepsObject(T container);

  abstract List<Object> getDataToConsolidate(T container);

  private enum ConsolidateOp {
    CHECK,
    ADD,
    REMOVE
  }

  /**
   * Opaque container for a pending operation on the reverse deps set. We use subclasses to save
   * 8 bytes of memory instead of keeping a field in this class, and we store
   * {@link ConsolidateOp#CHECK} operations as the bare {@link SkyKey} in order to save the wrapper
   * object in that case.
   */
  private abstract static class KeyToConsolidate {
    // Do not access directly -- use the {@link #key} static accessor instead.
    protected final SkyKey key;

    /** Do not call directly -- use the {@link #create} static method instead. */
    private KeyToConsolidate(SkyKey key) {
      this.key = key;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("key", key).toString();
    }

    /** Gets which operation was delayed for the given object. */
    static ConsolidateOp op(Object obj) {
      if (obj instanceof SkyKey) {
        return ConsolidateOp.CHECK;
      }
      if (obj instanceof KeyToRemove) {
        return ConsolidateOp.REMOVE;
      }
      Preconditions.checkState(obj instanceof KeyToAdd, obj);
      return ConsolidateOp.ADD;
    }

    /** Gets the key whose operation was delayed for the given object. */
    static SkyKey key(Object obj) {
      if (obj instanceof SkyKey) {
        return (SkyKey) obj;
      }
      Preconditions.checkState(obj instanceof KeyToConsolidate, obj);
      return ((KeyToConsolidate) obj).key;
    }

    static Object create(SkyKey key, ConsolidateOp op) {
      switch (op) {
        case CHECK:
          return key;
        case REMOVE:
          return consolidateInterner.intern(new KeyToRemove(key));
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

    @Override
    public int hashCode() {
      // Overridden in subclasses.
      throw new UnsupportedOperationException(key.toString());
    }
  }

  private static final class KeyToAdd extends KeyToConsolidate {
    KeyToAdd(SkyKey key) {
      super(key);
    }

    @Override
    public int hashCode() {
      return key.hashCode();
    }
  }

  private static final class KeyToRemove extends KeyToConsolidate {
    KeyToRemove(SkyKey key) {
      super(key);
    }

    @Override
    public int hashCode() {
      return 42 + 37 * key.hashCode();
    }
  }

  private void maybeDelayReverseDepOp(T container, Iterable<SkyKey> reverseDeps, ConsolidateOp op) {
    List<Object> consolidations = getDataToConsolidate(container);
    int currentReverseDepSize = getCurrentReverseDepSize(container);
    if (consolidations == null) {
      consolidations = new ArrayList<>(currentReverseDepSize);
      setDataToConsolidate(container, consolidations);
    }
    for (SkyKey reverseDep : reverseDeps) {
      consolidations.add(KeyToConsolidate.create(reverseDep, op));
    }
    // TODO(janakr): Should we consolidate more aggressively? This threshold can be customized.
    if (consolidations.size() == currentReverseDepSize) {
      consolidateData(container);
    }
  }

  private boolean isSingleReverseDep(T container) {
    return !(getReverseDepsObject(container) instanceof List);
  }

  /**
   *  We only check if reverse deps is small and there are no delayed data to consolidate, since
   *  then presence or absence would not be known.
   */
  @Override
  public void maybeCheckReverseDepNotPresent(T container, SkyKey reverseDep) {
    if (getDataToConsolidate(container) != null) {
      return;
    }
    if (isSingleReverseDep(container)) {
      Preconditions.checkState(
          !getReverseDepsObject(container).equals(reverseDep),
          "Reverse dep %s already present in %s",
          reverseDep,
          container);
      return;
    }
    @SuppressWarnings("unchecked")
    List<SkyKey> asList = (List<SkyKey>) getReverseDepsObject(container);
    if (asList.size() < MAYBE_CHECK_THRESHOLD) {
      Preconditions.checkState(
          !asList.contains(reverseDep),
          "Reverse dep %s already present in %s for %s",
          reverseDep,
          asList,
          container);
    }
  }

  private int getCurrentReverseDepSize(T container) {
    return isSingleReverseDep(container)
        ? 1
        : ((List<SkyKey>) getReverseDepsObject(container)).size();
  }

  /**
   * We use a memory-efficient trick to keep reverseDeps memory usage low. Edges in Bazel are
   * dominant over the number of nodes.
   *
   * <p>Most of the nodes have zero or one reverse dep. That is why we use immutable versions of the
   * lists for those cases. In case of the size being > 1 we switch to an ArrayList. That is because
   * we also have a decent number of nodes for which the reverseDeps are huge (for example almost
   * everything depends on BuildInfo node).
   *
   * <p>We also optimize for the case where we have only one dependency. In that case we keep the
   * object directly instead of a wrapper list.
   */
  @SuppressWarnings("unchecked")
  public void addReverseDeps(T container, Collection<SkyKey> newReverseDeps) {
    if (newReverseDeps.isEmpty()) {
      return;
    }
    List<Object> dataToConsolidate = getDataToConsolidate(container);
    if (dataToConsolidate != null) {
      maybeDelayReverseDepOp(container, newReverseDeps, ConsolidateOp.ADD);
      return;
    }
    Object reverseDeps = getReverseDepsObject(container);
    int reverseDepsSize = isSingleReverseDep(container) ? 1 : ((List<SkyKey>) reverseDeps).size();
    int newSize = reverseDepsSize + newReverseDeps.size();
    if (newSize == 1) {
      overwriteReverseDepsWithObject(container, Iterables.getOnlyElement(newReverseDeps));
    } else if (reverseDepsSize == 0) {
      overwriteReverseDepsList(container, Lists.newArrayList(newReverseDeps));
    } else if (reverseDepsSize == 1) {
      List<SkyKey> newList = Lists.newArrayListWithExpectedSize(newSize);
      newList.add((SkyKey) reverseDeps);
      newList.addAll(newReverseDeps);
      overwriteReverseDepsList(container, newList);
    } else {
      ((List<SkyKey>) reverseDeps).addAll(newReverseDeps);
    }
  }

  @Override
  public void checkReverseDep(T container, SkyKey reverseDep) {
    maybeDelayReverseDepOp(container, ImmutableList.of(reverseDep), ConsolidateOp.CHECK);
  }

  /**
   * See {@code addReverseDeps} method.
   */
  @Override
  public void removeReverseDep(T container, SkyKey reverseDep) {
    maybeDelayReverseDepOp(container, ImmutableList.of(reverseDep), ConsolidateOp.REMOVE);
  }

  @Override
  public ImmutableSet<SkyKey> getReverseDeps(T container) {
    consolidateData(container);

    // TODO(bazel-team): Unfortunately, we need to make a copy here right now to be on the safe side
    // wrt. thread-safety. The parents of a node get modified when any of the parents is deleted,
    // and we can't handle that right now.
    if (isSingleReverseDep(container)) {
      return ImmutableSet.of((SkyKey) getReverseDepsObject(container));
    } else {
      @SuppressWarnings("unchecked")
      List<SkyKey> reverseDeps = (List<SkyKey>) getReverseDepsObject(container);
      ImmutableSet<SkyKey> set = ImmutableSet.copyOf(reverseDeps);
      Preconditions.checkState(
          set.size() == reverseDeps.size(),
          "Duplicate reverse deps present in %s: %s. %s",
          this,
          reverseDeps,
          container);
      return set;
    }
  }

  @Override
  public void consolidateReverseDeps(T container) {
    consolidateData(container);
  }

  private void consolidateData(T container) {
    List<Object> dataToConsolidate = getDataToConsolidate(container);
    if (dataToConsolidate == null) {
      return;
    }
    setDataToConsolidate(container, null);
    Object reverseDeps = getReverseDepsObject(container);
    if (isSingleReverseDep(container)) {
      Preconditions.checkState(
          dataToConsolidate.size() == 1,
          "dataToConsolidate not size 1 even though only one rdep: %s %s %s",
          dataToConsolidate,
          reverseDeps,
          container);
      Object keyToConsolidate = Iterables.getOnlyElement(dataToConsolidate);
      SkyKey key = KeyToConsolidate.key(keyToConsolidate);
      switch (KeyToConsolidate.op(keyToConsolidate)) {
        case REMOVE:
          overwriteReverseDepsList(container, ImmutableList.<SkyKey>of());
          // Fall through to check.
        case CHECK:
          Preconditions.checkState(
              key.equals(reverseDeps), "%s %s %s", keyToConsolidate, reverseDeps, container);
          break;
        case ADD:
          throw new IllegalStateException(
              "Shouldn't delay add if only one element: "
                  + keyToConsolidate
                  + ", "
                  + reverseDeps
                  + ", "
                  + container);
        default:
          throw new IllegalStateException(keyToConsolidate + ", " + reverseDeps + ", " + container);
      }
      return;
    }
    List<SkyKey> reverseDepsAsList = (List<SkyKey>) reverseDeps;
    Set<SkyKey> reverseDepsAsSet = CompactHashSet.create(reverseDepsAsList);

    if (reverseDepsAsSet.size() != reverseDepsAsList.size()) {
      // We're about to crash. Try to print an informative error message.
      Set<SkyKey> seen = new HashSet<>();
      List<SkyKey> duplicates = new ArrayList<>();
      for (SkyKey key : reverseDepsAsList) {
        if (!seen.add(key)) {
          duplicates.add(key);
        }
      }
      throw new IllegalStateException(
          (reverseDepsAsList.size() - reverseDepsAsSet.size())
              + " duplicates: "
              + duplicates
              + " for "
              + container);
    }
    for (Object keyToConsolidate : dataToConsolidate) {
      SkyKey key = KeyToConsolidate.key(keyToConsolidate);
      switch (KeyToConsolidate.op(keyToConsolidate)) {
        case CHECK:
          Preconditions.checkState(
              reverseDepsAsSet.contains(key),
              "%s %s %s",
              keyToConsolidate,
              reverseDepsAsSet,
              container);
          break;
        case REMOVE:
          Preconditions.checkState(
              reverseDepsAsSet.remove(key), "%s %s %s", keyToConsolidate, reverseDeps, container);
          break;
        case ADD:
          Preconditions.checkState(
              reverseDepsAsSet.add(key), "%s %s %s", keyToConsolidate, reverseDeps, container);
          break;
        default:
          throw new IllegalStateException(
              keyToConsolidate + ", " + reverseDepsAsSet + ", " + container);
      }
    }

    if (reverseDepsAsSet.isEmpty()) {
      overwriteReverseDepsList(container, ImmutableList.<SkyKey>of());
    } else if (reverseDepsAsSet.size() == 1) {
      overwriteReverseDepsWithObject(container, Iterables.getOnlyElement(reverseDepsAsSet));
    } else {
      overwriteReverseDepsList(container, new ArrayList<>(reverseDepsAsSet));
    }
  }

  @Override
  public String toString(T container) {
    return MoreObjects.toStringHelper("ReverseDeps")
        .add("reverseDeps", getReverseDepsObject(container))
        .add("singleReverseDep", isSingleReverseDep(container))
        .add("dataToConsolidate", getDataToConsolidate(container))
        .toString();
  }

  private void overwriteReverseDepsWithObject(T container, SkyKey newObject) {
    setReverseDepsObject(container, newObject);
  }

  private void overwriteReverseDepsList(T container, List<SkyKey> list) {
    setReverseDepsObject(container, list);
  }
}
