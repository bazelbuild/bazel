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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.skyframe.KeyToConsolidate.Op;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A utility class that allows us to store reverse dependencies in a memory-efficient way. At the
 * same time it allows us to group the removals and uniqueness checks so that it also performs well.
 *
 * <p>The operations {@link #addReverseDep} and {@link #removeReverseDep} here are optimized for a
 * done entry. Done entries rarely have rdeps added and removed, but do have {@link Op#CHECK}
 * operations performed frequently. As well, done node entries may never have their data forcibly
 * consolidated, since their reverse deps will only be retrieved as a whole if they are marked
 * dirty. Thus, we consolidate periodically.
 *
 * <p>{@link IncrementalInMemoryNodeEntry} manages pending reverse dep operations on a marked-dirty
 * or initially evaluating node itself, using similar logic tuned to those cases, and calls into
 * {@link #consolidateDataAndReturnNewElements} when transitioning to done.
 *
 * <p>The storage schema for reverse dependencies of done node entries is:
 *
 * <ul>
 *   <li>0 rdeps: empty {@link ImmutableList}
 *   <li>1 rdep: bare {@link SkyKey}
 *   <li>2-4 rdeps: {@code SkyKey[]} (no nulls)
 *   <li>5+ rdeps: an {@link ArrayList}
 * </ul>
 *
 * This strategy saves memory in the common case of few reverse deps while still supporting constant
 * time additions for nodes with many rdeps by dynamically switching to an {@link ArrayList}.
 */
abstract class ReverseDepsUtility {

  /**
   * Returns the {@link Op} to store bare instead of wrapping in {@link KeyToConsolidate}.
   *
   * <p>We can store one type of operation bare in order to save memory. For nodes on their initial
   * build and nodes not keeping reverse deps, most operations are {@link Op#ADD}.
   *
   * <p>Done nodes have very few delayed ops - {@link Op#CHECK} is never stored on a done node and
   * {@link Op#ADD} is only delayed if there are already pending delayed ops. Returning {@link
   * Op#CHECK} in this case just makes it easy to distinguish from nodes on their initial build.
   */
  static Op getOpToStoreBare(AbstractInMemoryNodeEntry<?> entry) {
    DirtyBuildingState dirtyBuildingState = entry.dirtyBuildingState;
    if (dirtyBuildingState == null) {
      return Op.CHECK;
    }
    return entry.keepsEdges() && dirtyBuildingState.isIncremental() ? Op.CHECK : Op.ADD;
  }

  private static void maybeDelayReverseDepOp(
      IncrementalInMemoryNodeEntry entry, SkyKey reverseDep, Op op) {
    List<Object> consolidations = entry.getReverseDepsDataToConsolidateForReverseDepsUtil();
    int currentReverseDepSize = sizeOf(entry.getReverseDepsRawForReverseDepsUtil());
    if (consolidations == null) {
      consolidations = new ArrayList<>(currentReverseDepSize);
      entry.setReverseDepsDataToConsolidateForReverseDepsUtil(consolidations);
    }
    consolidations.add(KeyToConsolidate.create(reverseDep, op, entry));
    // TODO(janakr): Should we consolidate more aggressively? This threshold can be customized.
    if (consolidations.size() >= currentReverseDepSize) {
      consolidateData(entry);
    }
  }

  private static boolean isSingleReverseDep(Object raw) {
    return raw instanceof SkyKey;
  }

  @SuppressWarnings("unchecked")
  private static List<SkyKey> multipleAsList(Object raw) {
    return raw instanceof SkyKey[] ? Arrays.asList((SkyKey[]) raw) : (List<SkyKey>) raw;
  }

  private static int sizeOf(Object raw) {
    if (isSingleReverseDep(raw)) {
      return 1;
    }
    if (raw instanceof SkyKey[]) {
      return ((SkyKey[]) raw).length;
    }
    return ((List<?>) raw).size();
  }

  @SuppressWarnings("unchecked") // Cast to List<SkyKey>.
  static void addReverseDep(IncrementalInMemoryNodeEntry entry, SkyKey newReverseDep) {
    List<Object> dataToConsolidate = entry.getReverseDepsDataToConsolidateForReverseDepsUtil();
    if (dataToConsolidate != null) {
      maybeDelayReverseDepOp(entry, newReverseDep, Op.ADD);
      return;
    }
    var raw = entry.getReverseDepsRawForReverseDepsUtil();
    int newSize = sizeOf(raw) + 1;
    if (newSize == 1) {
      entry.setReverseDepsForReverseDepsUtil(newReverseDep);
    } else if (newSize == 2) {
      entry.setReverseDepsForReverseDepsUtil(new SkyKey[] {(SkyKey) raw, newReverseDep});
    } else if (newSize <= 4) {
      SkyKey[] newArray = Arrays.copyOf((SkyKey[]) raw, newSize);
      newArray[newSize - 1] = newReverseDep;
      entry.setReverseDepsForReverseDepsUtil(newArray);
    } else if (newSize == 5) {
      List<SkyKey> newList = new ArrayList<>(8);
      Collections.addAll(newList, (SkyKey[]) raw);
      newList.add(newReverseDep);
      entry.setReverseDepsForReverseDepsUtil(newList);
    } else {
      ((List<SkyKey>) raw).add(newReverseDep);
    }
  }

  /** See {@link #addReverseDep} method. */
  static void removeReverseDep(IncrementalInMemoryNodeEntry entry, SkyKey reverseDep) {
    maybeDelayReverseDepOp(entry, reverseDep, Op.REMOVE);
  }

  static void removeReverseDepsMatching(
      IncrementalInMemoryNodeEntry entry, Set<SkyKey> deletedKeys) {
    consolidateData(entry);
    ImmutableSet<SkyKey> currentReverseDeps =
        ImmutableSet.copyOf(getReverseDeps(entry, /* checkConsistency= */ true));
    writeReverseDepsSet(entry, Sets.difference(currentReverseDeps, deletedKeys));
  }

  static ImmutableCollection<SkyKey> getReverseDeps(
      IncrementalInMemoryNodeEntry entry, boolean checkConsistency) {
    consolidateData(entry);

    // TODO(bazel-team): Unfortunately, we need to make a copy here right now to be on the safe side
    // wrt. thread-safety. The parents of a node get modified when any of the parents is deleted,
    // and we can't handle that right now.
    var raw = entry.getReverseDepsRawForReverseDepsUtil();
    if (isSingleReverseDep(raw)) {
      return ImmutableSet.of((SkyKey) raw);
    } else {
      List<SkyKey> reverseDeps = multipleAsList(raw);
      if (!checkConsistency) {
        return ImmutableList.copyOf(reverseDeps);
      }
      ImmutableSet<SkyKey> set = ImmutableSet.copyOf(reverseDeps);
      Preconditions.checkState(
          set.size() == reverseDeps.size(),
          "Duplicate reverse deps present in %s: %s",
          reverseDeps,
          entry);
      return set;
    }
  }

  static Set<SkyKey> returnNewElements(IncrementalInMemoryNodeEntry entry) {
    return consolidateDataAndReturnNewElements(entry, /* mutateObject= */ false);
  }

  private static Set<SkyKey> consolidateDataAndReturnNewElements(
      IncrementalInMemoryNodeEntry entry, boolean mutateObject) {
    List<Object> dataToConsolidate = entry.getReverseDepsDataToConsolidateForReverseDepsUtil();
    if (dataToConsolidate == null) {
      return ImmutableSet.of();
    }

    // On a node's initial build (or if not keeping rdeps), we don't need to build up two different
    // sets for "all reverse deps" and "new reverse deps" - they are all new.
    Op opToStoreBare = getOpToStoreBare(entry);
    boolean allRdepsAreNew = opToStoreBare == Op.ADD;
    Set<SkyKey> allReverseDeps;
    Set<SkyKey> newReverseDeps;
    var raw = entry.getReverseDepsRawForReverseDepsUtil();
    if (isSingleReverseDep(raw)) {
      Preconditions.checkState(!allRdepsAreNew, entry);
      allReverseDeps = CompactHashSet.create((SkyKey) raw);
      newReverseDeps = CompactHashSet.create();
    } else {
      List<SkyKey> reverseDepsAsList = multipleAsList(raw);
      if (allRdepsAreNew) {
        Preconditions.checkState(reverseDepsAsList.isEmpty(), entry);
        allReverseDeps = null;
        newReverseDeps = CompactHashSet.createWithExpectedSize(dataToConsolidate.size());
      } else {
        allReverseDeps = getReverseDepsSet(entry, reverseDepsAsList);
        newReverseDeps = CompactHashSet.create();
      }
    }

    for (Object keyToConsolidate : dataToConsolidate) {
      SkyKey key = KeyToConsolidate.key(keyToConsolidate);
      switch (KeyToConsolidate.op(keyToConsolidate, opToStoreBare)) {
        case CHECK:
          Preconditions.checkState(!allRdepsAreNew, entry);
          Preconditions.checkState(
              allReverseDeps.contains(key),
              "Reverse dep not present: %s %s %s %s",
              keyToConsolidate,
              allReverseDeps,
              dataToConsolidate,
              entry);
          Preconditions.checkState(
              newReverseDeps.add(key),
              "Duplicate new reverse dep: %s %s %s %s",
              keyToConsolidate,
              allReverseDeps,
              dataToConsolidate,
              entry);
          break;
        case REMOVE:
          if (!allRdepsAreNew) {
            Preconditions.checkState(
                allReverseDeps.remove(key),
                "Reverse dep to be removed not present: %s %s %s %s",
                keyToConsolidate,
                allReverseDeps,
                dataToConsolidate,
                entry);
          }
          Preconditions.checkState(
              newReverseDeps.remove(key),
              "Reverse dep to be removed not present: %s %s %s %s",
              keyToConsolidate,
              newReverseDeps,
              dataToConsolidate,
              entry);
          break;
        case REMOVE_OLD:
          Preconditions.checkState(!allRdepsAreNew, entry);
          Preconditions.checkState(
              allReverseDeps.remove(key),
              "Reverse dep to be removed not present: %s %s %s %s",
              keyToConsolidate,
              allReverseDeps,
              dataToConsolidate,
              entry);
          Preconditions.checkState(
              !newReverseDeps.contains(key),
              "Reverse dep shouldn't have been added to new: %s %s %s %s",
              keyToConsolidate,
              allReverseDeps,
              dataToConsolidate,
              entry);
          break;
        case ADD:
          if (!allRdepsAreNew) {
            Preconditions.checkState(
                allReverseDeps.add(key),
                "Duplicate reverse deps: %s %s %s %s",
                keyToConsolidate,
                raw,
                dataToConsolidate,
                entry);
          }
          Preconditions.checkState(
              newReverseDeps.add(key),
              "Duplicate new reverse deps: %s %s %s %s",
              keyToConsolidate,
              raw,
              dataToConsolidate,
              entry);
          break;
      }
    }
    if (mutateObject) {
      entry.setReverseDepsDataToConsolidateForReverseDepsUtil(null);
      writeReverseDepsSet(entry, allRdepsAreNew ? newReverseDeps : allReverseDeps);
    }
    return newReverseDeps;
  }

  @CanIgnoreReturnValue
  static Set<SkyKey> consolidateDataAndReturnNewElements(IncrementalInMemoryNodeEntry entry) {
    return consolidateDataAndReturnNewElements(entry, /* mutateObject= */ true);
  }

  private static void consolidateData(IncrementalInMemoryNodeEntry entry) {
    List<Object> dataToConsolidate = entry.getReverseDepsDataToConsolidateForReverseDepsUtil();
    if (dataToConsolidate == null) {
      return;
    }
    entry.setReverseDepsDataToConsolidateForReverseDepsUtil(null);
    var raw = entry.getReverseDepsRawForReverseDepsUtil();
    if (isSingleReverseDep(raw)) {
      Preconditions.checkState(
          dataToConsolidate.size() == 1,
          "dataToConsolidate not size 1 even though only one rdep: %s %s %s",
          dataToConsolidate,
          raw,
          entry);
      Object keyToConsolidate = Iterables.getOnlyElement(dataToConsolidate);
      SkyKey key = KeyToConsolidate.key(keyToConsolidate);
      Preconditions.checkState(getOpToStoreBare(entry) == Op.CHECK, entry);
      switch (KeyToConsolidate.op(keyToConsolidate, Op.CHECK)) {
        case REMOVE:
          entry.setReverseDepsForReverseDepsUtil(ImmutableList.of());
          // Fall through to check.
        case CHECK:
          Preconditions.checkState(key.equals(raw), "%s %s %s", keyToConsolidate, raw, entry);
          break;
        case ADD:
          throw new IllegalStateException(
              "Shouldn't delay add if only one element: "
                  + keyToConsolidate
                  + ", "
                  + raw
                  + ", "
                  + entry);
        case REMOVE_OLD:
          throw new IllegalStateException(
              "Shouldn't be removing old deps if node already done: "
                  + keyToConsolidate
                  + ", "
                  + raw
                  + ", "
                  + entry);
      }
      return;
    }
    List<SkyKey> reverseDepsAsList = multipleAsList(raw);
    Set<SkyKey> reverseDepsAsSet = getReverseDepsSet(entry, reverseDepsAsList);

    Preconditions.checkState(getOpToStoreBare(entry) == Op.CHECK, entry);
    for (Object keyToConsolidate : dataToConsolidate) {
      SkyKey key = KeyToConsolidate.key(keyToConsolidate);
      switch (KeyToConsolidate.op(keyToConsolidate, Op.CHECK)) {
        case CHECK:
          Preconditions.checkState(
              reverseDepsAsSet.contains(key),
              "%s %s %s %s",
              keyToConsolidate,
              reverseDepsAsSet,
              dataToConsolidate,
              entry);
          break;
        case REMOVE:
          Preconditions.checkState(
              reverseDepsAsSet.remove(key),
              "%s %s %s %s",
              keyToConsolidate,
              raw,
              dataToConsolidate,
              entry);
          break;
        case ADD:
          Preconditions.checkState(
              reverseDepsAsSet.add(key),
              "%s %s %s %s",
              keyToConsolidate,
              raw,
              dataToConsolidate,
              entry);
          break;
        case REMOVE_OLD:
          throw new IllegalStateException(
              "Shouldn't be removing old deps if node already done: "
                  + keyToConsolidate
                  + ", "
                  + raw
                  + ", "
                  + dataToConsolidate
                  + ", "
                  + entry);
      }
    }
    writeReverseDepsSet(entry, reverseDepsAsSet);
  }

  private static void writeReverseDepsSet(
      IncrementalInMemoryNodeEntry entry, Set<SkyKey> reverseDepsAsSet) {
    if (!entry.keepsEdges() || reverseDepsAsSet.isEmpty()) {
      entry.setReverseDepsForReverseDepsUtil(ImmutableList.of());
    } else if (reverseDepsAsSet.size() == 1) {
      entry.setReverseDepsForReverseDepsUtil(Iterables.getOnlyElement(reverseDepsAsSet));
    } else if (reverseDepsAsSet.size() <= 4) {
      entry.setReverseDepsForReverseDepsUtil(reverseDepsAsSet.toArray(SkyKey[]::new));
    } else {
      entry.setReverseDepsForReverseDepsUtil(new ArrayList<>(reverseDepsAsSet));
    }
  }

  private static Set<SkyKey> getReverseDepsSet(
      IncrementalInMemoryNodeEntry entry, List<SkyKey> reverseDepsAsList) {
    Set<SkyKey> reverseDepsAsSet = CompactHashSet.create(reverseDepsAsList);
    checkForDuplicates(reverseDepsAsSet, reverseDepsAsList, entry);
    return reverseDepsAsSet;
  }

  static void checkForDuplicates(
      Set<SkyKey> reverseDepsAsSet, List<SkyKey> reverseDepsAsList, InMemoryNodeEntry entry) {
    if (reverseDepsAsSet.size() == reverseDepsAsList.size()) {
      return;
    }
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
            + " duplicate reverse deps: "
            + duplicates
            + " for "
            + entry);
  }

  static String toString(IncrementalInMemoryNodeEntry entry) {
    return MoreObjects.toStringHelper("ReverseDeps")
        .add("reverseDeps", entry.getReverseDepsRawForReverseDepsUtil())
        .add("singleReverseDep", isSingleReverseDep(entry))
        .add("dataToConsolidate", entry.getReverseDepsDataToConsolidateForReverseDepsUtil())
        .toString();
  }

  private ReverseDepsUtility() {}
}
