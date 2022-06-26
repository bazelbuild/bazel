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
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.skyframe.KeyToConsolidate.Op;
import com.google.devtools.build.skyframe.NodeEntry.KeepEdgesPolicy;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A utility class that allows us to keep the reverse dependencies as an array list instead of a
 * set. This is more memory-efficient. At the same time it allows us to group the removals and
 * uniqueness checks so that it also performs well.
 *
 * <p>We could simply make {@link InMemoryNodeEntry} extend this class, but we would be less
 * memory-efficient since object memory alignment does not cross classes (you would have two memory
 * alignments, one for the base class and one for the extended one). We could also merge this
 * functionality directly into {@link InMemoryNodeEntry} at the cost of increasing its size and
 * complexity even more.
 *
 * <p>The operations {@link #addReverseDep}, {@link #checkReverseDep}, and {@link #removeReverseDep}
 * here are optimized for a done entry. Done entries rarely have rdeps added and removed, but do
 * have {@link Op#CHECK} operations performed frequently. As well, done node entries may never have
 * their data forcibly consolidated, since their reverse deps will only be retrieved as a whole if
 * they are marked dirty. Thus, we consolidate periodically.
 *
 * <p>{@link InMemoryNodeEntry} manages pending reverse dep operations on a marked-dirty or
 * initially evaluating node itself, using similar logic tuned to those cases, and calls into {@link
 * #consolidateDataAndReturnNewElements} when transitioning to done.
 */
abstract class ReverseDepsUtility {

  /**
   * Returns the {@link Op} to store bare instead of wrapping in {@link KeyToConsolidate}.
   *
   * <p>We can store one type of operation bare in order to save memory. For done nodes, most
   * operations are {@link Op#CHECK}. For nodes on their initial build and nodes not keeping reverse
   * deps, most are {@link Op#ADD}.
   */
  static Op getOpToStoreBare(InMemoryNodeEntry entry) {
    DirtyBuildingState dirtyBuildingState = entry.dirtyBuildingState;
    if (dirtyBuildingState == null) {
      return Op.CHECK;
    }
    switch (entry.keepEdges()) {
      case ALL:
        return dirtyBuildingState.isDirty() ? Op.CHECK : Op.ADD;
      case NONE:
      case JUST_DEPS:
        return Op.ADD;
    }
    throw new AssertionError(entry);
  }

  private static void maybeDelayReverseDepOp(InMemoryNodeEntry entry, SkyKey reverseDep, Op op) {
    List<Object> consolidations = entry.getReverseDepsDataToConsolidateForReverseDepsUtil();
    int currentReverseDepSize = getCurrentReverseDepSize(entry);
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

  private static boolean isSingleReverseDep(InMemoryNodeEntry entry) {
    return !(entry.getReverseDepsRawForReverseDepsUtil() instanceof List);
  }

  @SuppressWarnings("unchecked") // Cast to list.
  private static int getCurrentReverseDepSize(InMemoryNodeEntry entry) {
    return isSingleReverseDep(entry)
        ? 1
        : ((List<SkyKey>) entry.getReverseDepsRawForReverseDepsUtil()).size();
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
  @SuppressWarnings("unchecked") // Cast to SkyKey and List.
  static void addReverseDep(InMemoryNodeEntry entry, SkyKey newReverseDep) {
    List<Object> dataToConsolidate = entry.getReverseDepsDataToConsolidateForReverseDepsUtil();
    if (dataToConsolidate != null) {
      maybeDelayReverseDepOp(entry, newReverseDep, Op.ADD);
      return;
    }
    Object reverseDeps = entry.getReverseDepsRawForReverseDepsUtil();
    int reverseDepsSize = isSingleReverseDep(entry) ? 1 : ((List<SkyKey>) reverseDeps).size();
    if (reverseDepsSize == 0) {
      entry.setSingleReverseDepForReverseDepsUtil(newReverseDep);
    } else if (reverseDepsSize == 1) {
      List<SkyKey> newList = Lists.newArrayListWithExpectedSize(2);
      newList.add((SkyKey) reverseDeps);
      newList.add(newReverseDep);
      entry.setReverseDepsForReverseDepsUtil(newList);
    } else {
      ((List<SkyKey>) reverseDeps).add(newReverseDep);
    }
  }

  static void checkReverseDep(InMemoryNodeEntry entry, SkyKey reverseDep) {
    maybeDelayReverseDepOp(entry, reverseDep, Op.CHECK);
  }

  /** See {@link #addReverseDep} method. */
  static void removeReverseDep(InMemoryNodeEntry entry, SkyKey reverseDep) {
    maybeDelayReverseDepOp(entry, reverseDep, Op.REMOVE);
  }

  static void removeReverseDepsMatching(InMemoryNodeEntry entry, Set<SkyKey> deletedKeys) {
    consolidateData(entry);
    ImmutableSet<SkyKey> currentReverseDeps =
        ImmutableSet.copyOf(getReverseDeps(entry, /*checkConsistency=*/ true));
    writeReverseDepsSet(entry, Sets.difference(currentReverseDeps, deletedKeys));
  }

  static ImmutableCollection<SkyKey> getReverseDeps(
      InMemoryNodeEntry entry, boolean checkConsistency) {
    consolidateData(entry);

    // TODO(bazel-team): Unfortunately, we need to make a copy here right now to be on the safe side
    // wrt. thread-safety. The parents of a node get modified when any of the parents is deleted,
    // and we can't handle that right now.
    if (isSingleReverseDep(entry)) {
      return ImmutableSet.of((SkyKey) entry.getReverseDepsRawForReverseDepsUtil());
    } else {
      @SuppressWarnings("unchecked")
      List<SkyKey> reverseDeps = (List<SkyKey>) entry.getReverseDepsRawForReverseDepsUtil();
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

  static Set<SkyKey> returnNewElements(InMemoryNodeEntry entry) {
    return consolidateDataAndReturnNewElements(entry, /*mutateObject=*/ false);
  }

  @SuppressWarnings("unchecked") // List and bare SkyKey casts.
  private static Set<SkyKey> consolidateDataAndReturnNewElements(
      InMemoryNodeEntry entry, boolean mutateObject) {
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
    Object reverseDeps = entry.getReverseDepsRawForReverseDepsUtil();
    if (isSingleReverseDep(entry)) {
      Preconditions.checkState(!allRdepsAreNew, entry);
      allReverseDeps = CompactHashSet.create((SkyKey) reverseDeps);
      newReverseDeps = CompactHashSet.create();
    } else {
      List<SkyKey> reverseDepsAsList = (List<SkyKey>) reverseDeps;
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
                reverseDeps,
                dataToConsolidate,
                entry);
          }
          Preconditions.checkState(
              newReverseDeps.add(key),
              "Duplicate new reverse deps: %s %s %s %s",
              keyToConsolidate,
              reverseDeps,
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

  static Set<SkyKey> consolidateDataAndReturnNewElements(InMemoryNodeEntry entry) {
    return consolidateDataAndReturnNewElements(entry, /*mutateObject=*/ true);
  }

  @SuppressWarnings("unchecked") // Casts to SkyKey and List.
  private static void consolidateData(InMemoryNodeEntry entry) {
    List<Object> dataToConsolidate = entry.getReverseDepsDataToConsolidateForReverseDepsUtil();
    if (dataToConsolidate == null) {
      return;
    }
    entry.setReverseDepsDataToConsolidateForReverseDepsUtil(null);
    Object reverseDeps = entry.getReverseDepsRawForReverseDepsUtil();
    if (isSingleReverseDep(entry)) {
      Preconditions.checkState(
          dataToConsolidate.size() == 1,
          "dataToConsolidate not size 1 even though only one rdep: %s %s %s",
          dataToConsolidate,
          reverseDeps,
          entry);
      Object keyToConsolidate = Iterables.getOnlyElement(dataToConsolidate);
      SkyKey key = KeyToConsolidate.key(keyToConsolidate);
      Preconditions.checkState(getOpToStoreBare(entry) == Op.CHECK, entry);
      switch (KeyToConsolidate.op(keyToConsolidate, Op.CHECK)) {
        case REMOVE:
          entry.setReverseDepsForReverseDepsUtil(ImmutableList.of());
          // Fall through to check.
        case CHECK:
          Preconditions.checkState(
              key.equals(reverseDeps), "%s %s %s", keyToConsolidate, reverseDeps, entry);
          break;
        case ADD:
          throw new IllegalStateException(
              "Shouldn't delay add if only one element: "
                  + keyToConsolidate
                  + ", "
                  + reverseDeps
                  + ", "
                  + entry);
        case REMOVE_OLD:
          throw new IllegalStateException(
              "Shouldn't be removing old deps if node already done: "
                  + keyToConsolidate
                  + ", "
                  + reverseDeps
                  + ", "
                  + entry);
        default:
          throw new IllegalStateException(keyToConsolidate + ", " + reverseDeps + ", " + entry);
      }
      return;
    }
    List<SkyKey> reverseDepsAsList = (List<SkyKey>) reverseDeps;
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
              reverseDeps,
              dataToConsolidate,
              entry);
          break;
        case ADD:
          Preconditions.checkState(
              reverseDepsAsSet.add(key),
              "%s %s %s %s",
              keyToConsolidate,
              reverseDeps,
              dataToConsolidate,
              entry);
          break;
        case REMOVE_OLD:
          throw new IllegalStateException(
              "Shouldn't be removing old deps if node already done: "
                  + keyToConsolidate
                  + ", "
                  + reverseDeps
                  + ", "
                  + dataToConsolidate
                  + ", "
                  + entry);
        default:
          throw new IllegalStateException(
              keyToConsolidate + ", " + reverseDepsAsSet + ", " + dataToConsolidate + ", " + entry);
      }
    }
    writeReverseDepsSet(entry, reverseDepsAsSet);
  }

  private static void writeReverseDepsSet(InMemoryNodeEntry entry, Set<SkyKey> reverseDepsAsSet) {
    if (entry.keepEdges() != KeepEdgesPolicy.ALL || reverseDepsAsSet.isEmpty()) {
      entry.setReverseDepsForReverseDepsUtil(ImmutableList.of());
    } else if (reverseDepsAsSet.size() == 1) {
      entry.setSingleReverseDepForReverseDepsUtil(Iterables.getOnlyElement(reverseDepsAsSet));
    } else {
      entry.setReverseDepsForReverseDepsUtil(new ArrayList<>(reverseDepsAsSet));
    }
  }

  private static Set<SkyKey> getReverseDepsSet(
      InMemoryNodeEntry entry, List<SkyKey> reverseDepsAsList) {
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
              + entry);
    }
    return reverseDepsAsSet;
  }

  static String toString(InMemoryNodeEntry entry) {
    return MoreObjects.toStringHelper("ReverseDeps")
        .add("reverseDeps", entry.getReverseDepsRawForReverseDepsUtil())
        .add("singleReverseDep", isSingleReverseDep(entry))
        .add("dataToConsolidate", entry.getReverseDepsDataToConsolidateForReverseDepsUtil())
        .toString();
  }
  private ReverseDepsUtility() {}
}
