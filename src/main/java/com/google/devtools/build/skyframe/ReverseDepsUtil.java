// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

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
 * <p>The reason of this class it to share non-trivial code between BuildingState and NodeEntry. We
 * could simply make those two classes extend this class instead, but we would be less
 * memory-efficient since object memory alignment does not cross classes (you would have two memory
 * alignments, one for the base class and one for the extended one).
 *
 * <p>This class is public only for the benefit of alternative graph implementations outside of the
 * package.
 */
public abstract class ReverseDepsUtil<T> {

  static final int MAYBE_CHECK_THRESHOLD = 10;

  abstract void setReverseDepsObject(T container, Object object);

  abstract void setSingleReverseDep(T container, boolean singleObject);

  abstract void setDataToConsolidate(T container, Object dataToConsolidate);

  abstract Object getReverseDepsObject(T container);

  abstract boolean isSingleReverseDep(T container);

  abstract Object getDataToConsolidate(T container);

  private enum DataState {
    NULL,
    CHECK_ONLY,
    BOTH
  }

  private static class RDepsToCheckAndRemove {
    // reverseDepstoCheck may be null, but not toRemove, because we store a bare list if
    // we have only toCheck.
    @Nullable List<SkyKey> toCheck;
    List<SkyKey> toRemove;

    RDepsToCheckAndRemove(List<SkyKey> toCheck, List<SkyKey> toRemove) {
      this.toCheck = toCheck;
      this.toRemove = Preconditions.checkNotNull(toRemove);
    }

    static RDepsToCheckAndRemove justRemove(List<SkyKey> toRemove) {
      return new RDepsToCheckAndRemove(null, toRemove);
    }

    static RDepsToCheckAndRemove replaceRemove(
        RDepsToCheckAndRemove original, List<SkyKey> toRemove) {
      return new RDepsToCheckAndRemove(original.toCheck, toRemove);
    }

    static RDepsToCheckAndRemove replaceCheck(
        RDepsToCheckAndRemove original, List<SkyKey> toCheck) {
      return new RDepsToCheckAndRemove(toCheck, original.toRemove);
    }
  }

  private static DataState getDataState(Object reverseDepsToCheckAndRemove) {
    if (reverseDepsToCheckAndRemove == null) {
      return DataState.NULL;
    } else if (reverseDepsToCheckAndRemove instanceof List) {
      return DataState.CHECK_ONLY;
    } else {
      Preconditions.checkState(
          reverseDepsToCheckAndRemove instanceof RDepsToCheckAndRemove,
          reverseDepsToCheckAndRemove);
      return DataState.BOTH;
    }
  }

  private void setReverseDepsToRemove(T container, List<SkyKey> toRemove) {
    Object reverseDepsToCheckAndRemove = getDataToConsolidate(container);
    switch (getDataState(reverseDepsToCheckAndRemove)) {
      case NULL:
        reverseDepsToCheckAndRemove = RDepsToCheckAndRemove.justRemove(toRemove);
        break;
      case CHECK_ONLY:
        reverseDepsToCheckAndRemove =
            new RDepsToCheckAndRemove(((List<SkyKey>) reverseDepsToCheckAndRemove), toRemove);
        break;
      case BOTH:
        reverseDepsToCheckAndRemove =
            RDepsToCheckAndRemove.replaceRemove(
                (RDepsToCheckAndRemove) reverseDepsToCheckAndRemove, toRemove);
        break;
      default:
        throw new IllegalStateException(container + ", " + toRemove);
    }
    setDataToConsolidate(container, reverseDepsToCheckAndRemove);
  }

  private void setReverseDepsToCheck(T container, List<SkyKey> toCheck) {
    Object reverseDepsToCheckAndRemove = getDataToConsolidate(container);
    switch (getDataState(reverseDepsToCheckAndRemove)) {
      case NULL:
      case CHECK_ONLY:
        reverseDepsToCheckAndRemove = toCheck;
        break;
      case BOTH:
        reverseDepsToCheckAndRemove =
            RDepsToCheckAndRemove.replaceCheck(
                (RDepsToCheckAndRemove) reverseDepsToCheckAndRemove, toCheck);
        break;
      default:
        throw new IllegalStateException(container + ", " + toCheck);
    }
    setDataToConsolidate(container, reverseDepsToCheckAndRemove);
  }

  @Nullable
  private List<SkyKey> getReverseDepsToRemove(T container) {
    Object reverseDepsToCheckAndRemove = getDataToConsolidate(container);
    switch (getDataState(reverseDepsToCheckAndRemove)) {
      case NULL:
      case CHECK_ONLY:
        return null;
      case BOTH:
        return ((RDepsToCheckAndRemove) reverseDepsToCheckAndRemove).toRemove;
      default:
        throw new IllegalStateException(reverseDepsToCheckAndRemove.toString());
    }
  }

  @Nullable
  private List<SkyKey> getReverseDepsToCheck(T container) {
    Object reverseDepsToCheckAndRemove = getDataToConsolidate(container);
    switch (getDataState(reverseDepsToCheckAndRemove)) {
      case NULL:
        return null;
      case CHECK_ONLY:
        return (List<SkyKey>) reverseDepsToCheckAndRemove;
      case BOTH:
        return ((RDepsToCheckAndRemove) reverseDepsToCheckAndRemove).toCheck;
      default:
        throw new IllegalStateException(reverseDepsToCheckAndRemove.toString());
    }
  }

  /**
   * We check that the reverse dependency is not already present. We only do that if reverseDeps is
   * small, so that it does not impact performance.
   */
  void maybeCheckReverseDepNotPresent(T container, SkyKey reverseDep) {
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

  void checkReverseDep(T container, SkyKey reverseDep) {
    Object reverseDepsObject = getReverseDepsObject(container);
    if (isSingleReverseDep(container)) {
      Preconditions.checkState(
          reverseDepsObject.equals(reverseDep),
          "%s %s %s",
          reverseDep,
          reverseDepsObject,
          container);
      return;
    }
    List<SkyKey> asList = (List<SkyKey>) reverseDepsObject;
    if (asList.size() < MAYBE_CHECK_THRESHOLD) {
      Preconditions.checkState(
          asList.contains(reverseDep), "%s not in %s for %s", reverseDep, asList, container);
    } else {
      List<SkyKey> reverseDepsToCheck = getReverseDepsToCheck(container);
      if (reverseDepsToCheck == null) {
        reverseDepsToCheck = new ArrayList<>();
        setReverseDepsToCheck(container, reverseDepsToCheck);
      }
      reverseDepsToCheck.add(reverseDep);
    }
  }

  /**
   * See {@code addReverseDeps} method.
   */
  void removeReverseDep(T container, SkyKey reverseDep) {
    if (isSingleReverseDep(container)) {
      // This removal is cheap so let's do it and not keep it in reverseDepsToRemove.
      Preconditions.checkState(
          getReverseDepsObject(container).equals(reverseDep),
          "toRemove: %s container: %s",
          reverseDep,
          container);
      overwriteReverseDepsList(container, ImmutableList.<SkyKey>of());
      return;
    }
    @SuppressWarnings("unchecked")
    List<SkyKey> reverseDepsAsList = (List<SkyKey>) getReverseDepsObject(container);
    Preconditions.checkState(
        !reverseDepsAsList.isEmpty(), "toRemove: %s container: %s", reverseDep, container);
    List<SkyKey> reverseDepsToRemove = getReverseDepsToRemove(container);
    if (reverseDepsToRemove == null) {
      reverseDepsToRemove = Lists.newArrayListWithExpectedSize(1);
      setReverseDepsToRemove(container, reverseDepsToRemove);
    }
    reverseDepsToRemove.add(reverseDep);
  }

  ImmutableSet<SkyKey> getReverseDeps(T container) {
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
      Preconditions.checkState(set.size() == reverseDeps.size(),
          "Duplicate reverse deps present in %s: %s. %s", this, reverseDeps, container);
      return set;
    }
  }

  void consolidateData(T container) {
    Object reverseDeps = getReverseDepsObject(container);
    List<SkyKey> reverseDepsToRemove = getReverseDepsToRemove(container);
    List<SkyKey> reverseDepsToCheck = getReverseDepsToCheck(container);
    if (reverseDepsToRemove == null && reverseDepsToCheck == null) {
      return;
    }
    Preconditions.checkState(
        !isSingleReverseDep(container),
        "We do not delay removals/checks for single lists: %s %s %s",
        container,
        reverseDepsToRemove,
        reverseDepsToCheck);
    @SuppressWarnings("unchecked")
    List<SkyKey> reverseDepsAsList = (List<SkyKey>) reverseDeps;
    // Should not happen, as we only create reverseDepsToRemove/Check in case we have at least one
    // reverse dep to remove/check.
    Preconditions.checkState(
        !reverseDepsAsList.isEmpty(),
        "Could not do delayed removal/check for %s elements from %s.\n"
            + "Reverse deps to check: %s. Container: %s",
        reverseDepsToRemove,
        reverseDeps,
        reverseDepsToCheck,
        container);
    if (reverseDepsToRemove == null) {
      Set<SkyKey> reverseDepsAsSet = new HashSet<>(reverseDepsAsList);
      Preconditions.checkState(
          reverseDepsAsSet.containsAll(reverseDepsToCheck), "%s %s", reverseDepsToCheck, container);
      setDataToConsolidate(container, null);
      return;
    }

    Set<SkyKey> toRemove = Sets.newHashSet(reverseDepsToRemove);
    int expectedRemovals = toRemove.size();
    Preconditions.checkState(expectedRemovals == reverseDepsToRemove.size(),
        "A reverse dependency tried to remove itself twice: %s. %s", reverseDepsToRemove,
        container);

    Set<SkyKey> toCheck =
        reverseDepsToCheck == null ? new HashSet<SkyKey>() : new HashSet<>(reverseDepsToCheck);
    List<SkyKey> newReverseDeps = Lists
        .newArrayListWithExpectedSize(Math.max(0, reverseDepsAsList.size() - expectedRemovals));

    for (SkyKey reverseDep : reverseDepsAsList) {
      toCheck.remove(reverseDep);
      if (!toRemove.contains(reverseDep)) {
        newReverseDeps.add(reverseDep);
      }
    }
    Preconditions.checkState(newReverseDeps.size() == reverseDepsAsList.size() - expectedRemovals,
        "Could not remove some elements from %s.\nReverse deps to remove: %s. %s", reverseDeps,
        toRemove, container);
    Preconditions.checkState(toCheck.isEmpty(), "%s %s", toCheck, container);

    if (newReverseDeps.isEmpty()) {
      overwriteReverseDepsList(container, ImmutableList.<SkyKey>of());
    } else if (newReverseDeps.size() == 1) {
      overwriteReverseDepsWithObject(container, newReverseDeps.get(0));
    } else {
      overwriteReverseDepsList(container, newReverseDeps);
    }
    setDataToConsolidate(container, null);
  }

  String toString(T container) {
    return MoreObjects.toStringHelper("ReverseDeps")
        .add("reverseDeps", getReverseDepsObject(container))
        .add("singleReverseDep", isSingleReverseDep(container))
        .add("dataToConsolidate", getDataToConsolidate(container))
        .toString();
  }

  private void overwriteReverseDepsWithObject(T container, SkyKey newObject) {
    setReverseDepsObject(container, newObject);
    setSingleReverseDep(container, true);
  }

  private void overwriteReverseDepsList(T container, List<SkyKey> list) {
    setReverseDepsObject(container, list);
    setSingleReverseDep(container, false);
  }
}
