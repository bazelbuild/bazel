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

import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * A utility class that allows us to keep the reverse dependencies as an array list instead of a
 * set. This is more memory-efficient. At the same time it allows us to group the removals and
 * uniqueness checks so that it also performs well.
 *
 * <p>The reason of this class it to share non-trivial code between BuildingState and NodeEntry. We
 * could simply make those two classes extend this class instead, but we would be less
 * memory-efficient since object memory alignment does not cross classes ( you would have two memory
 * alignments, one for the base class and one for the extended one).
 */
abstract class ReverseDepsUtil<T> {

  static final int MAYBE_CHECK_THRESHOLD = 10;

  abstract void setReverseDepsObject(T container, Object object);

  abstract void setSingleReverseDep(T container, boolean singleObject);

  abstract void setReverseDepsToRemove(T container, List<SkyKey> object);

  abstract Object getReverseDepsObject(T container);

  abstract boolean isSingleReverseDep(T container);

  abstract List<SkyKey> getReverseDepsToRemove(T container);

  /**
   * We check that the reverse dependency is not already present. We only do that if reverseDeps is
   * small, so that it does not impact performance.
   */
  void maybeCheckReverseDepNotPresent(T container, SkyKey reverseDep) {
    if (isSingleReverseDep(container)) {
      Preconditions.checkState(!getReverseDepsObject(container).equals(reverseDep),
          "Reverse dep %s already present", reverseDep);
      return;
    }
    @SuppressWarnings("unchecked")
    List<SkyKey> asList = (List<SkyKey>) getReverseDepsObject(container);
    if (asList.size() < MAYBE_CHECK_THRESHOLD) {
      Preconditions.checkState(!asList.contains(reverseDep), "Reverse dep %s already present"
          + " in %s", reverseDep, asList);
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
  void addReverseDeps(T container, Collection<SkyKey> newReverseDeps) {
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

  /**
   * See {@code addReverseDeps} method.
   */
  void removeReverseDep(T container, SkyKey reverseDep) {
    if (isSingleReverseDep(container)) {
      // This removal is cheap so let's do it and not keep it in reverseDepsToRemove.
      // !equals should only happen in case of catastrophe.
      if (getReverseDepsObject(container).equals(reverseDep)) {
        overwriteReverseDepsList(container, ImmutableList.<SkyKey>of());
      }
      return;
    }
    @SuppressWarnings("unchecked")
    List<SkyKey> reverseDepsAsList = (List<SkyKey>) getReverseDepsObject(container);
    if (reverseDepsAsList.isEmpty()) {
      return;
    }
    List<SkyKey> reverseDepsToRemove = getReverseDepsToRemove(container);
    if (reverseDepsToRemove == null) {
      reverseDepsToRemove = Lists.newArrayListWithExpectedSize(1);
      setReverseDepsToRemove(container, reverseDepsToRemove);
    }
    reverseDepsToRemove.add(reverseDep);
  }

  ImmutableSet<SkyKey> getReverseDeps(T container) {
    consolidateReverseDepsRemovals(container);

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

  void consolidateReverseDepsRemovals(T container) {
    List<SkyKey> reverseDepsToRemove = getReverseDepsToRemove(container);
    Object reverseDeps = getReverseDepsObject(container);
    if (reverseDepsToRemove == null) {
      return;
    }
    Preconditions.checkState(!isSingleReverseDep(container),
        "We do not use reverseDepsToRemove for single lists: %s", container);
    // Should not happen, as we only create reverseDepsToRemove in case we have at least one
    // reverse dep to remove.
    Preconditions.checkState((!((List<?>) reverseDeps).isEmpty()),
        "Could not remove %s elements from %s.\nReverse deps to remove: %s. %s",
        reverseDepsToRemove.size(), reverseDeps, reverseDepsToRemove, container);

    Set<SkyKey> toRemove = Sets.newHashSet(reverseDepsToRemove);
    int expectedRemovals = toRemove.size();
    Preconditions.checkState(expectedRemovals == reverseDepsToRemove.size(),
        "A reverse dependency tried to remove itself twice: %s. %s", reverseDepsToRemove,
        container);

    @SuppressWarnings("unchecked")
    List<SkyKey> reverseDepsAsList = (List<SkyKey>) reverseDeps;
    List<SkyKey> newReverseDeps = Lists
        .newArrayListWithExpectedSize(Math.max(0, reverseDepsAsList.size() - expectedRemovals));

    for (SkyKey reverseDep : reverseDepsAsList) {
      if (!toRemove.contains(reverseDep)) {
        newReverseDeps.add(reverseDep);
      }
    }
    Preconditions.checkState(newReverseDeps.size() == reverseDepsAsList.size() - expectedRemovals,
        "Could not remove some elements from %s.\nReverse deps to remove: %s. %s", reverseDeps,
        toRemove, container);

    if (newReverseDeps.isEmpty()) {
      overwriteReverseDepsList(container, ImmutableList.<SkyKey>of());
    } else if (newReverseDeps.size() == 1) {
      overwriteReverseDepsWithObject(container, newReverseDeps.get(0));
    } else {
      overwriteReverseDepsList(container, newReverseDeps);
    }
    setReverseDepsToRemove(container, null);
  }

  String toString(T container) {
    return MoreObjects.toStringHelper("ReverseDeps")
        .add("reverseDeps", getReverseDepsObject(container))
        .add("singleReverseDep", isSingleReverseDep(container))
        .add("reverseDepsToRemove", getReverseDepsToRemove(container))
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
