// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;

import com.android.ide.common.res2.MergingException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Handles the Merging of AndroidDataSet.
 */
public class AndroidDataMerger {

  /** An internal class that handles the homogenization of AndroidDataSets. */
  // TODO(corysmith): Move this functionality to AndroidDataSet?
  private static class ResourceMap {

    private final Map<FullyQualifiedName, DataResource> overwritableResources;
    private final Set<MergeConflict> conflicts;
    private final Map<FullyQualifiedName, DataResource> nonOverwritingResourceMap;

    private ResourceMap(
        Map<FullyQualifiedName, DataResource> overwritableResources,
        Set<MergeConflict> conflicts,
        Map<FullyQualifiedName, DataResource> nonOverwritingResourceMap) {
      this.overwritableResources = overwritableResources;
      this.conflicts = conflicts;
      this.nonOverwritingResourceMap = nonOverwritingResourceMap;
    }

    /**
     * Creates ResourceMap from an AndroidDataSet.
     */
    static ResourceMap from(AndroidDataSet data) {
      Map<FullyQualifiedName, DataResource> overwritingResourceMap = new HashMap<>();
      Set<MergeConflict> conflicts = new HashSet<>();
      for (DataResource resource : data.getOverwritingResources()) {
        if (overwritingResourceMap.containsKey(resource.fullyQualifiedName())) {
          conflicts.add(
              MergeConflict.between(
                  resource.fullyQualifiedName(),
                  overwritingResourceMap.get(resource.fullyQualifiedName()),
                  resource));
        }
        overwritingResourceMap.put(resource.fullyQualifiedName(), resource);
      }

      Map<FullyQualifiedName, DataResource> nonOverwritingResourceMap = new HashMap<>();
      for (DataResource resource : data.getNonOverwritingResources()) {
        nonOverwritingResourceMap.put(resource.fullyQualifiedName(), resource);
      }
      return new ResourceMap(overwritingResourceMap, conflicts, nonOverwritingResourceMap);
    }

    boolean containsOverwritable(FullyQualifiedName name) {
      return overwritableResources.containsKey(name);
    }

    Iterable<Map.Entry<FullyQualifiedName, DataResource>> iterateOverwritableEntries() {
      return overwritableResources.entrySet();
    }

    public MergeConflict foundConflict(FullyQualifiedName key, DataResource value) {
      return MergeConflict.between(key, overwritableResources.get(key), value);
    }

    public Collection<DataResource> mergeNonOverwritable(ResourceMap other) {
      Map<FullyQualifiedName, DataResource> merged = new HashMap<>(other.nonOverwritingResourceMap);
      merged.putAll(nonOverwritingResourceMap);
      return merged.values();
    }
  }

  /**
   * Merges DataResources into an UnwrittenMergedAndroidData.
   *
   * This method has two basic states, library and binary. These are distinguished by
   * allowPrimaryOverrideAll, which allows the primary data to overwrite any value in the closure,
   * a trait associated with binaries, as a binary is a leaf node. The other semantics are
   * slightly more complicated: a given resource can be overwritten only if it resides in the
   * direct dependencies of primary data. This forces an explicit simple priority for each resource,
   * instead of the more subtle semantics of multiple layers of libraries with potential overwrites.
   *
   * The UnwrittenMergedAndroidData contains only one of each FullyQualifiedName in both the
   * direct and transitive closure.
   *
   * The merge semantics are as follows:
   *   Key:
   *     A(): package A
   *     A(foo): package A with resource symbol foo
   *     A() -> B(): a dependency relationship of B.deps = [:A]
   *     A(),B() -> C(): a dependency relationship of C.deps = [:A,:B]
   *
   *   For android library (allowPrimaryOverrideAll = False)
   *
   *     A() -> B(foo) -> C(foo) == Valid
   *     A() -> B() -> C(foo) == Valid
   *     A() -> B() -> C(foo),D(foo) == Conflict
   *     A(foo) -> B(foo) -> C() == Conflict
   *     A(foo) -> B() -> C(foo) == Conflict
   *     A(foo),B(foo) -> C() -> D() == Conflict
   *     A() -> B(foo),C(foo) -> D() == Conflict
   *     A(foo),B(foo) -> C() -> D(foo) == Conflict
   *     A() -> B(foo),C(foo) -> D(foo) == Conflict
   *
   *   For android binary (allowPrimaryOverrideAll = True)
   *
   *     A() -> B(foo) -> C(foo) == Valid
   *     A() -> B() -> C(foo) == Valid
   *     A() -> B() -> C(foo),D(foo) == Conflict
   *     A(foo) -> B(foo) -> C() == Conflict
   *     A(foo) -> B() -> C(foo) == Valid
   *     A(foo),B(foo) -> C() -> D() == Conflict
   *     A() -> B(foo),C(foo) -> D() == Conflict
   *     A(foo),B(foo) -> C() -> D(foo) == Valid
   *     A() -> B(foo),C(foo) -> D(foo) == Valid
   *
   * @param transitive The transitive dependencies to merge.
   * @param direct The direct dependencies to merge.
   * @param primaryData The primary data to merge against.
   * @param allowPrimaryOverrideAll Boolean that indicates if the primary data will be considered
   *                                the ultimate source of truth, provided it doesn't conflict
   *                                with itself.
   * @return An UnwrittenMergedAndroidData, containing DataResource objects that can be written
   * to disk for aapt processing or serialized for future merge passes.
   * @throws MergingException if there are merge conflicts or issues with parsing resources from
   * Primary.
   * @throws IOException if there are issues with reading resources.
   */
  UnwrittenMergedAndroidData merge(
      AndroidDataSet transitive,
      AndroidDataSet direct,
      UnvalidatedAndroidData primaryData,
      boolean allowPrimaryOverrideAll)
      throws MergingException, IOException {

    // Extract the primary resources.
    AndroidDataSet primary = AndroidDataSet.from(primaryData);
    ResourceMap primaryMap = ResourceMap.from(primary);

    // Handle the overwriting resources first.
    ResourceMap directMap = ResourceMap.from(direct);
    ResourceMap transitiveMap = ResourceMap.from(transitive);

    List<DataResource> overwritableDeps = new ArrayList<>();

    Set<MergeConflict> conflicts = new HashSet<>();
    conflicts.addAll(primaryMap.conflicts);
    for (MergeConflict conflict : directMap.conflicts) {
      if (allowPrimaryOverrideAll
          && primaryMap.containsOverwritable(conflict.fullyQualifiedName())) {
        continue;
      }
      conflicts.add(conflict);
    }

    for (MergeConflict conflict : transitiveMap.conflicts) {
      if (allowPrimaryOverrideAll
          && primaryMap.containsOverwritable(conflict.fullyQualifiedName())) {
        continue;
      }
      conflicts.add(conflict);
    }

    for (Map.Entry<FullyQualifiedName, DataResource> entry :
        directMap.iterateOverwritableEntries()) {
      // Direct dependencies are simply overwritten, no conflict.
      if (!primaryMap.containsOverwritable(entry.getKey())) {
        overwritableDeps.add(entry.getValue());
      }
    }

    for (Map.Entry<FullyQualifiedName, DataResource> entry :
        transitiveMap.iterateOverwritableEntries()) {
      // If the primary is considered to be intentional (usually at the binary level),
      // skip.
      if (primaryMap.containsOverwritable(entry.getKey()) && allowPrimaryOverrideAll) {
        continue;
      }
      // If a transitive value is in the direct map report a conflict, as it is commonly
      // unintentional.
      if (directMap.containsOverwritable(entry.getKey())) {
        conflicts.add(directMap.foundConflict(entry.getKey(), entry.getValue()));
      } else if (primaryMap.containsOverwritable(entry.getKey())) {
        // If overwriting a transitive value with a primary map, assume it's an unintentional
        // override, unless allowPrimaryOverrideAll is set. At which point, this code path
        // should not be reached.
        conflicts.add(primaryMap.foundConflict(entry.getKey(), entry.getValue()));
      } else {
        // If it's in none of the of sources, add it.
        overwritableDeps.add(entry.getValue());
      }
    }

    if (!conflicts.isEmpty()) {
      List<String> messages = new ArrayList<>();
      for (MergeConflict conflict : conflicts) {
        messages.add(conflict.toConflictMessage());
      }
      throw new MergingException(Joiner.on("\n").join(messages));
    }

    Collections.sort(overwritableDeps);

    return UnwrittenMergedAndroidData.of(
        primaryData.getManifest(),
        primary,
        AndroidDataSet.of(
            overwritableDeps, ImmutableList.copyOf(directMap.mergeNonOverwritable(transitiveMap))));
  }
}
