// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.MoreObjects.toStringHelper;

/** Type representing a directory listing operation. */
abstract sealed class ListingDependencies
    implements FileSystemDependencies.FileOpDependency,
        FileDependencyDeserializer.ListingDependenciesOrFuture
    permits ListingDependencies.AvailableListingDependencies,
        ListingDependencies.MissingListingDependencies {

  static ListingDependencies from(FileDependencies realDirectory) {
    if (realDirectory.isMissingData()) {
      return newMissingInstance();
    }
    return new AvailableListingDependencies(realDirectory);
  }

  static ListingDependencies newMissingInstance() {
    return new MissingListingDependencies();
  }

  static final class AvailableListingDependencies extends ListingDependencies {
    private final FileDependencies realDirectory;

    private AvailableListingDependencies(FileDependencies realDirectory) {
      this.realDirectory = realDirectory;
    }

    @Override
    public boolean isMissingData() {
      return false;
    }

    /**
     * Determines if this listing is invalidated by anything in {@code changes}.
     *
     * <p>The caller should ensure the following.
     *
     * <ul>
     *   <li>This listing is known to be valid at {@code validityHorizon} (VH).
     *   <li>All changes over the range {@code (VH, VC])} are registered with {@code changes} before
     *       calling this method. (VC is the synced version of the cache reader.)
     * </ul>
     *
     * <p>See description of {@link VersionedChanges} for more details.
     *
     * @return the earliest version where a matching (invalidating) change is identified, otherwise
     *     {@link VersionedChanges#NO_MATCH}.
     */
    int findEarliestMatch(VersionedChanges changes, int validityHorizon) {
      return changes.matchListingChange(realDirectory.resolvedPath(), validityHorizon);
    }

    FileDependencies realDirectory() {
      return realDirectory;
    }

    @Override
    public String toString() {
      return toStringHelper(this).add("realDirectory", realDirectory).toString();
    }
  }

  /**
   * Signals missing listing data.
   *
   * <p>This is deliberately not a singleton to avoid a memory leak in the weak-value caches in
   * {@link FileDependencyDeserializer}.
   */
  static final class MissingListingDependencies extends ListingDependencies {
    private MissingListingDependencies() {}

    @Override
    public boolean isMissingData() {
      return true;
    }
  }
}
