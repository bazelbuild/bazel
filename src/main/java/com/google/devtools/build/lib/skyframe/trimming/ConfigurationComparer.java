// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.trimming;

import java.util.function.BiFunction;

/**
 * Interface describing a function which compares two configurations and determines their
 * relationship.
 */
@FunctionalInterface
public interface ConfigurationComparer<ConfigurationT>
    extends BiFunction<ConfigurationT, ConfigurationT, ConfigurationComparer.Result> {
  /** The outcome of comparing two configurations. */
  public enum Result {
    /**
     * All fragments in the first configuration are present and equal in the second, and vice versa.
     */
    EQUAL(true, true, true),
    /**
     * All shared fragments are equal, but the second configuration has additional fragments the
     * first does not.
     */
    SUBSET(true, false, true),
    /**
     * All shared fragments are equal, but the first configuration has additional fragments the
     * second does not.
     */
    SUPERSET(false, true, true),
    /**
     * The two configurations each have fragments the other does not, but the shared fragments are
     * equal.
     */
    ALL_SHARED_FRAGMENTS_EQUAL(false, false, true),
    /** At least one fragment shared between the two configurations is unequal. */
    DIFFERENT(false, false, false);

    private final boolean isSubsetOrEqual;
    private final boolean isSupersetOrEqual;
    private final boolean hasEqualSharedFragments;

    Result(boolean isSubsetOrEqual, boolean isSupersetOrEqual, boolean hasEqualSharedFragments) {
      this.isSubsetOrEqual = isSubsetOrEqual;
      this.isSupersetOrEqual = isSupersetOrEqual;
      this.hasEqualSharedFragments = hasEqualSharedFragments;
    }

    public boolean isSubsetOrEqual() {
      return isSubsetOrEqual;
    }

    public boolean isSupersetOrEqual() {
      return isSupersetOrEqual;
    }

    public boolean hasEqualSharedFragments() {
      return hasEqualSharedFragments;
    }
  }
}
