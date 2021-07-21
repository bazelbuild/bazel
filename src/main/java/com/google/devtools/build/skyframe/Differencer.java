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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.Maps;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Calculate set of changed values in a graph.
 */
public interface Differencer {

  /**
   * Represents a set of changed values.
   */
  interface Diff {
    default boolean isEmpty() {
      return changedKeysWithoutNewValues().isEmpty() && changedKeysWithNewValues().isEmpty();
    }

    /**
     * Returns the value keys whose values have changed, but for which we don't have the new values.
     */
    Collection<SkyKey> changedKeysWithoutNewValues();

    /**
     * Returns the value keys whose values have changed, along with their new values.
     *
     * <p> The values in here cannot have any dependencies. This is required in order to prevent
     * conflation of injected values and derived values.
     */
    Map<SkyKey, SkyValue> changedKeysWithNewValues();
  }

  /** A {@link Diff} that also potentially contains the new and old values for each changed key. */
  interface DiffWithDelta extends Diff {
    /** Returns the value keys whose values have changed, along with their old and new values. */
    Map<SkyKey, Delta> changedKeysWithNewAndOldValues();

    /** Represents the delta between two values of the same key. */
    final class Delta {
      @Nullable private final SkyValue oldValue;
      private final SkyValue newValue;

      public Delta(SkyValue newValue) {
        this(null, newValue);
      }

      public Delta(SkyValue oldValue, SkyValue newValue) {
        this.oldValue = oldValue;
        this.newValue = checkNotNull(newValue);
      }

      /** Returns the old value, if any. */
      @Nullable
      public SkyValue getOldValue() {
        return oldValue;
      }

      /** Returns the new value. */
      public SkyValue getNewValue() {
        return newValue;
      }

      public static Map<SkyKey, SkyValue> newValues(Map<SkyKey, Delta> delta) {
        return Maps.transformValues(delta, Delta::getNewValue);
      }
    }
  }

  /**
   * Returns the value keys that have changed between the two Versions.
   */
  Diff getDiff(WalkableGraph fromGraph, Version fromVersion, Version toVersion)
      throws InterruptedException;
}
