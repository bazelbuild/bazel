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

import java.util.Collection;
import java.util.Map;

/**
 * Calculate set of changed values in a graph.
 */
public interface Differencer {

  /**
   * Represents a set of changed values.
   */
  interface Diff {
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
    Map<SkyKey, ? extends SkyValue> changedKeysWithNewValues();
  }

  /**
   * Returns the value keys that have changed between the two Versions.
   */
  Diff getDiff(Version fromVersion, Version toVersion) throws InterruptedException;
}
