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

/**
 * Calculate set of changed nodes between two {@link Version}s of a graph.
 */
interface Differencer {
  /**
   * Returns the set of node keys that have changed between the given versions.
   */
  Iterable<NodeKey> getDiff(Version baseVersion, Version newVersion);

  /**
   * Equivalent to calling {@link #getDiff} with baseVersion equal to the most recent version before
   * newVersion. Useful if the caller can guarantee that all changes before newVersion have
   * been processed.
   */
  Iterable<NodeKey> getDiffFromLast(Version newVersion);
}
