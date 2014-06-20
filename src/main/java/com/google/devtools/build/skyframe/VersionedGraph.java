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

import javax.annotation.Nullable;

/**
 * The graph that represents the internal data structures of Blaze.
 */
interface VersionedGraph {
  /**
   * Returns the graph at the specified version or null if it hasn't yet been created.
   */
  @Nullable
  ProcessableGraph get(Version version);

  /**
   * Creates a new root version with the specified root.
   */
  ProcessableGraph createRoot(Version rootVersion);

  /**
   * Creates a child of the specified base version.
   */
  ProcessableGraph createChild(Version baseVersion, Version newVersion);
}
