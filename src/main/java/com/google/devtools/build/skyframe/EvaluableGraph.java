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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * Interface between a single version of the graph and the evaluator. Supports mutation of that
 * single version of the graph.
 */
@ThreadSafe
interface EvaluableGraph extends QueryableGraph {
  /**
   * Creates a new node with the specified key if it does not exist yet. Returns the node entry
   * (either the existing one or the one just created), never {@code null}.
   */
  NodeEntry createIfAbsent(SkyKey key);
}
