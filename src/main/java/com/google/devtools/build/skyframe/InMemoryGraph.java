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
package com.google.devtools.build.skyframe;

import java.util.Map;

/** {@link ProcessableGraph} that exposes the contents of the entire graph. */
interface InMemoryGraph extends ProcessableGraph, InvalidatableGraph {
  @Override
  Map<SkyKey, NodeEntry> getBatch(Iterable<SkyKey> keys);

  /**
   * Returns a read-only live view of the nodes in the graph. All node are included. Dirty values
   * include their Node value. Values in error have a null value.
   */
  Map<SkyKey, SkyValue> getValues();

  /**
   * Returns a read-only live view of the done values in the graph. Dirty, changed, and error values
   * are not present in the returned map
   */
  Map<SkyKey, SkyValue> getDoneValues();

  // Only for use by MemoizingEvaluator#delete
  Map<SkyKey, NodeEntry> getAllValues();
}
