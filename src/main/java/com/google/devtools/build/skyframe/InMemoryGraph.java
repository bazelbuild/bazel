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

import com.google.common.base.Predicates;
import com.google.common.collect.Maps;
import java.util.Collections;
import java.util.Map;
import javax.annotation.Nullable;

/** {@link ProcessableGraph} that exposes the contents of the entire graph. */
public interface InMemoryGraph extends ProcessableGraph {
  @Override
  Map<SkyKey, ? extends NodeEntry> createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys);

  @Nullable
  @Override
  NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey key);

  @Override
  Map<SkyKey, ? extends NodeEntry> getBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys);

  /**
   * Returns a read-only live view of the nodes in the graph. All node are included. Dirty values
   * include their Node value. Values in error have a null value.
   */
  Map<SkyKey, SkyValue> getValues();

  /**
   * Returns a read-only live view of the done values in the graph. Dirty, changed, and error values
   * are not present in the returned map
   */
  default Map<SkyKey, SkyValue> getDoneValues() {
    return transformDoneEntries(getAllValuesMutable());
  }

  // Only for use by MemoizingEvaluator#delete
  Map<SkyKey, ? extends NodeEntry> getAllValues();

  Map<SkyKey, ? extends NodeEntry> getAllValuesMutable();

  static Map<SkyKey, SkyValue> transformDoneEntries(Map<SkyKey, ? extends NodeEntry> nodeMap) {
    return Collections.unmodifiableMap(
        Maps.filterValues(
            Maps.transformValues(
                nodeMap,
                entry -> {
                  if (!entry.isDone()) {
                    return null;
                  }
                  try {
                    return entry.getValue();
                  } catch (InterruptedException e) {
                    throw new IllegalStateException("Interrupted getting " + entry, e);
                  }
                }),
            Predicates.notNull()));
  }
}
