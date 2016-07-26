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

/**
 * {@link DeterministicHelper.DeterministicProcessableGraph} that implements the {@link
 * InMemoryGraph} interface. Sadly, cannot be a {@link NotifyingInMemoryGraph} due to Java's
 * forbidding multiple inheritance.
 */
class DeterministicInMemoryGraph extends DeterministicHelper.DeterministicProcessableGraph
    implements InMemoryGraph {
  DeterministicInMemoryGraph(InMemoryGraph delegate, NotifyingHelper.Listener graphListener) {
    super(delegate, graphListener);
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatchForInvalidation(Iterable<SkyKey> keys) {
    return getBatchWithFieldHints(null, Reason.INVALIDATION, keys, NodeEntryField.ALL_FIELDS);
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    return ((InMemoryGraph) delegate).getValues();
  }

  @Override
  public Map<SkyKey, SkyValue> getDoneValues() {
    return ((InMemoryGraph) delegate).getDoneValues();
  }

  @Override
  public Map<SkyKey, NodeEntry> getAllValues() {
    return ((InMemoryGraph) delegate).getAllValues();
  }
}
