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
import javax.annotation.Nullable;

/** {@link NotifyingHelper} that additionally implements the {@link InMemoryGraph} interface. */
class NotifyingInMemoryGraph extends NotifyingHelper.NotifyingProcessableGraph
    implements InMemoryGraph {
  NotifyingInMemoryGraph(InMemoryGraph delegate, NotifyingHelper.Listener graphListener) {
    super(delegate, graphListener);
  }

  @Override
  public Map<SkyKey, ? extends NodeEntry> createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys) {
    try {
      return super.createIfAbsentBatch(requestor, reason, keys);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  @Nullable
  @Override
  public NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey key) {
    try {
      return super.get(requestor, reason, key);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public Map<SkyKey, ? extends NodeEntry> getBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    try {
      return super.getBatch(requestor, reason, keys);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    return ((InMemoryGraph) delegate).getValues();
  }

  @Override
  public Map<SkyKey, ? extends NodeEntry> getAllValues() {
    return ((InMemoryGraph) delegate).getAllValues();
  }

  @Override
  public Map<SkyKey, ? extends NodeEntry> getAllValuesMutable() {
    return ((InMemoryGraph) delegate).getAllValuesMutable();
  }
}
