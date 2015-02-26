// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;

import javax.annotation.Nullable;

/**
 * {@link WalkableGraph} that looks nodes up in a {@link QueryableGraph}.
 */
public class DelegatingWalkableGraph implements WalkableGraph {
  private final QueryableGraph graph;

  public DelegatingWalkableGraph(QueryableGraph graph) {
    this.graph = graph;
  }

  private NodeEntry getEntry(SkyKey key) {
    NodeEntry entry = Preconditions.checkNotNull(graph.get(key), key);
    Preconditions.checkState(entry.isDone(), "%s %s", key, entry);
    return entry;
  }

  @Override
  public boolean exists(SkyKey key) {
    NodeEntry entry = graph.get(key);
    return entry != null && entry.isDone();
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey key) {
    return getEntry(key).getValue();
  }

  @Nullable
  @Override
  public Exception getException(SkyKey key) {
    ErrorInfo errorInfo = getEntry(key).getErrorInfo();
    return errorInfo == null ? null : errorInfo.getException();
  }

  @Override
  public Iterable<SkyKey> getDirectDeps(SkyKey key) {
    return getEntry(key).getDirectDeps();
  }

  @Override
  public Iterable<SkyKey> getReverseDeps(SkyKey key) {
    return getEntry(key).getReverseDeps();
  }
}
