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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A simple Differencer which just records the invalidated nodes it's been given.
 */
@ThreadSafety.ThreadCompatible
public class RecordingDifferencer implements Differencer {

  private List<NodeKey> nodesToInvalidate;
  private Map<NodeKey, Node> nodesToInject;

  public RecordingDifferencer() {
    clear();
  }

  private void clear() {
    nodesToInvalidate = new ArrayList<>();
    nodesToInject = new HashMap<>();
  }

  @Override
  public Diff getDiff(Version fromVersion, Version toVersion) {
    Diff diff = new ImmutableDiff(nodesToInvalidate, nodesToInject);
    clear();
    return diff;
  }

  /**
   * Store the given nodes for invalidation.
   */
  public void invalidate(Iterable<NodeKey> nodes) {
    Iterables.addAll(nodesToInvalidate, nodes);
  }

  /**
   * Invalidates the cached values of any nodes in error.
   *
   * <p>If a future call to {@link AutoUpdatingGraph#update} requests a node that transitively
   * depends on any node that was in an error state (or is one of these), they will be re-computed.
   */
  public void invalidateErrors() {
    // All error nodes have a dependency on the single global ERROR_TRANSIENCE node,
    // so we only have to invalidate that one node to catch everything.
    invalidate(ImmutableList.of(ErrorTransienceNode.key()));
  }

  /**
   * Store the given nodes for injection.
   */
  public void inject(Map<NodeKey, ? extends Node> nodes) {
    nodesToInject.putAll(nodes);
  }
}
