// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.skyframe.DelegatingWalkableGraph;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.NodeEntry;
import com.google.devtools.build.skyframe.QueryableGraph;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * {@link com.google.devtools.build.skyframe.WalkableGraph} backed by a {@link SkyframeExecutor}.
 */
public class SkyframeExecutorWrappingWalkableGraph extends DelegatingWalkableGraph {

  private SkyframeExecutorWrappingWalkableGraph(MemoizingEvaluator evaluator) {
    super(
        new QueryableGraph() {
          @Nullable
          @Override
          public NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey key)
              throws InterruptedException {
            return evaluator.getExistingEntryAtLatestVersion(key);
          }

          @Override
          public Map<SkyKey, ? extends NodeEntry> getBatch(
              @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
              throws InterruptedException {
            Map<SkyKey, NodeEntry> result = new HashMap<>();
            for (SkyKey key : keys) {
              NodeEntry nodeEntry = get(requestor, reason, key);
              if (nodeEntry != null) {
                result.put(key, nodeEntry);
              }
            }
            return result;
          }

        });
  }

  public static SkyframeExecutorWrappingWalkableGraph of(SkyframeExecutor skyframeExecutor) {
    // TODO(janakr): Provide the graph in a more principled way.
    return new SkyframeExecutorWrappingWalkableGraph(skyframeExecutor.getEvaluatorForTesting());
  }
}
