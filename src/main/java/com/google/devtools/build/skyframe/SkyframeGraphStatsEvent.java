// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * Postable transporting data about the size/shape of the Skyframe graph. Note that the graph may
 * depend on the sequence of Bazel invocations prior to this one, not just the current one.
 */
public final class SkyframeGraphStatsEvent implements ExtendedEventHandler.Postable {
  /** Data about the Skyframe evaluations that happened during this command. */
  public record EvaluationStats(
      ImmutableMap<SkyFunctionName, Integer> dirtied,
      ImmutableMap<SkyFunctionName, Integer> changed,
      ImmutableMap<SkyFunctionName, Integer> built,
      ImmutableMap<SkyFunctionName, Integer> cleaned) {}

  private final int graphSize;
  private final EvaluationStats evaluationStats;

  SkyframeGraphStatsEvent(int graphSize, EvaluationStats evaluationStats) {
    this.graphSize = graphSize;
    this.evaluationStats = evaluationStats;
  }

  public int getGraphSize() {
    return graphSize;
  }

  public EvaluationStats getEvaluationStats() {
    return evaluationStats;
  }
}
