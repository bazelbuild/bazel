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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.skyframe.WalkableGraph;

import java.util.Collection;

/**
 *  Encapsulates the raw analysis result of top level targets and aspects coming from Skyframe.
 */
public class SkyframeAnalysisResult {
  private final ImmutableList<ConfiguredTarget> configuredTargets;
  private final WalkableGraph walkableGraph;
  private final ImmutableList<AspectValue> aspects;

  public SkyframeAnalysisResult(
      ImmutableList<ConfiguredTarget> configuredTargets,
      WalkableGraph walkableGraph,
      ImmutableList<AspectValue> aspects) {
    this.configuredTargets = configuredTargets;
    this.walkableGraph = walkableGraph;
    this.aspects = aspects;
  }

  public Collection<ConfiguredTarget> getConfiguredTargets() {
    return configuredTargets;
  }

  public WalkableGraph getWalkableGraph() {
    return walkableGraph;
  }

  public Collection<AspectValue> getAspects() {
    return aspects;
  }
}
