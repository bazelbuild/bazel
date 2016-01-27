// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.WalkableGraph;

import java.util.Collection;

/**
 *  Encapsulates the raw analysis result of top level targets and aspects coming from Skyframe.
 */
public class SkyframeAnalysisResult {
  private final boolean hasError;
  private final ImmutableList<ConfiguredTarget> configuredTargets;
  private final WalkableGraph walkableGraph;
  private final ImmutableList<AspectValue> aspects;
  private final ImmutableMap<PackageIdentifier, Path> packageRoots;

  public SkyframeAnalysisResult(
      boolean hasError,
      ImmutableList<ConfiguredTarget> configuredTargets,
      WalkableGraph walkableGraph,
      ImmutableList<AspectValue> aspects,
      ImmutableMap<PackageIdentifier, Path> packageRoots) {
    this.hasError = hasError;
    this.configuredTargets = configuredTargets;
    this.walkableGraph = walkableGraph;
    this.aspects = aspects;
    this.packageRoots = packageRoots;
  }

  public boolean hasError() {
    return hasError;
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

  public ImmutableMap<PackageIdentifier, Path> getPackageRoots() {
    return packageRoots;
  }
}
