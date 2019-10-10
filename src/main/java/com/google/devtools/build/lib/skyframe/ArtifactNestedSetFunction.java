// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * A builder of values for {@link ArtifactNestedSetKey}.
 *
 * <p>When an Action is executed with ActionExecutionFunction, the actions's input {@code
 * NestedSet<Artifact>} could be evaluated as an {@link ArtifactNestedSetKey}[1].
 *
 * <p>{@link ArtifactNestedSetFunction} then evaluates the {@link ArtifactNestedSetKey} by:
 *
 * <p>- Evaluating the directs elements as Artifacts. Commit the result into artifactToSkyValueMap.
 *
 * <p>- Evaluating the transitive elements as {@link ArtifactNestedSetKey}s.
 *
 * <p>ActionExecutionFunction can then access this map to get the Artifacts' values.
 *
 * <p>[1] Heuristic: If the size of the NestedSet exceeds a certain threshold, we evaluate it as an
 * ArtifactNestedSetKey.
 */
class ArtifactNestedSetFunction implements SkyFunction {

  /**
   * A concurrent map from Artifacts' SkyKeys to their ValueOrException, for Artifacts that are part
   * of NestedSets which were evaluated as {@link ArtifactNestedSetKey}.
   *
   * <p>Question: Why don't we clear artifactToSkyValueMap after each build?
   *
   * <p>The map maintains an invariant: if an ArtifactNestedSetKey exists on Skyframe, the SkyValues
   * of its member Artifacts are available in artifactToSkyValueMap.
   *
   * <p>Example: Action A has as input NestedSet X, where X = (X1, X2), where X1 & X2 are 2
   * transitive NestedSets.
   *
   * <p>Run 0: Establish dependency from A to X and from X to X1 & X2. Artifacts from X1 & X2 have
   * entries in artifactToSkyValueMap.
   *
   * <p>Run 1 (incremental): Some changes were made to an Artifact in X1 such that X1, X and A's
   * SkyKeys are marked as dirty. A's ActionLookupData has to be re-evaluated. This involves asking
   * Skyframe to compute SkyValues for its inputs.
   *
   * <p>However, X2 is not dirty, so Skyframe won't re-run ArtifactNestedSetFunction#compute for X2,
   * therefore not populating artifactToSkyValueMap with X2's member Artifacts. Hence if we clear
   * artifactToSkyValueMap between build 0 and 1, X2's member artifacts' SkyValues would not be
   * available in the map.
   *
   * <p>Keeping the map in between builds introduces a potential memory leak: if an Artifact is no
   * longer valid, it would still exist in the map. TODO(leba): Address this memory leak if it
   * causes serious regression.
   */
  private final ConcurrentMap<SkyKey, ValueOrException2<IOException, ActionExecutionException>>
      artifactToSkyValueMap;

  private static ArtifactNestedSetFunction singleton = null;

  private ArtifactNestedSetFunction() {
    artifactToSkyValueMap = new ConcurrentHashMap<>();
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    ArtifactNestedSetKey artifactNestedSetKey = (ArtifactNestedSetKey) skyKey;
    Map<SkyKey, ValueOrException2<IOException, ActionExecutionException>>
        directArtifactsEvalResult =
            env.getValuesOrThrow(
                artifactNestedSetKey.directKeys(),
                IOException.class,
                ActionExecutionException.class);
    if (env.valuesMissing()) {
      return null;
    }

    // Evaluate all children.
    env.getValues(artifactNestedSetKey.transitiveKeys());

    if (env.valuesMissing()) {
      return null;
    }

    // Only commit to the map when every value is present.
    artifactToSkyValueMap.putAll(directArtifactsEvalResult);
    return ArtifactNestedSetValue.createOrGetInstance();
  }

  public static ArtifactNestedSetFunction getInstance() {
    if (singleton == null) {
      singleton = new ArtifactNestedSetFunction();
    }
    return singleton;
  }

  Map<SkyKey, ValueOrException2<IOException, ActionExecutionException>> getArtifactToSkyValueMap() {
    return artifactToSkyValueMap;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
