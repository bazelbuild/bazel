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

import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;

/**
 * A builder of values for {@link ArtifactNestedSetKey}.
 *
 * <p>When an Action is executed with ActionExecutionFunction, the actions's input {@code
 * NestedSet<Artifact>} could be evaluated as an {@link ArtifactNestedSetKey}[1].
 *
 * <p>{@link ArtifactNestedSetFunction} then evaluates the {@link ArtifactNestedSetKey} by:
 *
 * <p>- Evaluating the directs elements as Artifacts. Commit the result into
 * artifactSkyKeyToValueOrException.
 *
 * <p>- Evaluating the transitive elements as {@link ArtifactNestedSetKey}s.
 *
 * <p>ActionExecutionFunction can then access this map to get the Artifacts' values.
 *
 * <p>[1] Heuristic: If the size of the NestedSet exceeds a certain threshold, we evaluate it as an
 * ArtifactNestedSetKey.
 */
public class ArtifactNestedSetFunction implements SkyFunction {

  /**
   * A concurrent map from Artifacts' SkyKeys to their ValueOrException, for Artifacts that are part
   * of NestedSets which were evaluated as {@link ArtifactNestedSetKey}.
   *
   * <p>Question: Why don't we clear artifactSkyKeyToValueOrException after each build?
   *
   * <p>The map maintains an invariant: if an ArtifactNestedSetKey exists on Skyframe, the SkyValues
   * of its member Artifacts are available in artifactSkyKeyToValueOrException.
   *
   * <p>Example: Action A has as input NestedSet X, where X = (X1, X2), where X1 & X2 are 2
   * transitive NestedSets.
   *
   * <p>Run 0: Establish dependency from A to X and from X to X1 & X2. Artifacts from X1 & X2 have
   * entries in artifactSkyKeyToValueOrException.
   *
   * <p>Run 1 (incremental): Some changes were made to an Artifact in X1 such that X1, X and A's
   * SkyKeys are marked as dirty. A's ActionLookupData has to be re-evaluated. This involves asking
   * Skyframe to compute SkyValues for its inputs.
   *
   * <p>However, X2 is not dirty, so Skyframe won't re-run ArtifactNestedSetFunction#compute for X2,
   * therefore not populating artifactSkyKeyToValueOrException with X2's member Artifacts. Hence if
   * we clear artifactSkyKeyToValueOrException between build 0 and 1, X2's member artifacts'
   * SkyValues would not be available in the map.
   */
  private final ConcurrentMap<SkyKey, ValueOrException2<IOException, ActionExecutionException>>
      artifactSkyKeyToValueOrException;

  /**
   * Maps the NestedSets' underlying objects to the corresponding SkyKey. This is to avoid
   * re-creating SkyKey for the same nested set upon reevaluation because of e.g. a missing value.
   *
   * <p>The map has weak references to keys to prevent memory leaks: if a nested set no longer
   * exists, its entry would be automatically removed from the map by the GC.
   */
  private final ConcurrentMap<Object, SkyKey> nestedSetToSkyKey;

  private static ArtifactNestedSetFunction singleton = null;

  private static Integer sizeThreshold = null;

  private ArtifactNestedSetFunction() {
    artifactSkyKeyToValueOrException = Maps.newConcurrentMap();
    nestedSetToSkyKey = new MapMaker().weakKeys().makeMap();
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

    // Evaluate all children.
    ArrayList<SkyKey> transitiveKeys = new ArrayList<>();
    for (Object transitive : artifactNestedSetKey.transitiveMembers()) {
      nestedSetToSkyKey.putIfAbsent(transitive, new ArtifactNestedSetKey(transitive));
      transitiveKeys.add(nestedSetToSkyKey.get(transitive));
    }
    env.getValues(transitiveKeys);

    if (env.valuesMissing()) {
      return null;
    }

    // Only commit to the map when every value is present.
    artifactSkyKeyToValueOrException.putAll(directArtifactsEvalResult);
    return new ArtifactNestedSetValue();
  }

  public static ArtifactNestedSetFunction getInstance() {
    if (singleton == null) {
      return createInstance();
    }
    return singleton;
  }

  /**
   * Creates a new instance. Should only be used in {@code SkyframeExecutor#skyFunctions}. Keeping
   * this method separated from {@code #getInstance} since sometimes we need to overwrite the
   * existing instance.
   */
  public static ArtifactNestedSetFunction createInstance() {
    singleton = new ArtifactNestedSetFunction();
    return singleton;
  }

  Map<SkyKey, ValueOrException2<IOException, ActionExecutionException>>
      getArtifactSkyKeyToValueOrException() {
    return artifactSkyKeyToValueOrException;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Get the threshold to which we evaluate a NestedSet as a Skykey. If sizeThreshold is unset,
   * return the default value of 0.
   */
  public static int getSizeThreshold() {
    return sizeThreshold == null ? 0 : sizeThreshold;
  }

  /**
   * Updates the sizeThreshold value if the existing value differs from newValue.
   *
   * @param newValue The new value from --experimental_nested_set_as_skykey_threshold.
   * @return whether an update was made.
   */
  public static boolean sizeThresholdUpdatedTo(int newValue) {
    // If this is the first time the value is set, it's not considered "updated".
    if (sizeThreshold == null) {
      sizeThreshold = newValue;
      return false;
    }

    if (sizeThreshold == newValue || (sizeThreshold <= 0 && newValue <= 0)) {
      return false;
    }
    sizeThreshold = newValue;
    return true;
  }
}
