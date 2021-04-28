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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.SourceArtifactException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException3;
import java.util.ArrayList;
import java.util.List;
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
 * artifactSkyKeyToSkyValue.
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
   * A concurrent map from Artifacts' SkyKeys to their SkyValue, for Artifacts that are part of
   * NestedSets which were evaluated as {@link ArtifactNestedSetKey}.
   *
   * <p>Question: Why don't we clear artifactSkyKeyToSkyValue after each build?
   *
   * <p>The map maintains an invariant: if an ArtifactNestedSetKey exists on Skyframe, the SkyValues
   * of its member Artifacts are available in artifactSkyKeyToSkyValue.
   *
   * <p>Example: Action A has as input NestedSet X, where X = (X1, X2), where X1 & X2 are 2
   * transitive NestedSets.
   *
   * <p>Run 0: Establish dependency from A to X and from X to X1 & X2. Artifacts from X1 & X2 have
   * entries in artifactSkyKeyToSkyValue.
   *
   * <p>Run 1 (incremental): Some changes were made to an Artifact in X1 such that X1, X and A's
   * SkyKeys are marked as dirty. A's ActionLookupData has to be re-evaluated. This involves asking
   * Skyframe to compute SkyValues for its inputs.
   *
   * <p>However, X2 is not dirty, so Skyframe won't re-run ArtifactNestedSetFunction#compute for X2,
   * therefore not populating artifactSkyKeyToSkyValue with X2's member Artifacts. Hence if we clear
   * artifactSkyKeyToSkyValue between build 0 and 1, X2's member artifacts' SkyValues would not be
   * available in the map. TODO(leba): Make this weak-keyed.
   */
  private ConcurrentMap<SkyKey, SkyValue> artifactSkyKeyToSkyValue;

  /**
   * Maps the NestedSets' underlying objects to the corresponding SkyKey. This is to avoid
   * re-creating SkyKey for the same nested set upon reevaluation because of e.g. a missing value.
   *
   * <p>The map weakly references its values: when the ArtifactNestedSetKey becomes otherwise
   * unreachable, the map entry is collected.
   */
  private final ConcurrentMap<NestedSet.Node, ArtifactNestedSetKey>
      nestedSetToSkyKey; // note: weak values!

  private static ArtifactNestedSetFunction singleton = null;

  private static Integer sizeThreshold = null;

  private ArtifactNestedSetFunction() {
    artifactSkyKeyToSkyValue = Maps.newConcurrentMap();
    nestedSetToSkyKey = new MapMaker().weakValues().makeMap();
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, ArtifactNestedSetFunctionException {
    List<SkyKey> depKeys = getDepSkyKeys((ArtifactNestedSetKey) skyKey);
    List<
            ValueOrException3<
                SourceArtifactException, ActionExecutionException, ArtifactNestedSetEvalException>>
        depsEvalResult =
            env.getOrderedValuesOrThrow(
                depKeys,
                SourceArtifactException.class,
                ActionExecutionException.class,
                ArtifactNestedSetEvalException.class);

    NestedSetBuilder<Pair<SkyKey, Exception>> transitiveExceptionsBuilder =
        NestedSetBuilder.stableOrder();
    boolean catastrophic = false;

    // Throw a SkyFunctionException when a dep evaluation results in an exception.
    // Only non-null values should be committed to
    // ArtifactNestedSetFunction#artifacSkyKeyToSkyValue.
    int i = 0;
    for (ValueOrException3<
            SourceArtifactException, ActionExecutionException, ArtifactNestedSetEvalException>
        valueOrException : depsEvalResult) {
      SkyKey key = depKeys.get(i++);
      try {
        // Trigger the exception, if any.
        SkyValue value = valueOrException.get();
        if (key instanceof ArtifactNestedSetKey || value == null) {
          continue;
        }
        artifactSkyKeyToSkyValue.put(key, value);
      } catch (SourceArtifactException e) {
        // SourceArtifactException is never catastrophic.
        transitiveExceptionsBuilder.add(Pair.of(key, e));
      } catch (ActionExecutionException e) {
        transitiveExceptionsBuilder.add(Pair.of(key, e));
        catastrophic |= e.isCatastrophe();
      } catch (ArtifactNestedSetEvalException e) {
        catastrophic |= e.isCatastrophic();
        transitiveExceptionsBuilder.addTransitive(e.getNestedExceptions());
      }
    }

    if (!transitiveExceptionsBuilder.isEmpty()) {
      NestedSet<Pair<SkyKey, Exception>> transitiveExceptions = transitiveExceptionsBuilder.build();
      // The NestedSet of exceptions is usually small, hence flattening won't be too costly.
      Pair<SkyKey, Exception> firstSkyKeyAndException = transitiveExceptions.toList().get(0);
      throw new ArtifactNestedSetFunctionException(
          new ArtifactNestedSetEvalException(
              "Error evaluating artifact nested set. First exception: "
                  + firstSkyKeyAndException.getSecond()
                  + ", SkyKey: "
                  + firstSkyKeyAndException.getFirst(),
              transitiveExceptions,
              catastrophic));
    }

    // This should only happen when all error handling is done.
    if (env.valuesMissing()) {
      return null;
    }
    return new ArtifactNestedSetValue();
  }

  private List<SkyKey> getDepSkyKeys(ArtifactNestedSetKey skyKey) {
    NestedSet<Artifact> set = skyKey.getSet();
    List<SkyKey> keys = new ArrayList<>();
    for (Artifact file : set.getLeaves()) {
      keys.add(Artifact.key(file));
    }
    for (NestedSet<Artifact> nonLeaf : set.getNonLeaves()) {
      keys.add(
          nestedSetToSkyKey.computeIfAbsent(
              nonLeaf.toNode(), (node) -> new ArtifactNestedSetKey(nonLeaf, node)));
    }
    return keys;
  }

  static ArtifactNestedSetFunction getInstance() {
    checkNotNull(singleton);
    return singleton;
  }

  /**
   * Creates a new instance. Should only be used in {@code SkyframeExecutor#skyFunctions}. Keeping
   * this method separated from {@code #getInstance} since sometimes we need to overwrite the
   * existing instance.
   */
  static ArtifactNestedSetFunction createInstance() {
    singleton = new ArtifactNestedSetFunction();
    return singleton;
  }

  /** Reset the various state-keeping maps of ArtifactNestedSetFunction. */
  void resetArtifactNestedSetFunctionMaps() {
    artifactSkyKeyToSkyValue = Maps.newConcurrentMap();
  }

  SkyValue getValueForKey(SkyKey skyKey) {
    return artifactSkyKeyToSkyValue.get(skyKey);
  }

  void updateValueForKey(SkyKey skyKey, SkyValue skyValue) {
    artifactSkyKeyToSkyValue.put(skyKey, skyValue);
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Get the threshold to which we evaluate a NestedSet as a Skykey. If sizeThreshold is unset,
   * return the default value of 0.
   */
  static int getSizeThreshold() {
    return sizeThreshold == null ? 0 : sizeThreshold;
  }

  /**
   * Updates the sizeThreshold value if the existing value differs from newValue.
   *
   * @param newValue The new value from --experimental_nested_set_as_skykey_threshold.
   * @return whether an update was made.
   */
  static boolean sizeThresholdUpdated(int newValue) {
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

  /** Mainly used for error bubbling when evaluating direct/transitive children. */
  private static final class ArtifactNestedSetFunctionException extends SkyFunctionException {

    private final boolean catastrophic;

    ArtifactNestedSetFunctionException(ArtifactNestedSetEvalException e) {
      super(e, Transience.PERSISTENT);
      this.catastrophic = e.isCatastrophic();
    }

    @Override
    public boolean isCatastrophic() {
      return catastrophic;
    }
  }

  /** Bundles the exceptions from the evaluation of the children keys together. */
  static final class ArtifactNestedSetEvalException extends Exception {

    private final NestedSet<Pair<SkyKey, Exception>> nestedExceptions;
    private final boolean catastrophic;

    ArtifactNestedSetEvalException(
        String message, NestedSet<Pair<SkyKey, Exception>> nestedExceptions, boolean catastrophic) {
      super(message);
      this.nestedExceptions = nestedExceptions;
      this.catastrophic = catastrophic;
    }

    NestedSet<Pair<SkyKey, Exception>> getNestedExceptions() {
      return nestedExceptions;
    }

    // Should be true if at least one child exception is catastrophic.
    boolean isCatastrophic() {
      return catastrophic;
    }
  }
}
