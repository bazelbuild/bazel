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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException3;
import java.io.IOException;
import java.util.List;
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
final class ArtifactNestedSetFunction implements SkyFunction {

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
   *
   * <p>We can't make this a:
   *
   * <p>- Weak-keyd map since ActionExecutionValue holds a reference to Artifact.
   *
   * <p>- Weak-valued map since there's nothing else holding on to ValueOrException and the entry
   * will GCed immediately. // TODO(leba): Re-evaluate the above point about weak-valued map.
   *
   * <p>This map will be removed when --experimental_nsos_eval_keys_as_one_group is stable, and
   * replaced by {@link #artifactSkyKeyToSkyValue}.
   */
  private ConcurrentMap<
          SkyKey,
          ValueOrException3<IOException, ActionExecutionException, ArtifactNestedSetEvalException>>
      artifactSkyKeyToValueOrException;

  /**
   * A concurrent map from Artifacts' SkyKeys to their SkyValue, for Artifacts that are part of
   * NestedSets which were evaluated as {@link ArtifactNestedSetKey}. This is expected to replace
   * the above {@link #artifactSkyKeyToValueOrException}.
   */
  private ConcurrentMap<SkyKey, SkyValue> artifactSkyKeyToSkyValue;

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

  private static Boolean evalKeysAsOneGroup = null;

  private ArtifactNestedSetFunction() {
    artifactSkyKeyToSkyValue = Maps.newConcurrentMap();
    artifactSkyKeyToValueOrException = Maps.newConcurrentMap();
    nestedSetToSkyKey = new MapMaker().weakKeys().makeMap();
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, ArtifactNestedSetFunctionException {

    if (evalKeysAsOneGroup) {
      return evalDepsInOneGroup(skyKey, env);
    }

    ArtifactNestedSetKey artifactNestedSetKey = (ArtifactNestedSetKey) skyKey;
    Map<
            SkyKey,
            ValueOrException3<
                IOException, ActionExecutionException, ArtifactNestedSetEvalException>>
        directArtifactsEvalResult =
            env.getValuesOrThrow(
                artifactNestedSetKey.directKeys(),
                IOException.class,
                ActionExecutionException.class,
                ArtifactNestedSetEvalException.class);

    // Evaluate all children.
    List<Object> transitiveMembers = artifactNestedSetKey.transitiveMembers();
    List<SkyKey> transitiveKeys = Lists.newArrayListWithCapacity(transitiveMembers.size());
    for (Object transitiveMember : transitiveMembers) {
      transitiveKeys.add(
          nestedSetToSkyKey.computeIfAbsent(transitiveMember, ArtifactNestedSetKey::new));
    }
    env.getValues(transitiveKeys);

    if (env.valuesMissing()) {
      return null;
    }

    // Only commit to the map when every value is present.
    artifactSkyKeyToValueOrException.putAll(directArtifactsEvalResult);
    return new ArtifactNestedSetValue();
  }

  /** The main path with --experimental_nsos_eval_keys_as_one_group. */
  private SkyValue evalDepsInOneGroup(SkyKey skyKey, Environment env)
      throws InterruptedException, ArtifactNestedSetFunctionException {
    ArtifactNestedSetKey artifactNestedSetKey = (ArtifactNestedSetKey) skyKey;
    Iterable<SkyKey> directKeys = artifactNestedSetKey.directKeys();
    List<Object> transitiveMembers = artifactNestedSetKey.transitiveMembers();
    List<SkyKey> transitiveKeys = Lists.newArrayListWithCapacity(transitiveMembers.size());
    for (Object transitiveMember : transitiveMembers) {
      transitiveKeys.add(
          nestedSetToSkyKey.computeIfAbsent(transitiveMember, ArtifactNestedSetKey::new));
    }
    Map<
            SkyKey,
            ValueOrException3<
                IOException, ActionExecutionException, ArtifactNestedSetEvalException>>
        depsEvalResult =
            env.getValuesOrThrow(
                Iterables.concat(directKeys, transitiveKeys),
                IOException.class,
                ActionExecutionException.class,
                ArtifactNestedSetEvalException.class);

    NestedSetBuilder<Pair<SkyKey, Exception>> transitiveExceptionsBuilder =
        NestedSetBuilder.stableOrder();

    // Throw a SkyFunctionException when a dep evaluation results in an exception.
    // Only non-null values should be committed to
    // ArtifactNestedSetFunction#artifacSkyKeyToSkyValue.
    for (Map.Entry<
            SkyKey,
            ValueOrException3<
                IOException, ActionExecutionException, ArtifactNestedSetEvalException>>
        entry : depsEvalResult.entrySet()) {
      try {
        // Trigger the exception, if any.
        SkyValue value = entry.getValue().get();
        if (entry.getKey() instanceof ArtifactNestedSetKey || value == null) {
          continue;
        }
        artifactSkyKeyToSkyValue.put(entry.getKey(), value);
      } catch (IOException | ActionExecutionException e) {
        transitiveExceptionsBuilder.add(Pair.of(entry.getKey(), e));
      } catch (ArtifactNestedSetEvalException e) {
        transitiveExceptionsBuilder.addTransitive(e.getNestedExceptions());
      }
    }

    if (!transitiveExceptionsBuilder.isEmpty()) {
      NestedSet<Pair<SkyKey, Exception>> transitiveExceptions = transitiveExceptionsBuilder.build();
      throw new ArtifactNestedSetFunctionException(
          new ArtifactNestedSetEvalException(
              transitiveExceptions.memoizedFlattenAndGetSize()
                  + " error(s) encountered while evaluating NestedSet.",
              transitiveExceptions),
          skyKey);
    }

    // This should only happen when all error handling is done.
    if (env.valuesMissing()) {
      return null;
    }
    return new ArtifactNestedSetValue();
  }

  static ArtifactNestedSetFunction getInstance() {
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
  static ArtifactNestedSetFunction createInstance() {
    singleton = new ArtifactNestedSetFunction();
    return singleton;
  }

  /** Reset the various state-keeping maps of ArtifactNestedSetFunction. */
  void resetArtifactNestedSetFunctionMaps() {
    artifactSkyKeyToValueOrException = Maps.newConcurrentMap();
    artifactSkyKeyToSkyValue = Maps.newConcurrentMap();
  }

  Map<SkyKey, SkyValue> getArtifactSkyKeyToSkyValue() {
    return artifactSkyKeyToSkyValue;
  }

  Map<
          SkyKey,
          ValueOrException3<IOException, ActionExecutionException, ArtifactNestedSetEvalException>>
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
  static int getSizeThreshold() {
    return sizeThreshold == null ? 0 : sizeThreshold;
  }

  static boolean evalKeysAsOneGroup() {
    return evalKeysAsOneGroup;
  }

  /**
   * Updates the evalKeysAsOneGroup attr if the existing value differs from the new one.
   *
   * @return whether an update was made.
   */
  static boolean evalKeysAsOneGroupUpdated(boolean newEvalKeysAsOneGroup) {
    // If this is the first time the value is set, it's not considered "updated".
    if (evalKeysAsOneGroup == null) {
      evalKeysAsOneGroup = newEvalKeysAsOneGroup;
      return false;
    }

    if (evalKeysAsOneGroup != newEvalKeysAsOneGroup) {
      evalKeysAsOneGroup = newEvalKeysAsOneGroup;
      return true;
    }

    return false;
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

    ArtifactNestedSetFunctionException(ArtifactNestedSetEvalException e, SkyKey child) {
      super(e, child);
    }

    @Override
    public boolean isCatastrophic() {
      return false;
    }
  }

  /** Bundles the exceptions from the evaluation of the children keys together. */
  static final class ArtifactNestedSetEvalException extends Exception {

    private final NestedSet<Pair<SkyKey, Exception>> nestedExceptions;

    ArtifactNestedSetEvalException(
        String message, NestedSet<Pair<SkyKey, Exception>> nestedExceptions) {
      super(message);
      this.nestedExceptions = nestedExceptions;
    }

    NestedSet<Pair<SkyKey, Exception>> getNestedExceptions() {
      return nestedExceptions;
    }
  }
}
