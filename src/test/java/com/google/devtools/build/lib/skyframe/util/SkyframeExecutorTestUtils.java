// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Helper functions for manually dealing with a {@link SkyframeExecutor}'s graph in tests.
 */
public class SkyframeExecutorTestUtils {

  private SkyframeExecutorTestUtils() {
  }

  /** Returns an existing value, or {@code null} if the given key is not currently in the graph. */
  @Nullable
  public static SkyValue getExistingValue(SkyframeExecutor skyframeExecutor, SkyKey key)
      throws InterruptedException {
    return skyframeExecutor.getEvaluator().getExistingValue(key);
  }

  /**
   * Returns an existing error info, or {@code null} if the given key is not currently in the graph.
   */
  @Nullable
  public static ErrorInfo getExistingError(SkyframeExecutor skyframeExecutor, SkyKey key)
      throws InterruptedException {
    return skyframeExecutor.getEvaluator().getExistingErrorForTesting(key);
  }

  /** Calls {@link MemoizingEvaluator#evaluate} on the given {@link SkyframeExecutor}'s graph. */
  public static <T extends SkyValue> EvaluationResult<T> evaluate(
      SkyframeExecutor skyframeExecutor,
      SkyKey key,
      boolean keepGoing,
      ExtendedEventHandler errorEventListener)
      throws InterruptedException {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setParallelism(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(errorEventListener)
            .build();
    return skyframeExecutor.getEvaluator().evaluate(ImmutableList.of(key), evaluationContext);
  }

  /**
   * Returns an existing configured target value, or {@code null} if there is not an appropriate
   * configured target value key in the graph.
   *
   * <p>This helper is provided so legacy tests don't need to know about details of skyframe keys.
   */
  @Nullable
  public static ConfiguredTargetValue getExistingConfiguredTargetValue(
      SkyframeExecutor skyframeExecutor, Label label, BuildConfigurationValue config)
      throws InterruptedException {
    return (ConfiguredTargetValue)
        getExistingValue(
            skyframeExecutor,
            ConfiguredTargetKey.builder().setLabel(label).setConfiguration(config).build());
  }

  /**
   * Returns the configured target for an existing configured target value, or {@code null} if there
   * is not an appropriate configured target value key in the graph.
   *
   * <p>This helper is provided so legacy tests don't need to know about details of skyframe keys.
   */
  @Nullable
  public static ConfiguredTarget getExistingConfiguredTarget(
      SkyframeExecutor skyframeExecutor, Label label, BuildConfigurationValue config)
      throws InterruptedException {
    ConfiguredTargetValue value = getExistingConfiguredTargetValue(skyframeExecutor, label, config);
    if (value == null) {
      return null;
    }
    return value.getConfiguredTarget();
  }

  /**
   * Returns all configured targets currently in the graph with the given label.
   *
   * <p>Unlike {@link #getExistingConfiguredTarget(SkyframeExecutor, Label,
   * BuildConfigurationValue)}, this doesn't make the caller request a specific configuration.
   */
  public static Iterable<ConfiguredTarget> getExistingConfiguredTargets(
      SkyframeExecutor skyframeExecutor, Label label) {
    return Iterables.filter(
        getAllExistingConfiguredTargets(skyframeExecutor), input -> input.getLabel().equals(label));
  }

  /** Returns all configured targets currently in the graph. */
  public static ImmutableList<ConfiguredTarget> getAllExistingConfiguredTargets(
      SkyframeExecutor skyframeExecutor) {
    Collection<SkyValue> values =
        Maps.filterKeys(
                skyframeExecutor.getEvaluator().getValues(),
                SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET))
            .values();
    var cts = ImmutableList.<ConfiguredTarget>builder();
    for (SkyValue value : values) {
      if (value != null) {
        cts.add(((ConfiguredTargetValue) value).getConfiguredTarget());
      }
    }
    return cts.build();
  }

  /**
   * Returns the target for an existing target value, or {@code null} if there is not an appropriate
   * target value key in the graph.
   *
   * <p>This helper is provided so legacy tests don't need to know about details of skyframe keys.
   */
  @Nullable
  public static Target getExistingTarget(SkyframeExecutor skyframeExecutor, Label label)
      throws InterruptedException {
    PackageValue value =
        (PackageValue) getExistingValue(skyframeExecutor, label.getPackageIdentifier());
    if (value == null) {
      return null;
    }
    try {
      return value.getPackage().getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      return null;
    }
  }

  /**
   * Returns the error info for an existing target value, or {@code null} if there is not an
   * appropriate target value key in the graph.
   *
   * <p>This helper is provided so legacy tests don't need to know about details of skyframe keys.
   */
  @Nullable
  public static ErrorInfo getExistingFailedPackage(SkyframeExecutor skyframeExecutor, Label label)
      throws InterruptedException {
    SkyKey key = label.getPackageIdentifier();
    return getExistingError(skyframeExecutor, key);
  }
}
