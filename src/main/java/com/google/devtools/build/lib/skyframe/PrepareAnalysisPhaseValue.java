// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver.TopLevelTargetsAndConfigsResult;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A value referring to a set of build configuration keys in order to reconstruct the
 * legacy {@link BuildConfigurationCollection} as well as a set of top level configured target keys
 * that are subsequently requested to trigger the analysis phase.
 *
 * <p>The public interface returns {@link BuildConfigurationCollection} and {@link
 * TargetAndConfiguration} even though these are not internally stored - the construction of these
 * objects requires additional Skyframe calls. The intention is that these are temporary until a
 * larger fraction of the code has been ported to Skyframe, at which point we'll use the internal
 * representation.
 */
@Immutable
@ThreadSafe
@AutoCodec
public final class PrepareAnalysisPhaseValue implements SkyValue {
  private final BuildConfigurationValue.Key hostConfigurationKey;
  private final ImmutableList<BuildConfigurationValue.Key> targetConfigurationKeys;
  private final ImmutableList<ConfiguredTargetKey> topLevelCtKeys;

  PrepareAnalysisPhaseValue(
      BuildConfigurationValue.Key hostConfigurationKey,
      ImmutableList<BuildConfigurationValue.Key> targetConfigurationKeys,
      ImmutableList<ConfiguredTargetKey> topLevelCtKeys) {
    this.hostConfigurationKey = Preconditions.checkNotNull(hostConfigurationKey);
    this.targetConfigurationKeys = Preconditions.checkNotNull(targetConfigurationKeys);
    this.topLevelCtKeys = Preconditions.checkNotNull(topLevelCtKeys);
  }

  /**
   * Returns the legacy {@link BuildConfigurationCollection}. Note that this performs additional
   * Skyframe calls, which may be expensive.
   */
  public BuildConfigurationCollection getConfigurations(
      ExtendedEventHandler eventHandler, SkyframeExecutor skyframeExecutor)
          throws InvalidConfigurationException {
    BuildConfiguration hostConfiguration =
        skyframeExecutor.getConfiguration(eventHandler, hostConfigurationKey);
    ImmutableList<BuildConfiguration> targetConfigurations =
        ImmutableList.copyOf(
            skyframeExecutor.getConfigurations(eventHandler, targetConfigurationKeys).values());
    return new BuildConfigurationCollection(targetConfigurations, hostConfiguration);
  }

  /**
   * Returns the intended top-level targets and configurations for the build. Note that this
   * performs additional Skyframe calls for the involved configurations and targets, which may be
   * expensive.
   *
   * <p>Skips targets that have errors and registers the errors to be reported later as part of
   * {@link com.google.devtools.build.lib.analysis.AnalysisResult} error resolution.
   */
  public TopLevelTargetsAndConfigsResult getTopLevelCts(
      ExtendedEventHandler eventHandler, SkyframeExecutor skyframeExecutor) {
    List<TargetAndConfiguration> result = new ArrayList<>();
    Map<BuildConfigurationValue.Key, BuildConfiguration> configs =
        skyframeExecutor.getConfigurations(
            eventHandler,
            topLevelCtKeys.stream()
                .map(ctk -> ctk.getConfigurationKey())
                .filter(Predicates.notNull())
                .collect(Collectors.toSet()));

    // TODO(ulfjack): This performs one Skyframe call per top-level target. This is not a
    // regression, but we should fix it nevertheless, either by doing a bulk lookup call or by
    // migrating the consumers of these to Skyframe so they can directly request the values.
    boolean hasError = false;
    for (ConfiguredTargetKey key : topLevelCtKeys) {
      Target target;
      try {
        target = skyframeExecutor.getPackageManager().getTarget(eventHandler, key.getLabel());
      } catch (NoSuchPackageException | NoSuchTargetException | InterruptedException e) {
        eventHandler.handle(
            Event.error("Failed to get package from TargetPatternPhaseValue: " + e.getMessage()));
        hasError = true;
        continue;
      }
      BuildConfiguration config =
          key.getConfigurationKey() == null ? null : configs.get(key.getConfigurationKey());
      result.add(new TargetAndConfiguration(target, config));
    }
    return new TopLevelTargetsAndConfigsResult(result, hasError);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof PrepareAnalysisPhaseValue)) {
      return false;
    }
    PrepareAnalysisPhaseValue that = (PrepareAnalysisPhaseValue) obj;
    return this.hostConfigurationKey.equals(that.hostConfigurationKey)
        && this.targetConfigurationKeys.equals(that.targetConfigurationKeys)
        && this.topLevelCtKeys.equals(that.topLevelCtKeys);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        this.hostConfigurationKey,
        this.targetConfigurationKeys,
        this.topLevelCtKeys);
  }

  /** Create a prepare analysis phase key. */
  @ThreadSafe
  public static SkyKey key(
      FragmentClassSet fragments,
      BuildOptions.OptionsDiffForReconstruction optionsDiff,
      Set<String> multiCpu,
      Collection<Label> labels) {
    return new PrepareAnalysisPhaseKey(fragments, optionsDiff, multiCpu, labels);
  }

  /** The configuration needed to prepare the analysis phase. */
  @ThreadSafe
  @VisibleForSerialization
  @AutoCodec
  public static final class PrepareAnalysisPhaseKey implements SkyKey, Serializable {
    private final FragmentClassSet fragments;
    private final BuildOptions.OptionsDiffForReconstruction optionsDiff;
    private final ImmutableSortedSet<String> multiCpu;
    private final ImmutableSet<Label> labels;

    PrepareAnalysisPhaseKey(
        FragmentClassSet fragments,
        BuildOptions.OptionsDiffForReconstruction optionsDiff,
        Set<String> multiCpu,
        Collection<Label> labels) {
      this.fragments = Preconditions.checkNotNull(fragments);
      this.optionsDiff = Preconditions.checkNotNull(optionsDiff);
      this.multiCpu = ImmutableSortedSet.copyOf(multiCpu);
      this.labels = ImmutableSet.copyOf(labels);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PREPARE_ANALYSIS_PHASE;
    }

    public FragmentClassSet getFragments() {
      return fragments;
    }

    public BuildOptions.OptionsDiffForReconstruction getOptionsDiff() {
      return optionsDiff;
    }

    public ImmutableSortedSet<String> getMultiCpu() {
      return multiCpu;
    }

    public ImmutableSet<Label> getLabels() {
      return labels;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(PrepareAnalysisPhaseKey.class)
          .add("fragments", fragments)
          .add("optionsDiff", optionsDiff)
          .add("multiCpu", multiCpu)
          .add("labels", labels)
          .toString();
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          fragments,
          optionsDiff,
          multiCpu,
          labels);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof PrepareAnalysisPhaseKey)) {
        return false;
      }
      PrepareAnalysisPhaseKey other = (PrepareAnalysisPhaseKey) obj;
      return other.fragments.equals(this.fragments)
          && other.optionsDiff.equals(this.optionsDiff)
          && other.multiCpu.equals(multiCpu)
          && other.labels.equals(labels);
    }
  }
}
