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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.trimming.TrimmedConfigurationCache;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Skyframe progress receiver which keeps a {@link TrimmedConfigurationCache} in sync with Skyframe
 * invalidations and revalidations.
 */
public final class TrimmedConfigurationProgressReceiver implements EvaluationProgressReceiver {

  private final TrimmedConfigurationCache<SkyKey, Label, BuildOptions.OptionsDiffForReconstruction>
      cache;

  private boolean enabled = true;

  public TrimmedConfigurationProgressReceiver(
      TrimmedConfigurationCache<SkyKey, Label, BuildOptions.OptionsDiffForReconstruction> cache) {
    this.cache = cache;
  }

  public static TrimmedConfigurationCache<SkyKey, Label, BuildOptions.OptionsDiffForReconstruction>
      buildCache() {
    return new TrimmedConfigurationCache<>(
        TrimmedConfigurationProgressReceiver::extractLabel,
        TrimmedConfigurationProgressReceiver::extractOptionsDiff,
        BuildOptions.OptionsDiffForReconstruction::compareFragments);
  }

  public TrimmedConfigurationCache<SkyKey, Label, BuildOptions.OptionsDiffForReconstruction>
      getCache() {
    return this.cache;
  }

  public static Label extractLabel(SkyKey key) {
    Preconditions.checkArgument(isKeyCacheable(key));
    ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key;
    return configuredTargetKey.getLabel();
  }

  public static BuildOptions.OptionsDiffForReconstruction extractOptionsDiff(SkyKey key) {
    Preconditions.checkArgument(isKeyCacheable(key));
    ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key;
    BuildConfigurationValue.Key configurationKey = configuredTargetKey.getConfigurationKey();
    return configurationKey.getOptionsDiff();
  }

  public static boolean isKeyCacheable(SkyKey key) {
    if (!(key instanceof ConfiguredTargetKey)) {
      // Only configured targets go in the cache.
      // TODO(b/129286648): add aspect support
      return false;
    }
    ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key;
    if (configuredTargetKey.getConfigurationKey() == null) {
      // Null-configured targets do not go in the cache.
      return false;
    }
    return true;
  }

  public void activate() {
    if (this.enabled) {
      return;
    }
    this.enabled = true;
  }

  public void deactivate() {
    if (!this.enabled) {
      return;
    }
    this.enabled = false;
    this.cache.clear();
  }

  @Override
  public void invalidated(SkyKey key, InvalidationState state) {
    if (!enabled || !isKeyCacheable(key)) {
      return;
    }
    switch (state) {
      case DIRTY:
        cache.invalidate(key);
        break;
      case DELETED:
        cache.remove(key);
        break;
    }
  }

  @Override
  public void evaluated(
      SkyKey key,
      @Nullable SkyValue newValue,
      @Nullable ErrorInfo newError,
      Supplier<EvaluationSuccessState> evaluationSuccessState,
      EvaluationState state) {
    if (!enabled || !isKeyCacheable(key)) {
      return;
    }
    switch (state) {
      case BUILT:
        // Do nothing; the evaluation would have handled putting itself (back) in the cache.
        break;
      case CLEAN:
        cache.revalidate(key);
        break;
    }
  }
}
