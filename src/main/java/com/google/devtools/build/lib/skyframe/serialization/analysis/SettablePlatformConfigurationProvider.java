// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.PlatformConfigurationProvider;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A write-once holder for {@link PlatformConfigurationProvider} that allows early registration
 * before the underlying configuration and baselines are computed.
 */
public final class SettablePlatformConfigurationProvider implements PlatformConfigurationProvider {
  private final AtomicReference<PlatformConfigurationProvider> delegate = new AtomicReference<>();

  public void setOnce(PlatformConfigurationProvider provider) {
    checkNotNull(provider);
    checkState(
        delegate.compareAndSet(null, provider),
        "PlatformConfigurationProvider has already been initialized");
  }

  private PlatformConfigurationProvider get() {
    PlatformConfigurationProvider provider = delegate.get();
    if (provider == null) {
      throw new IllegalStateException(
          "PlatformConfigurationProvider accessed before BuildView.createConfigurations"
              + " initialized it.");
    }
    return provider;
  }

  @Override
  public Label getTopLevelPlatformLabel() {
    return get().getTopLevelPlatformLabel();
  }

  @Override
  public BuildOptions getBaseOptionsForPlatform(
      Label platformLabel, boolean isExec, boolean trimTestOptions) throws SerializationException {
    return get().getBaseOptionsForPlatform(platformLabel, isExec, trimTestOptions);
  }

  @Override
  public boolean trimTestOptions(BuildOptions options) {
    return get().trimTestOptions(options);
  }

  @Override
  public boolean usePlatformInOutputDir() {
    return get().usePlatformInOutputDir();
  }

  @Override
  public String resolveMnemonic(BuildOptions targetOptions) {
    return get().resolveMnemonic(targetOptions);
  }
}
