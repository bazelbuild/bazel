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
package com.google.devtools.build.lib.skyframe.config;

import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.OutputPathMnemonicComputer;
import com.google.devtools.build.lib.analysis.config.OutputPathMnemonicComputer.InvalidMnemonicException;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Looks up the output directory mnemonic of a configuration from the done {@link
 * BuildConfigurationValue} and {@link BaselineOptionsValue} nodes in the Skyframe graph.
 *
 * <p>Used by {@code ArtifactCodecs} when serializing derived artifacts to bytes (both when
 * uploading values to the remote analysis cache and when serializing an {@code Artifact} {@link
 * SkyKey} to look up its cached value); not used when deserializing bytes from the cache.
 */
public final class ConfigurationMnemonicLookup {

  private record CachedMnemonic(BaselineOptionsValue baseline, String mnemonic) {}

  /** Returns the value of a done node in the Skyframe graph, or {@code null}. */
  private final Function<SkyKey, SkyValue> doneValues;

  private final ConcurrentHashMap<BuildConfigurationKey, CachedMnemonic> cachedMnemonics =
      new ConcurrentHashMap<>();

  public ConfigurationMnemonicLookup(Function<SkyKey, SkyValue> doneValues) {
    this.doneValues = doneValues;
  }

  /** Returns the output directory mnemonic for {@code key}. */
  public String getMnemonic(BuildConfigurationKey key) throws SerializationException {
    if (doneValues.apply(key) instanceof BuildConfigurationValue configuration) {
      return configuration.getMnemonic();
    }
    // Locally evaluated targets always have a BuildConfigurationValue in the graph. Everything
    // below is only for serializing an artifact whose owning target was fetched from the remote
    // cache (so it has no BuildConfigurationValue in the graph, but deserializing its root fetched
    // BaselineOptionsValue via Skyframe), or for the no-config configuration.
    BaselineOptionsValue.Key baselineKey =
        BuildConfigurationFunction.baselineOptionsKey(key.getOptions());
    if (baselineKey == null) {
      return computeMnemonic(key, /* baselineOptions= */ null);
    }
    if (!(doneValues.apply(baselineKey) instanceof BaselineOptionsValue baseline)) {
      throw new SerializationException("Output directory mnemonic of " + key + " is unknown");
    }
    CachedMnemonic cached = cachedMnemonics.get(key);
    if (cached != null && cached.baseline().equals(baseline)) {
      return cached.mnemonic();
    }
    String mnemonic = computeMnemonic(key, baseline.toOptions());
    cachedMnemonics.put(key, new CachedMnemonic(baseline, mnemonic));
    return mnemonic;
  }

  private static String computeMnemonic(
      BuildConfigurationKey key, @Nullable BuildOptions baselineOptions)
      throws SerializationException {
    try {
      return OutputPathMnemonicComputer.computeMnemonic(key.getOptions(), baselineOptions)
          .mnemonic();
    } catch (InvalidMnemonicException e) {
      throw new SerializationException("Invalid mnemonic for " + key.getOptions().checksum(), e);
    }
  }
}
