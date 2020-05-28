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

import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A (Label, Configuration key) pair. Note that this pair may be used to look up the generating
 * action of an artifact.
 */
@AutoCodec
public class ConfiguredTargetKey extends ActionLookupKey {
  private final Label label;
  @Nullable private final BuildConfigurationValue.Key configurationKey;

  private transient int hashCode;

  private ConfiguredTargetKey(Label label, @Nullable BuildConfigurationValue.Key configurationKey) {
    this.label = Preconditions.checkNotNull(label);
    this.configurationKey = configurationKey;
  }

  // TODO(jcater): Remove as part of converting to Builders.
  public static ConfiguredTargetKey of(
      ConfiguredTarget configuredTarget, BuildConfiguration buildConfiguration) {
    return of(configuredTarget.getOriginalLabel(), buildConfiguration);
  }

  // TODO(jcater): Remove as part of converting to Builders.
  public static ConfiguredTargetKey of(
      ConfiguredTarget configuredTarget,
      BuildConfigurationValue.Key configurationKey,
      boolean ignored) {
    return of(configuredTarget.getOriginalLabel(), configurationKey);
  }

  // TODO(jcater): Remove as part of converting to Builders.
  public static ConfiguredTargetKey inTargetConfig(ConfiguredTarget configuredTarget) {
    return of(configuredTarget.getOriginalLabel(), configuredTarget.getConfigurationKey());
  }

  /**
   * Caches so that the number of ConfiguredTargetKey instances is {@code O(configured targets)} and
   * not {@code O(edges between configured targets)}.
   */
  private static final Interner<ConfiguredTargetKey> interner = BlazeInterners.newWeakInterner();

  public static ConfiguredTargetKey of(Label label, @Nullable BuildConfiguration configuration) {
    return of(
        label, configuration == null ? null : BuildConfigurationValue.key(configuration), false);
  }

  // TODO(jcater): Remove as part of converting to Builders.
  public static ConfiguredTargetKey of(
      Label label, @Nullable BuildConfigurationValue.Key configurationKey, boolean ignored) {
    return of(label, configurationKey);
  }

  @AutoCodec.Instantiator
  public static ConfiguredTargetKey of(
      Label label, @Nullable BuildConfigurationValue.Key configurationKey) {
    return interner.intern(new ConfiguredTargetKey(label, configurationKey));
  }

  // TODO(katre): Remove this.
  static KeyAndHost keyFromConfiguration(@Nullable BuildConfiguration configuration) {
    return configuration == null
        ? KeyAndHost.NULL_INSTANCE
        : new KeyAndHost(
            BuildConfigurationValue.key(configuration), configuration.isHostConfiguration());
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.CONFIGURED_TARGET;
  }

  @Nullable
  BuildConfigurationValue.Key getConfigurationKey() {
    return configurationKey;
  }

  @Override
  public int hashCode() {
    // We use the hash code caching strategy employed by java.lang.String. There are three subtle
    // things going on here:
    //
    // (1) We use a value of 0 to indicate that the hash code hasn't been computed and cached yet.
    // Yes, this means that if the hash code is really 0 then we will "recompute" it each time. But
    // this isn't a problem in practice since a hash code of 0 should be rare.
    //
    // (2) Since we have no synchronization, multiple threads can race here thinking there are the
    // first one to compute and cache the hash code.
    //
    // (3) Moreover, since 'hashCode' is non-volatile, the cached hash code value written from one
    // thread may not be visible by another.
    //
    // All three of these issues are benign from a correctness perspective; in the end we have no
    // overhead from synchronization, at the cost of potentially computing the hash code more than
    // once.
    int h = hashCode;
    if (h == 0) {
      h = computeHashCode();
      hashCode = h;
    }
    return h;
  }

  private int computeHashCode() {
    int configVal = configurationKey == null ? 79 : configurationKey.hashCode();
    return 31 * label.hashCode() + configVal;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof ConfiguredTargetKey)) {
      return false;
    }
    ConfiguredTargetKey other = (ConfiguredTargetKey) obj;
    return Objects.equals(label, other.label)
        && Objects.equals(configurationKey, other.configurationKey);
  }

  public String prettyPrint() {
    if (label == null) {
      return "null";
    }
    return label.toString();
  }

  @Override
  public String toString() {
    return String.format("%s %s", label, configurationKey);
  }

  /**
   * Simple wrapper class for turning a {@link BuildConfiguration} into a {@link
   * BuildConfigurationValue.Key} and boolean isHost.
   */
  public static class KeyAndHost {
    private static final KeyAndHost NULL_INSTANCE = new KeyAndHost(null, false);

    @Nullable public final BuildConfigurationValue.Key key;
    final boolean isHost;

    private KeyAndHost(@Nullable BuildConfigurationValue.Key key, boolean isHost) {
      this.key = key;
      this.isHost = isHost;
    }
  }
}
