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
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A (Label, Configuration key) pair. Note that this pair may be used to look up the generating
 * action of an artifact.
 */
public class ConfiguredTargetKey extends ActionLookupKey {
  private final Label label;
  @Nullable private final SkyKey configurationKey;

  private transient int hashCode;

  private ConfiguredTargetKey(Label label, @Nullable SkyKey configurationKey) {
    this.label = Preconditions.checkNotNull(label);
    this.configurationKey = configurationKey;
  }

  public static ConfiguredTargetKey of(ConfiguredTarget configuredTarget) {
    AliasProvider aliasProvider = configuredTarget.getProvider(AliasProvider.class);
    Label label =
        aliasProvider != null ? aliasProvider.getAliasChain().get(0) : configuredTarget.getLabel();
    return of(label, configuredTarget.getConfiguration());
  }

  /**
   * Caches so that the number of ConfiguredTargetKey instances is {@code O(configured targets)} and
   * not {@code O(edges between configured targets)}.
   */
  private static final Interner<ConfiguredTargetKey> interner = BlazeInterners.newWeakInterner();
  private static final Interner<HostConfiguredTargetKey> hostInterner =
      BlazeInterners.newWeakInterner();

  public static ConfiguredTargetKey of(Label label, @Nullable BuildConfiguration configuration) {
    SkyKey configurationKey =
        configuration == null
            ? null
            : BuildConfigurationValue.key(
            configuration.fragmentClasses(), configuration.getOptions());
    return of(
        label,
        configurationKey,
        configuration != null && configuration.isHostConfiguration());
  }

  static ConfiguredTargetKey of(
      Label label, @Nullable SkyKey configurationKey, boolean isHostConfiguration) {
    if (isHostConfiguration) {
      return hostInterner.intern(new HostConfiguredTargetKey(label, configurationKey));
    } else {
      return interner.intern(new ConfiguredTargetKey(label, configurationKey));
    }
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
  SkyKey getConfigurationKey() {
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
    return 31 * label.hashCode() + configVal + (isHostConfiguration() ? 41 : 0);
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
    return this.isHostConfiguration() == other.isHostConfiguration()
        && Objects.equals(label, other.label)
        && Objects.equals(configurationKey, other.configurationKey);
  }

  public boolean isHostConfiguration() {
    return false;
  }

  public String prettyPrint() {
    if (label == null) {
      return "null";
    }
    return isHostConfiguration() ? (label + " (host)") : label.toString();
  }

  @Override
  public String toString() {
    return String.format(
        "%s %s %s (%s)",
        label,
        configurationKey,
        isHostConfiguration(),
        System.identityHashCode(this));
  }

  private static class HostConfiguredTargetKey extends ConfiguredTargetKey {
    private HostConfiguredTargetKey(Label label, @Nullable SkyKey configurationKey) {
      super(label, configurationKey);
    }

    @Override
    public boolean isHostConfiguration() {
      return true;
    }
  }
}
