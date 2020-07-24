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
import com.google.devtools.build.lib.actions.ActionLookupKey;
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
public class ConfiguredTargetKey implements ActionLookupKey {
  /**
   * Cache so that the number of ConfiguredTargetKey instances is {@code O(configured targets)} and
   * not {@code O(edges between configured targets)}.
   */
  private static final Interner<ConfiguredTargetKey> interner = BlazeInterners.newWeakInterner();

  private final Label label;
  @Nullable private final BuildConfigurationValue.Key configurationKey;

  private transient int hashCode;

  ConfiguredTargetKey(Label label, @Nullable BuildConfigurationValue.Key configurationKey) {
    this.label = Preconditions.checkNotNull(label);
    this.configurationKey = configurationKey;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static ConfiguredTargetKey create(
      Label label, @Nullable BuildConfigurationValue.Key configurationKey) {
    return interner.intern(new ConfiguredTargetKey(label, configurationKey));
  }

  @Override
  public final Label getLabel() {
    return label;
  }

  @Override
  public final SkyFunctionName functionName() {
    return SkyFunctions.CONFIGURED_TARGET;
  }

  @Nullable
  final BuildConfigurationValue.Key getConfigurationKey() {
    return configurationKey;
  }

  @Nullable
  ToolchainContextKey getToolchainContextKey() {
    return null;
  }

  @Override
  public final int hashCode() {
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
    int toolchainContextVal =
        getToolchainContextKey() == null ? 47 : getToolchainContextKey().hashCode();
    return 31 * label.hashCode() + configVal + toolchainContextVal;
  }

  @Override
  public final boolean equals(Object obj) {
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
        && Objects.equals(configurationKey, other.configurationKey)
        && Objects.equals(getToolchainContextKey(), other.getToolchainContextKey());
  }

  public final String prettyPrint() {
    if (label == null) {
      return "null";
    }
    return label.toString();
  }

  @Override
  public final String toString() {
    if (getToolchainContextKey() != null) {
      return String.format("%s %s %s", label, configurationKey, getToolchainContextKey());
    }
    return String.format("%s %s", label, configurationKey);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class ConfiguredTargetKeyWithToolchainContext extends ConfiguredTargetKey {
    private static final Interner<ConfiguredTargetKeyWithToolchainContext>
        withToolchainContextInterner = BlazeInterners.newWeakInterner();

    private final ToolchainContextKey toolchainContextKey;

    private ConfiguredTargetKeyWithToolchainContext(
        Label label,
        @Nullable BuildConfigurationValue.Key configurationKey,
        ToolchainContextKey toolchainContextKey) {
      super(label, configurationKey);
      this.toolchainContextKey = toolchainContextKey;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static ConfiguredTargetKeyWithToolchainContext create(
        Label label,
        @Nullable BuildConfigurationValue.Key configurationKey,
        ToolchainContextKey toolchainContextKey) {
      return withToolchainContextInterner.intern(
          new ConfiguredTargetKeyWithToolchainContext(
              label, configurationKey, toolchainContextKey));
    }

    @Override
    @Nullable
    final ToolchainContextKey getToolchainContextKey() {
      return toolchainContextKey;
    }
  }

  /** Returns a new {@link Builder} to create instances of {@link ConfiguredTargetKey}. */
  public static Builder builder() {
    return new Builder();
  }

  /** A helper class to create instances of {@link ConfiguredTargetKey}. */
  public static class Builder {
    private Label label = null;
    private BuildConfigurationValue.Key configurationKey = null;
    private ToolchainContextKey toolchainContextKey = null;

    /** Sets the label for the target. */
    public Builder setLabel(Label label) {
      this.label = label;
      return this;
    }

    /**
     * Sets the {@link ConfiguredTarget} that we want a key for.
     *
     * <p>This sets both the label and configurationKey data.
     */
    public Builder setConfiguredTarget(ConfiguredTarget configuredTarget) {
      setLabel(configuredTarget.getOriginalLabel());
      if (this.configurationKey == null) {
        setConfigurationKey(configuredTarget.getConfigurationKey());
      }
      return this;
    }

    /** Sets the {@link BuildConfiguration} for the configured target. */
    public Builder setConfiguration(@Nullable BuildConfiguration buildConfiguration) {
      if (buildConfiguration == null) {
        return setConfigurationKey(null);
      } else {
        return setConfigurationKey(BuildConfigurationValue.key(buildConfiguration));
      }
    }

    /** Sets the configuration key for the configured target. */
    public Builder setConfigurationKey(@Nullable BuildConfigurationValue.Key configurationKey) {
      this.configurationKey = configurationKey;
      return this;
    }

    /**
     * Sets the {@link ToolchainContextKey} this configured target should use for toolchain
     * resolution. When present, this overrides the normally determined toolchain context.
     */
    public Builder setToolchainContextKey(ToolchainContextKey toolchainContextKey) {
      this.toolchainContextKey = toolchainContextKey;
      return this;
    }

    /** Builds a new {@link ConfiguredTargetKey} based on the supplied data. */
    public ConfiguredTargetKey build() {
      if (this.toolchainContextKey != null) {
        return ConfiguredTargetKeyWithToolchainContext.create(
            label, configurationKey, toolchainContextKey);
      }
      return create(label, configurationKey);
    }
  }
}
