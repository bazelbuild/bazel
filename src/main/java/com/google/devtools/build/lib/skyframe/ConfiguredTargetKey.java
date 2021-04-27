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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
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

  private final transient int hashCode;

  ConfiguredTargetKey(
      Label label, @Nullable BuildConfigurationValue.Key configurationKey, int hashCode) {
    this.label = checkNotNull(label);
    this.configurationKey = configurationKey;
    this.hashCode = hashCode;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static ConfiguredTargetKey create(
      Label label, @Nullable BuildConfigurationValue.Key configurationKey) {
    int hashCode = computeHashCode(label, configurationKey, /*toolchainContextKey=*/ null);
    return interner.intern(new ConfiguredTargetKey(label, configurationKey, hashCode));
  }

  public Builder toBuilder() {
    return builder()
        .setConfigurationKey(configurationKey)
        .setLabel(label)
        .setToolchainContextKey(getToolchainContextKey());
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
  public final BuildConfigurationValue.Key getConfigurationKey() {
    return configurationKey;
  }

  @Nullable
  ToolchainContextKey getToolchainContextKey() {
    return null;
  }

  @Override
  public final int hashCode() {
    return hashCode;
  }

  private static int computeHashCode(
      Label label,
      @Nullable BuildConfigurationValue.Key configurationKey,
      @Nullable ToolchainContextKey toolchainContextKey) {
    int configVal = configurationKey == null ? 79 : configurationKey.hashCode();
    int toolchainContextVal = toolchainContextKey == null ? 47 : toolchainContextKey.hashCode();
    return 31 * label.hashCode() + configVal + toolchainContextVal;
  }

  @Override
  public final boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ConfiguredTargetKey)) {
      return false;
    }
    ConfiguredTargetKey other = (ConfiguredTargetKey) obj;
    return hashCode == other.hashCode
        && label.equals(other.label)
        && Objects.equals(configurationKey, other.configurationKey)
        && Objects.equals(getToolchainContextKey(), other.getToolchainContextKey());
  }

  public final String prettyPrint() {
    if (label == null) {
      return "null";
    }
    return String.format(
        "%s (%s)",
        label, configurationKey == null ? "null" : configurationKey.getOptions().checksum());
  }

  @Override
  public final String toString() {
    // TODO(b/162809183): consider reverting to less verbose toString when bug is resolved.
    MoreObjects.ToStringHelper helper =
        MoreObjects.toStringHelper(this).add("label", label).add("config", configurationKey);
    if (getToolchainContextKey() != null) {
      helper.add("toolchainContextKey", getToolchainContextKey());
    }
    return helper.toString();
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
        int hashCode,
        ToolchainContextKey toolchainContextKey) {
      super(label, configurationKey, hashCode);
      this.toolchainContextKey = checkNotNull(toolchainContextKey);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static ConfiguredTargetKeyWithToolchainContext create(
        Label label,
        @Nullable BuildConfigurationValue.Key configurationKey,
        ToolchainContextKey toolchainContextKey) {
      int hashCode = computeHashCode(label, configurationKey, toolchainContextKey);
      return withToolchainContextInterner.intern(
          new ConfiguredTargetKeyWithToolchainContext(
              label, configurationKey, hashCode, toolchainContextKey));
    }

    @Override
    final ToolchainContextKey getToolchainContextKey() {
      return toolchainContextKey;
    }
  }

  /** Returns a new {@link Builder} to create instances of {@link ConfiguredTargetKey}. */
  public static Builder builder() {
    return new Builder();
  }

  /** A helper class to create instances of {@link ConfiguredTargetKey}. */
  public static final class Builder {
    private Label label = null;
    private BuildConfigurationValue.Key configurationKey = null;
    private ToolchainContextKey toolchainContextKey = null;

    private Builder() {}

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
    public Builder setToolchainContextKey(@Nullable ToolchainContextKey toolchainContextKey) {
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
