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
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * In simple form, a ({@link Label}, {@link BuildConfigurationValue}) pair used to trigger immediate
 * dependency resolution and the rule analysis.
 *
 * <p>In practice, a ({@link Label} and post-transition {@link BuildConfigurationKey}) pair plus a
 * possible execution platform override {@link Label} with special constraints. To elaborate, in
 * order of highest to lowest potential for concern:
 *
 * <p>1. The {@link BuildConfigurationKey} must be post-transition and thus ready for immediate use
 * in dependency resolution and analysis. In practice, this means that if the rule has an
 * incoming-edge transition (cfg in {@link RuleClass}) or there are global trimming transitions,
 * THOSE TRANSITIONS MUST ALREADY BE DONE before creating the key. Failure to do so will lead to
 * build graphs with ConfiguredTarget that have seemingly impossible {@link BuildConfigurationValue}
 * (due to the skipped transitions).
 *
 * <p>2. A build should not request keys with equal ({@link Label}, {@link BuildConfigurationValue})
 * pairs but different execution platform override {@link Label} if the invoked rule will register
 * actions. (This is potentially OK if all outputs of all registered actions incorporate the
 * execution platform in their name unless the build also requests keys without an override that
 * happen to resolve to the same execution platform.) In practice, this issue has not been seen in
 * any 'real' builds; however, pathologically failure could lead to multiple (potentially different)
 * ConfiguredTarget that have the same ({@link Label}, {@link BuildConfigurationValue}) pair.
 *
 * <p>Note that this key may be used to look up the generating action of an artifact.
 *
 * <p>TODO(blaze-configurability-team): Consider just using BuildOptions over a
 * BuildConfigurationKey.
 */
@AutoCodec
public class ConfiguredTargetKey extends ActionLookupKey {
  /**
   * Cache so that the number of ConfiguredTargetKey instances is {@code O(configured targets)} and
   * not {@code O(edges between configured targets)}.
   */
  private static final SkyKeyInterner<ConfiguredTargetKey> interner = SkyKey.newInterner();

  private final Label label;
  @Nullable private final BuildConfigurationKey configurationKey;

  private final transient int hashCode;

  ConfiguredTargetKey(Label label, @Nullable BuildConfigurationKey configurationKey, int hashCode) {
    this.label = checkNotNull(label);
    this.configurationKey = configurationKey;
    this.hashCode = hashCode;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static ConfiguredTargetKey create(Label label, @Nullable BuildConfigurationKey configurationKey) {
    int hashCode = computeHashCode(label, configurationKey, /*executionPlatformLabel=*/ null);
    return interner.intern(new ConfiguredTargetKey(label, configurationKey, hashCode));
  }

  public Builder toBuilder() {
    return builder()
        .setConfigurationKey(configurationKey)
        .setLabel(label)
        .setExecutionPlatformLabel(getExecutionPlatformLabel());
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
  @Override
  public final BuildConfigurationKey getConfigurationKey() {
    return configurationKey;
  }

  @Nullable
  public Label getExecutionPlatformLabel() {
    return null;
  }

  @Override
  public final int hashCode() {
    return hashCode;
  }

  private static int computeHashCode(
      Label label,
      @Nullable BuildConfigurationKey configurationKey,
      @Nullable Label executionPlatformLabel) {
    int configVal = configurationKey == null ? 79 : configurationKey.hashCode();
    int executionPlatformLabelVal =
        executionPlatformLabel == null ? 47 : executionPlatformLabel.hashCode();
    return 31 * label.hashCode() + configVal + executionPlatformLabelVal;
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
        && Objects.equals(getExecutionPlatformLabel(), other.getExecutionPlatformLabel());
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
    if (getExecutionPlatformLabel() != null) {
      helper.add("executionPlatformLabel", getExecutionPlatformLabel());
    }
    return helper.toString();
  }

  @Override
  public SkyKeyInterner<? extends ConfiguredTargetKey> getSkyKeyInterner() {
    return interner;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class ToolchainDependencyConfiguredTargetKey extends ConfiguredTargetKey {
    private static final SkyKeyInterner<ToolchainDependencyConfiguredTargetKey>
        toolchainDependencyConfiguredTargetKeyInterner = SkyKey.newInterner();

    private final Label executionPlatformLabel;

    private ToolchainDependencyConfiguredTargetKey(
        Label label,
        @Nullable BuildConfigurationKey configurationKey,
        int hashCode,
        Label executionPlatformLabel) {
      super(label, configurationKey, hashCode);
      this.executionPlatformLabel = checkNotNull(executionPlatformLabel);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static ToolchainDependencyConfiguredTargetKey create(
        Label label,
        @Nullable BuildConfigurationKey configurationKey,
        Label executionPlatformLabel) {
      int hashCode = computeHashCode(label, configurationKey, executionPlatformLabel);
      return toolchainDependencyConfiguredTargetKeyInterner.intern(
          new ToolchainDependencyConfiguredTargetKey(
              label, configurationKey, hashCode, executionPlatformLabel));
    }

    @Override
    public final Label getExecutionPlatformLabel() {
      return executionPlatformLabel;
    }

    @Override
    public final SkyKeyInterner<? extends ConfiguredTargetKey> getSkyKeyInterner() {
      return toolchainDependencyConfiguredTargetKeyInterner;
    }
  }

  /** Returns a new {@link Builder} to create instances of {@link ConfiguredTargetKey}. */
  public static Builder builder() {
    return new Builder();
  }

  /** Returns a new {@link ConfiguredTargetKey}. */
  public static ConfiguredTargetKey fromConfiguredTarget(ConfiguredTarget configuredTarget) {
    return builder()
        .setLabel(configuredTarget.getOriginalLabel())
        .setConfigurationKey(configuredTarget.getConfigurationKey())
        .build();
  }

  /** A helper class to create instances of {@link ConfiguredTargetKey}. */
  public static final class Builder {
    private Label label = null;
    private BuildConfigurationKey configurationKey = null;
    private Label executionPlatformLabel = null;

    private Builder() {}

    /** Sets the label for the target. */
    @CanIgnoreReturnValue
    public Builder setLabel(Label label) {
      this.label = label;
      return this;
    }

    /** Sets the {@link BuildConfigurationValue} for the configured target. */
    @CanIgnoreReturnValue
    public Builder setConfiguration(@Nullable BuildConfigurationValue buildConfiguration) {
      return setConfigurationKey(buildConfiguration == null ? null : buildConfiguration.getKey());
    }

    /** Sets the configuration key for the configured target. */
    @CanIgnoreReturnValue
    public Builder setConfigurationKey(@Nullable BuildConfigurationKey configurationKey) {
      this.configurationKey = configurationKey;
      return this;
    }

    /**
     * Sets the execution platform {@link Label} this configured target should use for toolchain
     * resolution. When present, this overrides the normally determined execution platform.
     */
    @CanIgnoreReturnValue
    public Builder setExecutionPlatformLabel(@Nullable Label executionPlatformLabel) {
      this.executionPlatformLabel = executionPlatformLabel;
      return this;
    }

    /** Builds a new {@link ConfiguredTargetKey} based on the supplied data. */
    public ConfiguredTargetKey build() {
      if (this.executionPlatformLabel != null) {
        return ToolchainDependencyConfiguredTargetKey.create(
            label, configurationKey, executionPlatformLabel);
      }
      return create(label, configurationKey);
    }
  }
}
