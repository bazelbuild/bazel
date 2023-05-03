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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.util.HashCodes.hashObjects;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupKeyOrProxy;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
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
 * <p>The {@link ConfiguredTargetKey} is not a {@link SkyKey} and must be cast to one using {@link
 * ActionLookupKeyOrProxy#toKey}.
 *
 * <p>TODO(blaze-configurability-team): Consider just using BuildOptions over a
 * BuildConfigurationKey.
 */
public abstract class ConfiguredTargetKey implements ActionLookupKeyOrProxy {
  /**
   * Cache so that the number of ConfiguredTargetKey instances is {@code O(configured targets)} and
   * not {@code O(edges between configured targets)}.
   */
  private static final SkyKey.SkyKeyInterner<SkyKey> interner = SkyKey.newInterner();

  @Nullable private final BuildConfigurationKey configurationKey;
  private final transient int hashCode;

  private ConfiguredTargetKey(@Nullable BuildConfigurationKey configurationKey, int hashCode) {
    this.configurationKey = configurationKey;
    this.hashCode = hashCode;
  }

  public Builder toBuilder() {
    return builder()
        .setConfigurationKey(configurationKey)
        .setLabel(getLabel())
        .setExecutionPlatformLabel(getExecutionPlatformLabel());
  }

  @Nullable
  @Override
  public final BuildConfigurationKey getConfigurationKey() {
    return configurationKey;
  }

  public abstract Label getExecutionPlatformLabel();

  @Override
  public final int hashCode() {
    return hashCode;
  }

  public boolean isProxy() {
    return false;
  }

  private static int computeHashCode(
      Label label,
      @Nullable BuildConfigurationKey configurationKey,
      @Nullable Label executionPlatformLabel) {
    return hashObjects(label, configurationKey, executionPlatformLabel);
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
        && getLabel().equals(other.getLabel())
        && Objects.equals(configurationKey, other.configurationKey)
        && Objects.equals(getExecutionPlatformLabel(), other.getExecutionPlatformLabel());
  }

  public String prettyPrint() {
    if (getLabel() == null) {
      return "null";
    }
    return String.format("%s (%s)", getLabel(), formatConfigurationKey(configurationKey));
  }

  private static ConfiguredTargetKey intern(ConfiguredTargetKey key) {
    return (ConfiguredTargetKey) interner.intern((SkyKey) key);
  }

  @Override
  public String toString() {
    // TODO(b/162809183): consider reverting to less verbose toString when bug is resolved.
    MoreObjects.ToStringHelper helper =
        MoreObjects.toStringHelper(this).add("label", getLabel()).add("config", configurationKey);
    if (getExecutionPlatformLabel() != null) {
      helper.add("executionPlatformLabel", getExecutionPlatformLabel());
    }
    return helper.toString();
  }

  private static final class RealConfiguredTargetKey extends ConfiguredTargetKey
      implements ActionLookupKey {
    private final Label label;

    private RealConfiguredTargetKey(
        Label label, @Nullable BuildConfigurationKey configurationKey, int hashCode) {
      super(configurationKey, hashCode);
      this.label = label;
    }

    static ConfiguredTargetKey create(
        Label label, @Nullable BuildConfigurationKey configurationKey) {
      int hashCode = computeHashCode(label, configurationKey, /* executionPlatformLabel= */ null);
      return intern(new RealConfiguredTargetKey(label, configurationKey, hashCode));
    }

    @Override
    public final SkyFunctionName functionName() {
      return SkyFunctions.CONFIGURED_TARGET;
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Nullable
    @Override
    public Label getExecutionPlatformLabel() {
      return null;
    }
  }

  private static final class ToolchainDependencyConfiguredTargetKey extends ConfiguredTargetKey
      implements ActionLookupKey {
    private final Label label;
    private final Label executionPlatformLabel;

    private ToolchainDependencyConfiguredTargetKey(
        Label label,
        @Nullable BuildConfigurationKey configurationKey,
        int hashCode,
        Label executionPlatformLabel) {
      super(configurationKey, hashCode);
      this.label = label;
      this.executionPlatformLabel = checkNotNull(executionPlatformLabel);
    }

    private static ConfiguredTargetKey create(
        Label label,
        @Nullable BuildConfigurationKey configurationKey,
        Label executionPlatformLabel) {
      int hashCode = computeHashCode(label, configurationKey, executionPlatformLabel);
      return intern(
          new ToolchainDependencyConfiguredTargetKey(
              label, configurationKey, hashCode, executionPlatformLabel));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.CONFIGURED_TARGET;
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    public Label getExecutionPlatformLabel() {
      return executionPlatformLabel;
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
    }
  }

  // This class implements SkyKey only so that it can share the interner. It should never be used as
  // a SkyKey.
  private static final class ProxyConfiguredTargetKey extends ConfiguredTargetKey
      implements SkyKey {
    private final ConfiguredTargetKey delegate;

    private static ConfiguredTargetKey create(
        ConfiguredTargetKey delegate, @Nullable BuildConfigurationKey configurationKey) {
      int hashCode =
          computeHashCode(
              delegate.getLabel(), configurationKey, delegate.getExecutionPlatformLabel());
      return intern(new ProxyConfiguredTargetKey(delegate, configurationKey, hashCode));
    }

    private ProxyConfiguredTargetKey(
        ConfiguredTargetKey delegate,
        @Nullable BuildConfigurationKey configurationKey,
        int hashCode) {
      super(configurationKey, hashCode);
      checkArgument(
          !delegate.isProxy(), "Proxy keys must not be nested: %s %s", delegate, configurationKey);
      this.delegate = delegate;
    }

    @Override
    public SkyFunctionName functionName() {
      // ProxyConfiguredTargetKey is never used directly by Skyframe. It must always be cast using
      // toKey.
      throw new UnsupportedOperationException();
    }

    @Override
    public Label getLabel() {
      return delegate.getLabel();
    }

    @Override
    @Nullable
    public Label getExecutionPlatformLabel() {
      return delegate.getExecutionPlatformLabel();
    }

    @Override
    public ActionLookupKey toKey() {
      return (ActionLookupKey) delegate;
    }

    @Override
    public boolean isProxy() {
      return true;
    }

    @Override
    public Builder toBuilder() {
      return new Builder().setDelegate(delegate).setConfigurationKey(getConfigurationKey());
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("delegate", delegate)
          .add("config", getConfigurationKey())
          .toString();
    }

    @Override
    public String prettyPrint() {
      return super.prettyPrint()
          + " virtual("
          + formatConfigurationKey(getConfigurationKey())
          + ")";
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
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
    private ConfiguredTargetKey delegate;

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

    /**
     * If set, creates a {@link ProxyConfiguredTargetKey}.
     *
     * <p>It's invalid to set a label or execution platform label if this is set. Those will be
     * defined by the corresponding values of {@code delegate}.
     */
    @CanIgnoreReturnValue
    public Builder setDelegate(ConfiguredTargetKey delegate) {
      this.delegate = delegate;
      return this;
    }

    /** Builds a new {@link ConfiguredTargetKey} based on the supplied data. */
    public ConfiguredTargetKey build() {
      if (this.delegate != null) {
        checkArgument(label == null);
        checkArgument(executionPlatformLabel == null);
        return ProxyConfiguredTargetKey.create(delegate, configurationKey);
      }
      if (this.executionPlatformLabel != null) {
        return ToolchainDependencyConfiguredTargetKey.create(
            label, configurationKey, executionPlatformLabel);
      }
      return RealConfiguredTargetKey.create(label, configurationKey);
    }
  }

  private static String formatConfigurationKey(@Nullable BuildConfigurationKey key) {
    if (key == null) {
      return "null";
    }
    return key.getOptions().checksum();
  }

  /**
   * Codec for all {@link ConfiguredTargetKey} subtypes.
   *
   * <p>By design, {@link ProxyConfiguredTargetKey} serializes as a key without delegation. Upon
   * deserialization, if the key is locally delegated, it becomes delegating again due to interning.
   * If not, it deserializes to the appropriate non-delegating key.
   */
  @Keep
  private static class ConfiguredTargetKeyCodec implements ObjectCodec<ConfiguredTargetKey> {
    @Override
    public Class<ConfiguredTargetKey> getEncodedClass() {
      return ConfiguredTargetKey.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ConfiguredTargetKey key, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(key.getLabel(), codedOut);
      context.serialize(key.getConfigurationKey(), codedOut);
      context.serialize(key.getExecutionPlatformLabel(), codedOut);
    }

    @Override
    public ConfiguredTargetKey deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return builder()
          .setLabel(context.deserialize(codedIn))
          .setConfigurationKey(context.deserialize(codedIn))
          .setExecutionPlatformLabel(context.deserialize(codedIn))
          .build();
    }
  }
}
