// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import javax.annotation.Nullable;

/** A base class for keys that have AspectValue as a Sky value. */
public abstract class AspectValueKey extends ActionLookupKey {

  private static final Interner<AspectKey> aspectKeyInterner = BlazeInterners.newWeakInterner();
  private static final Interner<StarlarkAspectLoadingKey> starlarkAspectKeyInterner =
      BlazeInterners.newWeakInterner();

  public abstract String getDescription();

  @Override
  public abstract Label getLabel();

  // Methods to create aspect keys.

  public static AspectKey createAspectKey(
      Label label,
      @Nullable BuildConfiguration baseConfiguration,
      ImmutableList<AspectKey> baseKeys,
      AspectDescriptor aspectDescriptor,
      @Nullable BuildConfiguration aspectConfiguration) {
    return AspectKey.createAspectKey(
        ConfiguredTargetKey.of(label, baseConfiguration),
        baseKeys,
        aspectDescriptor,
        aspectConfiguration == null ? null : BuildConfigurationValue.key(aspectConfiguration));
  }

  public static AspectKey createAspectKey(
      Label label,
      @Nullable BuildConfiguration baseConfiguration,
      AspectDescriptor aspectDescriptor,
      @Nullable BuildConfiguration aspectConfiguration) {
    return AspectKey.createAspectKey(
        ConfiguredTargetKey.of(label, baseConfiguration),
        ImmutableList.of(),
        aspectDescriptor,
        aspectConfiguration == null ? null : BuildConfigurationValue.key(aspectConfiguration));
  }

  public static StarlarkAspectLoadingKey createStarlarkAspectKey(
      Label targetLabel,
      @Nullable BuildConfiguration aspectConfiguration,
      @Nullable BuildConfiguration targetConfiguration,
      Label starlarkFileLabel,
      String starlarkExportName) {
    StarlarkAspectLoadingKey key =
        new StarlarkAspectLoadingKey(
            targetLabel,
            aspectConfiguration == null ? null : BuildConfigurationValue.key(aspectConfiguration),
            ConfiguredTargetKey.of(targetLabel, targetConfiguration),
            starlarkFileLabel,
            starlarkExportName);

    return starlarkAspectKeyInterner.intern(key);
  }

  // Specific subtypes of aspect keys.

  /** A base class for a key representing an aspect applied to a particular target. */
  @AutoCodec
  public static class AspectKey extends AspectValueKey {
    private final ImmutableList<AspectKey> baseKeys;
    @Nullable private final BuildConfigurationValue.Key aspectConfigurationKey;
    private final ConfiguredTargetKey baseConfiguredTargetKey;
    private final AspectDescriptor aspectDescriptor;
    private int hashCode;

    private AspectKey(
        @Nullable BuildConfigurationValue.Key aspectConfigurationKey,
        ConfiguredTargetKey baseConfiguredTargetKey,
        ImmutableList<AspectKey> baseKeys,
        AspectDescriptor aspectDescriptor) {
      this.baseKeys = baseKeys;
      this.aspectConfigurationKey = aspectConfigurationKey;
      this.baseConfiguredTargetKey = baseConfiguredTargetKey;
      this.aspectDescriptor = aspectDescriptor;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static AspectKey createAspectKey(
        ConfiguredTargetKey baseConfiguredTargetKey,
        ImmutableList<AspectKey> baseKeys,
        AspectDescriptor aspectDescriptor,
        @Nullable BuildConfigurationValue.Key aspectConfigurationKey) {
      return aspectKeyInterner.intern(
          new AspectKey(
              aspectConfigurationKey, baseConfiguredTargetKey, baseKeys, aspectDescriptor));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ASPECT;
    }


    @Override
    public Label getLabel() {
      return baseConfiguredTargetKey.getLabel();
    }

    public AspectClass getAspectClass() {
      return aspectDescriptor.getAspectClass();
    }

    @Nullable
    public AspectParameters getParameters() {
      return aspectDescriptor.getParameters();
    }

    public AspectDescriptor getAspectDescriptor() {
      return aspectDescriptor;
    }

    @Nullable
    ImmutableList<AspectKey> getBaseKeys() {
      return baseKeys;
    }

    @Override
    public String getDescription() {
      if (baseKeys.isEmpty()) {
        return String.format("%s of %s",
            aspectDescriptor.getAspectClass().getName(), getLabel());
      } else {
        return String.format("%s on top of %s",
            aspectDescriptor.getAspectClass().getName(), baseKeys.toString());
      }
    }

    /**
     * Returns the key of the configured target of the aspect; that is, the configuration in which
     * the aspect will be evaluated.
     *
     * <p>In trimmed configuration mode, the aspect may require more fragments than the target on
     * which it is being evaluated; in addition to configuration fragments required by the target
     * and its dependencies, an aspect has configuration fragment requirements of its own, as well
     * as dependencies of its own with their own configuration fragment requirements.
     *
     * <p>The aspect configuration contains all of these fragments, and is used to create the
     * aspect's RuleContext and to retrieve the dependencies. Note that dependencies will have their
     * configurations trimmed from this one as normal.
     *
     * <p>Because of these properties, this configuration is always a superset of the base target's
     * configuration. In untrimmed configuration mode, this configuration will be equivalent to the
     * base target's configuration.
     */
    @Nullable
    BuildConfigurationValue.Key getAspectConfigurationKey() {
      return aspectConfigurationKey;
    }

    /** Returns the key for the base configured target for this aspect. */
    ConfiguredTargetKey getBaseConfiguredTargetKey() {
      return baseConfiguredTargetKey;
    }

    @Override
    public int hashCode() {
      // We use the hash code caching strategy employed by java.lang.String. There are three subtle
      // things going on here:
      //
      // (1) We use a value of 0 to indicate that the hash code hasn't been computed and cached yet.
      // Yes, this means that if the hash code is really 0 then we will "recompute" it each time.
      // But this isn't a problem in practice since a hash code of 0 should be rare.
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
      return Objects.hashCode(
          baseKeys, aspectConfigurationKey, baseConfiguredTargetKey, aspectDescriptor);
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }

      if (!(other instanceof AspectKey)) {
        return false;
      }

      AspectKey that = (AspectKey) other;
      return Objects.equal(baseKeys, that.baseKeys)
          && Objects.equal(aspectConfigurationKey, that.aspectConfigurationKey)
          && Objects.equal(baseConfiguredTargetKey, that.baseConfiguredTargetKey)
          && Objects.equal(aspectDescriptor, that.aspectDescriptor);
    }

    public String prettyPrint() {
      if (getLabel() == null) {
        return "null";
      }

      String baseKeysString =
          baseKeys.isEmpty()
          ? ""
          : String.format(" (over %s)", baseKeys.toString());
      return String.format(
          "%s with aspect %s%s",
          getLabel(), aspectDescriptor.getAspectClass().getName(), baseKeysString);
    }

    @Override
    public String toString() {
      return (baseKeys == null ? getLabel() : baseKeys.toString())
          + "#"
          + aspectDescriptor
          + " "
          + aspectConfigurationKey
          + " "
          + baseConfiguredTargetKey
          + " "
          + aspectDescriptor.getParameters();
    }

    AspectKey withLabel(Label label) {
      ImmutableList.Builder<AspectKey> newBaseKeys = ImmutableList.builder();
      for (AspectKey baseKey : baseKeys) {
        newBaseKeys.add(baseKey.withLabel(label));
      }

      return createAspectKey(
          ConfiguredTargetKey.builder()
              .setLabel(label)
              .setConfigurationKey(baseConfiguredTargetKey.getConfigurationKey())
              .build(),
          newBaseKeys.build(),
          aspectDescriptor,
          aspectConfigurationKey);
    }
  }

  /** The key for a Starlark aspect. */
  public static class StarlarkAspectLoadingKey extends AspectValueKey {

    private final Label targetLabel;
    private final BuildConfigurationValue.Key aspectConfigurationKey;
    private final ConfiguredTargetKey baseConfiguredTargetKey;
    private final Label starlarkFileLabel;
    private final String starlarkValueName;
    private int hashCode;

    private StarlarkAspectLoadingKey(
        Label targetLabel,
        BuildConfigurationValue.Key aspectConfigurationKey,
        ConfiguredTargetKey baseConfiguredTargetKey,
        Label starlarkFileLabel,
        String starlarkFunctionName) {
      this.targetLabel = targetLabel;
      this.aspectConfigurationKey = aspectConfigurationKey;
      this.baseConfiguredTargetKey = baseConfiguredTargetKey;
      this.starlarkFileLabel = starlarkFileLabel;
      this.starlarkValueName = starlarkFunctionName;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.LOAD_STARLARK_ASPECT;
    }

    String getStarlarkValueName() {
      return starlarkValueName;
    }

    Label getStarlarkFileLabel() {
      return starlarkFileLabel;
    }

    protected boolean isAspectConfigurationHost() {
      return false;
    }

    @Override
    public Label getLabel() {
      return targetLabel;
    }

    @Override
    public String getDescription() {
      // Starlark aspects are referred to on command line with <file>%<value ame>
      return String.format("%s%%%s of %s", starlarkFileLabel, starlarkValueName, targetLabel);
    }

    @Override
    public int hashCode() {
      // We use the hash code caching strategy employed by java.lang.String. There are three subtle
      // things going on here:
      //
      // (1) We use a value of 0 to indicate that the hash code hasn't been computed and cached yet.
      // Yes, this means that if the hash code is really 0 then we will "recompute" it each time.
      // But this isn't a problem in practice since a hash code of 0 should be rare.
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
      return Objects.hashCode(
          targetLabel,
          aspectConfigurationKey,
          baseConfiguredTargetKey,
          starlarkFileLabel,
          starlarkValueName);
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }

      if (!(o instanceof StarlarkAspectLoadingKey)) {
        return false;
      }
      StarlarkAspectLoadingKey that = (StarlarkAspectLoadingKey) o;
      return Objects.equal(targetLabel, that.targetLabel)
          && Objects.equal(aspectConfigurationKey, that.aspectConfigurationKey)
          && Objects.equal(baseConfiguredTargetKey, that.baseConfiguredTargetKey)
          && Objects.equal(starlarkFileLabel, that.starlarkFileLabel)
          && Objects.equal(starlarkValueName, that.starlarkValueName);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("targetLabel", targetLabel)
          .add("aspectConfigurationKey", aspectConfigurationKey)
          .add("baseConfiguredTargetKey", baseConfiguredTargetKey)
          .add("starlarkFileLabel", starlarkFileLabel)
          .add("starlarkValueName", starlarkValueName)
          .toString();
    }

    AspectKey toAspectKey(AspectClass aspectClass) {
      return AspectKey.createAspectKey(
          baseConfiguredTargetKey,
          ImmutableList.of(),
          new AspectDescriptor(aspectClass, AspectParameters.EMPTY),
          aspectConfigurationKey);
    }
  }
}
