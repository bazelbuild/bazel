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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.skyframe.SkyFunctionName;
import javax.annotation.Nullable;

/** The class responsible for creating & interning the various types of AspectKeys. */
public final class AspectKeyCreator {

  private AspectKeyCreator() {}

  private static final Interner<AspectKey> aspectKeyInterner = BlazeInterners.newWeakInterner();
  private static final Interner<TopLevelAspectsKey> topLevelAspectsKeyInterner =
      BlazeInterners.newWeakInterner();

  public static AspectKey createAspectKey(
      Label label,
      @Nullable BuildConfigurationValue baseConfiguration,
      ImmutableList<AspectKey> baseKeys,
      AspectDescriptor aspectDescriptor,
      @Nullable BuildConfigurationValue aspectConfiguration) {
    return AspectKey.createAspectKey(
        ConfiguredTargetKey.builder().setLabel(label).setConfiguration(baseConfiguration).build(),
        baseKeys,
        aspectDescriptor,
        aspectConfiguration == null ? null : aspectConfiguration.getKey());
  }

  public static AspectKey createAspectKey(
      AspectDescriptor aspectDescriptor,
      ImmutableList<AspectKey> baseKeys,
      BuildConfigurationKey aspectConfigurationKey,
      ConfiguredTargetKey baseConfiguredTargetKey) {
    return AspectKey.createAspectKey(
        baseConfiguredTargetKey, baseKeys, aspectDescriptor, aspectConfigurationKey);
  }

  public static AspectKey createAspectKey(
      Label label,
      @Nullable BuildConfigurationValue baseConfiguration,
      AspectDescriptor aspectDescriptor,
      @Nullable BuildConfigurationValue aspectConfiguration) {
    return AspectKey.createAspectKey(
        ConfiguredTargetKey.builder().setLabel(label).setConfiguration(baseConfiguration).build(),
        ImmutableList.of(),
        aspectDescriptor,
        aspectConfiguration == null ? null : aspectConfiguration.getKey());
  }

  public static TopLevelAspectsKey createTopLevelAspectsKey(
      ImmutableList<AspectClass> topLevelAspectsClasses,
      Label targetLabel,
      @Nullable BuildConfigurationValue configuration) {
    return TopLevelAspectsKey.createInternal(
        topLevelAspectsClasses,
        targetLabel,
        ConfiguredTargetKey.builder()
            .setLabel(targetLabel)
            .setConfiguration(configuration)
            .build());
  }

  /** Common superclass for {@link AspectKey} and {@link TopLevelAspectsKey}. */
  public abstract static class AspectBaseKey implements ActionLookupKey {
    private final ConfiguredTargetKey baseConfiguredTargetKey;
    private final int hashCode;

    private AspectBaseKey(ConfiguredTargetKey baseConfiguredTargetKey, int hashCode) {
      this.baseConfiguredTargetKey = baseConfiguredTargetKey;
      this.hashCode = hashCode;
    }

    /** Returns the key for the base configured target for this aspect. */
    public final ConfiguredTargetKey getBaseConfiguredTargetKey() {
      return baseConfiguredTargetKey;
    }

    @Override
    public final int hashCode() {
      return hashCode;
    }
  }

  // Specific subtypes of aspect keys.

  /** Represents an aspect applied to a particular target. */
  public static final class AspectKey extends AspectBaseKey {
    private final ImmutableList<AspectKey> baseKeys;
    @Nullable private final BuildConfigurationKey aspectConfigurationKey;
    private final AspectDescriptor aspectDescriptor;

    private AspectKey(
        ConfiguredTargetKey baseConfiguredTargetKey,
        ImmutableList<AspectKey> baseKeys,
        AspectDescriptor aspectDescriptor,
        @Nullable BuildConfigurationKey aspectConfigurationKey,
        int hashCode) {
      super(baseConfiguredTargetKey, hashCode);
      this.baseKeys = baseKeys;
      this.aspectConfigurationKey = aspectConfigurationKey;
      this.aspectDescriptor = aspectDescriptor;
    }

    @VisibleForTesting
    static AspectKey createAspectKey(
        ConfiguredTargetKey baseConfiguredTargetKey,
        ImmutableList<AspectKey> baseKeys,
        AspectDescriptor aspectDescriptor,
        @Nullable BuildConfigurationKey aspectConfigurationKey) {
      return aspectKeyInterner.intern(
          new AspectKey(
              baseConfiguredTargetKey,
              baseKeys,
              aspectDescriptor,
              aspectConfigurationKey,
              Objects.hashCode(
                  baseConfiguredTargetKey, baseKeys, aspectDescriptor, aspectConfigurationKey)));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ASPECT;
    }

    /**
     * Gets the name of the aspect that would be returned by the corresponding value's {@code
     * aspectValue.getAspect().getAspectClass().getName()}, if the value could be produced.
     *
     * <p>Only needed for reporting errors in BEP when the key's AspectValue fails evaluation.
     */
    public String getAspectName() {
      return aspectDescriptor.getDescription();
    }

    @Override
    public Label getLabel() {
      return getBaseConfiguredTargetKey().getLabel();
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
    public ImmutableList<AspectKey> getBaseKeys() {
      return baseKeys;
    }

    public String getDescription() {
      if (baseKeys.isEmpty()) {
        return String.format("%s of %s", aspectDescriptor.getAspectClass().getName(), getLabel());
      } else {
        return String.format(
            "%s on top of %s", aspectDescriptor.getAspectClass().getName(), baseKeys);
      }
    }

    /**
     * Returns the key of the configured target of the aspect; that is, the configuration in which
     * the aspect will be evaluated.
     */
    @Override
    @Nullable
    public BuildConfigurationKey getConfigurationKey() {
      return aspectConfigurationKey;
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
      return hashCode() == that.hashCode()
          && Objects.equal(baseKeys, that.baseKeys)
          && Objects.equal(aspectConfigurationKey, that.aspectConfigurationKey)
          && Objects.equal(getBaseConfiguredTargetKey(), that.getBaseConfiguredTargetKey())
          && Objects.equal(aspectDescriptor, that.aspectDescriptor);
    }

    public String prettyPrint() {
      if (getLabel() == null) {
        return "null";
      }

      String baseKeysString = baseKeys.isEmpty() ? "" : String.format(" (over %s)", baseKeys);
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
          + getBaseConfiguredTargetKey()
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
              .setConfigurationKey(getBaseConfiguredTargetKey().getConfigurationKey())
              .build(),
          newBaseKeys.build(),
          aspectDescriptor,
          aspectConfigurationKey);
    }
  }

  /** The key for top level aspects specified by --aspects option on a top level target. */
  public static final class TopLevelAspectsKey extends AspectBaseKey {
    private final ImmutableList<AspectClass> topLevelAspectsClasses;
    private final Label targetLabel;

    private static TopLevelAspectsKey createInternal(
        ImmutableList<AspectClass> topLevelAspectsClasses,
        Label targetLabel,
        ConfiguredTargetKey baseConfiguredTargetKey) {
      return topLevelAspectsKeyInterner.intern(
          new TopLevelAspectsKey(
              topLevelAspectsClasses,
              targetLabel,
              baseConfiguredTargetKey,
              Objects.hashCode(topLevelAspectsClasses, targetLabel, baseConfiguredTargetKey)));
    }

    private TopLevelAspectsKey(
        ImmutableList<AspectClass> topLevelAspectsClasses,
        Label targetLabel,
        ConfiguredTargetKey baseConfiguredTargetKey,
        int hashCode) {
      super(baseConfiguredTargetKey, hashCode);
      this.topLevelAspectsClasses = topLevelAspectsClasses;
      this.targetLabel = targetLabel;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TOP_LEVEL_ASPECTS;
    }

    @Override
    public BuildConfigurationKey getConfigurationKey() {
      return getBaseConfiguredTargetKey().getConfigurationKey();
    }

    ImmutableList<AspectClass> getTopLevelAspectsClasses() {
      return topLevelAspectsClasses;
    }

    @Override
    public Label getLabel() {
      return targetLabel;
    }

    String getDescription() {
      return topLevelAspectsClasses + " on " + targetLabel;
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof TopLevelAspectsKey)) {
        return false;
      }
      TopLevelAspectsKey that = (TopLevelAspectsKey) o;
      return hashCode() == that.hashCode()
          && Objects.equal(targetLabel, that.targetLabel)
          && Objects.equal(getBaseConfiguredTargetKey(), that.getBaseConfiguredTargetKey())
          && Objects.equal(topLevelAspectsClasses, that.topLevelAspectsClasses);
    }
  }
}
