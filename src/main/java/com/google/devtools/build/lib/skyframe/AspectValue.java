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

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ProviderCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue.Key;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey.KeyAndHost;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.skyframe.SkyFunctionName;
import javax.annotation.Nullable;

/** An aspect in the context of the Skyframe graph. */
public final class AspectValue extends BasicActionLookupValue implements ConfiguredObjectValue {
  /**
   * A base class for keys that have AspectValue as a Sky value.
   */
  public abstract static class AspectValueKey extends ActionLookupKey {
    public abstract String getDescription();

    @Override
    public abstract Label getLabel();
  }

  /** A base class for a key representing an aspect applied to a particular target. */
  @AutoCodec
  public static class AspectKey extends AspectValueKey {
    private final Label label;
    private final ImmutableList<AspectKey> baseKeys;
    private final BuildConfigurationValue.Key aspectConfigurationKey;
    private final ConfiguredTargetKey baseConfiguredTargetKey;
    private final AspectDescriptor aspectDescriptor;
    private int hashCode;

    private AspectKey(
        Label label,
        BuildConfigurationValue.Key aspectConfigurationKey,
        ConfiguredTargetKey baseConfiguredTargetKey,
        ImmutableList<AspectKey> baseKeys,
        AspectDescriptor aspectDescriptor) {
      this.baseKeys = baseKeys;
      this.label = label;
      this.aspectConfigurationKey = aspectConfigurationKey;
      this.baseConfiguredTargetKey = baseConfiguredTargetKey;
      this.aspectDescriptor = aspectDescriptor;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static AspectKey createAspectKey(
        Label label,
        ConfiguredTargetKey baseConfiguredTargetKey,
        ImmutableList<AspectKey> baseKeys,
        AspectDescriptor aspectDescriptor,
        BuildConfigurationValue.Key aspectConfigurationKey,
        boolean aspectConfigurationIsHost) {
      return aspectKeyInterner.intern(
          aspectConfigurationIsHost
              ? new HostAspectKey(
                  label,
                  aspectConfigurationKey,
                  baseConfiguredTargetKey,
                  baseKeys,
                  aspectDescriptor)
              : new AspectKey(
                  label,
                  aspectConfigurationKey,
                  baseConfiguredTargetKey,
                  baseKeys,
                  aspectDescriptor));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ASPECT;
    }


    @Override
    public Label getLabel() {
      return label;
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

    // Note that this does not factor into equality/hash-code computations because its value is
    // already encoded in the aspectConfigurationKey, albeit in an opaque way.
    protected boolean aspectConfigurationIsHost() {
      return false;
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
          label, baseKeys, aspectConfigurationKey, baseConfiguredTargetKey, aspectDescriptor);
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
      return Objects.equal(label, that.label)
          && Objects.equal(baseKeys, that.baseKeys)
          && Objects.equal(aspectConfigurationKey, that.aspectConfigurationKey)
          && Objects.equal(baseConfiguredTargetKey, that.baseConfiguredTargetKey)
          && Objects.equal(aspectDescriptor, that.aspectDescriptor);
    }

    public String prettyPrint() {
      if (label == null) {
        return "null";
      }

      String baseKeysString =
          baseKeys.isEmpty()
          ? ""
          : String.format(" (over %s)", baseKeys.toString());
      return String.format(
          "%s with aspect %s%s%s",
          label.toString(),
          aspectDescriptor.getAspectClass().getName(),
          (aspectConfigurationKey != null && aspectConfigurationIsHost()) ? "(host) " : "",
          baseKeysString);
    }

    @Override
    public String toString() {
      return (baseKeys == null ? label : baseKeys.toString())
          + "#"
          + aspectDescriptor
          + " "
          + aspectConfigurationKey
          + " "
          + baseConfiguredTargetKey
          + " "
          + aspectDescriptor.getParameters()
          + (aspectConfigurationIsHost() ? " (host)" : "");
    }

    AspectKey withLabel(Label label) {
      ImmutableList.Builder<AspectKey> newBaseKeys = ImmutableList.builder();
      for (AspectKey baseKey : baseKeys) {
        newBaseKeys.add(baseKey.withLabel(label));
      }

      return createAspectKey(
          label,
          ConfiguredTargetKey.of(
              label,
              baseConfiguredTargetKey.getConfigurationKey(),
              baseConfiguredTargetKey.isHostConfiguration()),
          newBaseKeys.build(),
          aspectDescriptor,
          aspectConfigurationKey,
          aspectConfigurationIsHost());
    }
  }

  /** An {@link AspectKey} for an aspect in the host configuration. */
  static class HostAspectKey extends AspectKey {
    private HostAspectKey(
        Label label,
        Key aspectConfigurationKey,
        ConfiguredTargetKey baseConfiguredTargetKey,
        ImmutableList<AspectKey> baseKeys,
        AspectDescriptor aspectDescriptor) {
      super(label, aspectConfigurationKey, baseConfiguredTargetKey, baseKeys, aspectDescriptor);
    }

    @Override
    protected boolean aspectConfigurationIsHost() {
      return true;
    }
  }

  /** The key for a Starlark aspect. */
  public static class SkylarkAspectLoadingKey extends AspectValueKey {

    private final Label targetLabel;
    private final BuildConfigurationValue.Key aspectConfigurationKey;
    private final ConfiguredTargetKey baseConfiguredTargetKey;
    private final Label skylarkFileLabel;
    private final String skylarkValueName;
    private int hashCode;

    private SkylarkAspectLoadingKey(
        Label targetLabel,
        BuildConfigurationValue.Key aspectConfigurationKey,
        ConfiguredTargetKey baseConfiguredTargetKey,
        Label skylarkFileLabel,
        String skylarkFunctionName) {
      this.targetLabel = targetLabel;
      this.aspectConfigurationKey = aspectConfigurationKey;
      this.baseConfiguredTargetKey = baseConfiguredTargetKey;
      this.skylarkFileLabel = skylarkFileLabel;
      this.skylarkValueName = skylarkFunctionName;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.LOAD_STARLARK_ASPECT;
    }

    String getSkylarkValueName() {
      return skylarkValueName;
    }

    Label getSkylarkFileLabel() {
      return skylarkFileLabel;
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
      return String.format("%s%%%s of %s", skylarkFileLabel, skylarkValueName, targetLabel);
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
          skylarkFileLabel,
          skylarkValueName);
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }

      if (!(o instanceof SkylarkAspectLoadingKey)) {
        return false;
      }
      SkylarkAspectLoadingKey that = (SkylarkAspectLoadingKey) o;
      return Objects.equal(targetLabel, that.targetLabel)
          && Objects.equal(aspectConfigurationKey, that.aspectConfigurationKey)
          && Objects.equal(baseConfiguredTargetKey, that.baseConfiguredTargetKey)
          && Objects.equal(skylarkFileLabel, that.skylarkFileLabel)
          && Objects.equal(skylarkValueName, that.skylarkValueName);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("targetLabel", targetLabel)
          .add("aspectConfigurationKey", aspectConfigurationKey)
          .add("baseConfiguredTargetKey", baseConfiguredTargetKey)
          .add("skylarkFileLabel", skylarkFileLabel)
          .add("skylarkValueName", skylarkValueName)
          .toString();
    }

    AspectKey toAspectKey(AspectClass aspectClass) {
      return AspectKey.createAspectKey(
          targetLabel,
          baseConfiguredTargetKey,
          ImmutableList.of(),
          new AspectDescriptor(aspectClass, AspectParameters.EMPTY),
          aspectConfigurationKey,
          isAspectConfigurationHost());
    }
  }

  /** A {@link SkylarkAspectLoadingKey} for an aspect in the host configuration. */
  private static class HostSkylarkAspectLoadingKey extends SkylarkAspectLoadingKey {

    private HostSkylarkAspectLoadingKey(
        Label targetLabel,
        Key aspectConfigurationKey,
        ConfiguredTargetKey baseConfiguredTargetKey,
        Label skylarkFileLabel,
        String skylarkFunctionName) {
      super(
          targetLabel,
          aspectConfigurationKey,
          baseConfiguredTargetKey,
          skylarkFileLabel,
          skylarkFunctionName);
    }

    @Override
    protected boolean isAspectConfigurationHost() {
      return true;
    }
  }

  // These variables are only non-final because they may be clear()ed to save memory. They are null
  // only after they are cleared except for transitivePackagesForPackageRootResolution.
  @Nullable private Label label;
  @Nullable private Aspect aspect;
  @Nullable private Location location;
  @Nullable private AspectKey key;
  @Nullable private ConfiguredAspect configuredAspect;
  // May be null either after clearing or because transitive packages are not tracked.
  @Nullable private transient NestedSet<Package> transitivePackagesForPackageRootResolution;

  public AspectValue(
      AspectKey key,
      Aspect aspect,
      Label label,
      Location location,
      ConfiguredAspect configuredAspect,
      NestedSet<Package> transitivePackagesForPackageRootResolution) {
    super(configuredAspect.getActions());
    this.label = Preconditions.checkNotNull(label, actions);
    this.aspect = Preconditions.checkNotNull(aspect, label);
    this.location = Preconditions.checkNotNull(location, label);
    this.key = Preconditions.checkNotNull(key, label);
    this.configuredAspect = Preconditions.checkNotNull(configuredAspect, label);
    this.transitivePackagesForPackageRootResolution = transitivePackagesForPackageRootResolution;
  }

  public ConfiguredAspect getConfiguredAspect() {
    return Preconditions.checkNotNull(configuredAspect);
  }

  public Label getLabel() {
    return Preconditions.checkNotNull(label);
  }

  public Location getLocation() {
    return Preconditions.checkNotNull(location);
  }

  public AspectKey getKey() {
    return Preconditions.checkNotNull(key);
  }

  public Aspect getAspect() {
    return Preconditions.checkNotNull(aspect);
  }

  @Override
  public void clear(boolean clearEverything) {
    Preconditions.checkNotNull(label, this);
    Preconditions.checkNotNull(aspect, this);
    Preconditions.checkNotNull(location, this);
    Preconditions.checkNotNull(key, this);
    Preconditions.checkNotNull(configuredAspect, this);
    if (clearEverything) {
      label = null;
      aspect = null;
      location = null;
      key = null;
      configuredAspect = null;
    }
    transitivePackagesForPackageRootResolution = null;
  }

  @Override
  public NestedSet<Package> getTransitivePackagesForPackageRootResolution() {
    return Preconditions.checkNotNull(transitivePackagesForPackageRootResolution);
  }

  @Override
  public String toString() {
    return getStringHelper()
        .add("label", label)
        .add("key", key)
        .add("location", location)
        .add("aspect", aspect)
        .add("configuredAspect", configuredAspect)
        .toString();
  }

  @Override
  public ProviderCollection getConfiguredObject() {
    return getConfiguredAspect();
  }

  public static AspectKey createAspectKey(
      Label label,
      BuildConfiguration baseConfiguration,
      ImmutableList<AspectKey> baseKeys,
      AspectDescriptor aspectDescriptor,
      BuildConfiguration aspectConfiguration) {
    KeyAndHost aspectKeyAndHost = ConfiguredTargetKey.keyFromConfiguration(aspectConfiguration);
    return AspectKey.createAspectKey(
        label,
        ConfiguredTargetKey.of(label, baseConfiguration),
        baseKeys,
        aspectDescriptor,
        aspectKeyAndHost.key,
        aspectKeyAndHost.isHost);
  }

  private static final Interner<AspectKey> aspectKeyInterner = BlazeInterners.newWeakInterner();

  public static AspectKey createAspectKey(
      Label label,
      BuildConfiguration baseConfiguration,
      AspectDescriptor aspectDescriptor,
      BuildConfiguration aspectConfiguration) {
    KeyAndHost aspectKeyAndHost = ConfiguredTargetKey.keyFromConfiguration(aspectConfiguration);
    return AspectKey.createAspectKey(
        label,
        ConfiguredTargetKey.of(label, baseConfiguration),
        ImmutableList.of(),
        aspectDescriptor,
        aspectKeyAndHost.key,
        aspectKeyAndHost.isHost);
  }

  private static final Interner<SkylarkAspectLoadingKey> skylarkAspectKeyInterner =
      BlazeInterners.newWeakInterner();

  public static SkylarkAspectLoadingKey createSkylarkAspectKey(
      Label targetLabel,
      BuildConfiguration aspectConfiguration,
      BuildConfiguration targetConfiguration,
      Label skylarkFileLabel,
      String skylarkExportName) {
    KeyAndHost keyAndHost = ConfiguredTargetKey.keyFromConfiguration(aspectConfiguration);
    SkylarkAspectLoadingKey key =
        keyAndHost.isHost
            ? new HostSkylarkAspectLoadingKey(
                targetLabel,
                keyAndHost.key,
                ConfiguredTargetKey.of(targetLabel, targetConfiguration),
                skylarkFileLabel,
                skylarkExportName)
            : new SkylarkAspectLoadingKey(
                targetLabel,
                keyAndHost.key,
                ConfiguredTargetKey.of(targetLabel, targetConfiguration),
                skylarkFileLabel,
                skylarkExportName);

    return skylarkAspectKeyInterner.intern(key);
  }
}
