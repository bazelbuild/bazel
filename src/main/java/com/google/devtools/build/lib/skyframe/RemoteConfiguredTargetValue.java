// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetData;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * A {@link ConfiguredTargetValue} fetched from a remote source.
 *
 * <p>This doesn't contain actions, but contains enough information for dependents to perform
 * analysis. In particular, contains {@link TargetData}, allowing the construction of {@link
 * ConfiguredTargetAndData}, containing everything needed by dependents of the {@link
 * ConfiguredTargetValue} in analysis.
 */
public final class RemoteConfiguredTargetValue implements ConfiguredTargetValue {

  @Nullable // Null after clearing.
  private ConfiguredTarget configuredTarget;

  @Nullable // Null after clearing.
  private TargetData targetData;

  private RemoteConfiguredTargetValue(ConfiguredTarget configuredTarget, TargetData targetData) {
    this.configuredTarget = configuredTarget;
    this.targetData = targetData;
  }

  @Nullable // Null after clearing everything.
  @Override
  public ConfiguredTarget getConfiguredTarget() {
    return configuredTarget;
  }

  @Nullable // Never serialized.
  @Override
  public NestedSet<Package> getTransitivePackages() {
    return null;
  }

  @Override
  public void clear(boolean clearEverything) {
    if (clearEverything) {
      configuredTarget = null;
      targetData = null;
    }
  }

  @Nullable // Null after clearing everything.
  public TargetData getTargetData() {
    return targetData;
  }

  @Override
  public String toString() {
    return toStringHelper(this)
        .add("configuredTarget", configuredTarget)
        .add("targetData", targetData)
        .toString();
  }

  public static ConfiguredTargetValueCodec codec() {
    return ConfiguredTargetValueCodec.INSTANCE;
  }

  /**
   * Codec for {@link ConfiguredTargetValue}s.
   *
   * <p>This codec is crafted to serialize the minimal amount of data needed by its rdeps.
   *
   * <p>The serialized constituents are: the {@link ConfiguredTarget}, followed by its (compact)
   * {@link TargetData}, if it already exists. Otherwise, the {@link TargetData} will be constructed
   * from the {@link Target} in the {@link Package} dep.
   */
  @Keep // Accessed reflectively.
  private static class ConfiguredTargetValueCodec
      extends DeferredObjectCodec<ConfiguredTargetValue> {

    private static final ConfiguredTargetValueCodec INSTANCE = new ConfiguredTargetValueCodec();

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<ConfiguredTargetValue> getEncodedClass() {
      return ConfiguredTargetValue.class;
    }

    @Override
    public ImmutableSet<Class<? extends ConfiguredTargetValue>> additionalEncodedClasses() {
      return ImmutableSet.of(
          RuleConfiguredTargetValue.class,
          NonRuleConfiguredTargetValue.class,
          RemoteConfiguredTargetValue.class);
    }

    @Override
    public void serialize(
        SerializationContext context, ConfiguredTargetValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      ConfiguredTarget configuredTarget =
          checkNotNull(
              obj.getConfiguredTarget(),
              "tried to serialize a cleared ConfiguredTargetValue? %s",
              obj);
      context.serialize(configuredTarget, codedOut);
      if (obj instanceof RemoteConfiguredTargetValue value) {
        context.serialize(value.targetData, codedOut);
        return;
      }

      // Looks up the Target and serializes it as TargetData.
      Label label = configuredTarget.getLabel();
      var pkgFunction = context.getDependency(PrerequisitePackageFunction.class);
      Package pkg;
      try {
        pkg = pkgFunction.getExistingPackage(label.getPackageIdentifier());
      } catch (InterruptedException e) {
        throw new SerializationException(
            "serialization of ConfiguredTargetValue "
                + configuredTarget.getLabel()
                + " interrupted while looking up its package",
            e);
      }

      Target target;
      try {
        target = pkg.getTarget(label.getName());
      } catch (NoSuchTargetException e) {
        throw new IllegalStateException(
            "The target associated with " + configuredTarget + " was unexpectedly missing", e);
      }
      context.serialize(target.reduceForSerialization(), codedOut);
    }

    @Override
    public DeferredValue<RemoteConfiguredTargetValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var value = new DeserializationBuilder();
      context.deserialize(codedIn, value, DeserializationBuilder::setConfiguredTarget);
      context.deserialize(codedIn, value, DeserializationBuilder::setTargetData);
      return value;
    }

    private static class DeserializationBuilder
        implements DeferredValue<RemoteConfiguredTargetValue> {

      private final RemoteConfiguredTargetValue value;

      private DeserializationBuilder() {
        this.value =
            new RemoteConfiguredTargetValue(/* configuredTarget= */ null, /* targetData= */ null);
      }

      @Override
      public RemoteConfiguredTargetValue call() {
        return value;
      }

      private static void setConfiguredTarget(DeserializationBuilder builder, Object value) {
        builder.value.configuredTarget = (ConfiguredTarget) value;
      }

      private static void setTargetData(DeserializationBuilder builder, Object value) {
        builder.value.targetData = (TargetData) value;
      }
    }
  }
}
