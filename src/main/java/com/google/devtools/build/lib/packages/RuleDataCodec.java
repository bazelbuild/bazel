// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule.RuleData;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import net.starlark.java.syntax.Location;

/**
 * A codec for {@link RuleData}.
 *
 * <p>For native rules, serializes {@link RuleClassData} by name using a {@link RuleClassProvider}
 * dependency to look up the {@link RuleClass} on deserialization.
 *
 * <p>For Starlark rules, {@link RuleClassData} is reduced to {@link
 * RuleDataCodec.StarlarkRuleClassData}.
 */
final class RuleDataCodec extends DeferredObjectCodec<RuleData> {
  private static final byte RULE_CLASS_IS_STARLARK = 0b0000_0001;
  private static final byte RULE_TAGS_MASK = 0b0000_0010;
  private static final byte DEPRECATION_WARNING_MASK = 0b0000_0100;
  private static final byte IS_TEST_ONLY_MASK = 0b0000_1000;
  private static final byte TEST_TIMEOUT_MASK = 0b0001_0000;

  @Override
  public Class<RuleData> getEncodedClass() {
    return RuleData.class;
  }

  @Override
  public void serialize(SerializationContext context, RuleData obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    // There are quite a few fields here that are often null or empty. Writing the mask makes it so
    // that nulls and empty sets take 0 additional storage.
    byte presenceMask = 0;
    RuleClassData ruleClassData = obj.getRuleClassData();
    if (ruleClassData.isStarlark()) {
      presenceMask |= RULE_CLASS_IS_STARLARK;
    }
    ImmutableSet<String> ruleTags = obj.getRuleTags();
    if (!ruleTags.isEmpty()) {
      presenceMask |= RULE_TAGS_MASK;
    }
    String deprecationWarning = obj.getDeprecationWarning();
    if (deprecationWarning != null) {
      presenceMask |= DEPRECATION_WARNING_MASK;
    }
    if (obj.isTestOnly()) {
      presenceMask |= IS_TEST_ONLY_MASK;
    }
    TestTimeout testTimeout = obj.getTestTimeout();
    if (testTimeout != null) {
      presenceMask |= TEST_TIMEOUT_MASK;
    }
    codedOut.writeRawByte(presenceMask);

    serializeRuleClassData(context, ruleClassData, codedOut);

    context.serialize(obj.getLocation(), codedOut);
    context.serialize(obj.getLabel(), codedOut);

    if (!ruleTags.isEmpty()) {
      context.serialize(ruleTags, codedOut);
    }
    if (deprecationWarning != null) {
      context.serialize(deprecationWarning, codedOut);
    }
    if (testTimeout != null) {
      context.serialize(testTimeout, codedOut);
    }
  }

  @Override
  public DeferredValue<RuleData> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    byte presenceMask = codedIn.readRawByte();

    Builder builder;
    if ((presenceMask & RULE_CLASS_IS_STARLARK) != 0) {
      StarlarkRuleClassBuilder starlarkBuilder = new StarlarkRuleClassBuilder(presenceMask);
      context.deserialize(codedIn, starlarkBuilder, StarlarkRuleClassBuilder::setRuleClassData);
      builder = starlarkBuilder;
    } else {
      NativeRuleClassBuilder nativeBuilder =
          new NativeRuleClassBuilder(
              presenceMask, context.getDependency(RuleClassProvider.class).getRuleClassMap());
      context.deserialize(codedIn, nativeBuilder, NativeRuleClassBuilder::setRuleClassName);
      builder = nativeBuilder;
    }

    context.deserialize(codedIn, builder, Builder::setLocation);
    context.deserialize(codedIn, builder, Builder::setLabel);

    if ((presenceMask & RULE_TAGS_MASK) != 0) {
      context.deserialize(codedIn, builder, Builder::setRuleTags);
    } else {
      builder.ruleTags = ImmutableSet.of();
    }

    if ((presenceMask & DEPRECATION_WARNING_MASK) != 0) {
      context.deserialize(codedIn, builder, Builder::setDeprecationWarning);
    }

    if ((presenceMask & TEST_TIMEOUT_MASK) != 0) {
      context.deserialize(codedIn, builder, Builder::setTestTimeout);
    }

    return builder;
  }

  /**
   * Builder for deserialized {@link RuleData}.
   *
   * <p>This is abstract due to the differences in deserialization of {@link RuleClassData} for
   * Starlark and native.
   *
   * <ul>
   *   <li>The {@link NativeRuleClassBuilder} uses the {@link RuleClassProvider} serialization
   *       dependency to deserialize a rule class from its name alone.
   *   <li>The {@link StarlarkRuleClassBuilder} directly deserializes the {@link
   *       StarlarkRuleClassData} object.
   * </ul>
   */
  private abstract static class Builder implements DeferredValue<RuleData> {
    private final byte presenceMask;
    private Location location;
    private Label label;
    private ImmutableSet<String> ruleTags;
    private String deprecationWarning;
    private TestTimeout testTimeout;

    Builder(byte presenceMask) {
      this.presenceMask = presenceMask;
    }

    @Override
    public RuleData call() {
      return new RuleData(
          getRuleClassData(),
          location,
          ruleTags,
          label,
          deprecationWarning,
          (presenceMask & IS_TEST_ONLY_MASK) != 0,
          testTimeout);
    }

    abstract RuleClassData getRuleClassData();

    private static void setLocation(Builder builder, Object value) {
      builder.location = (Location) value;
    }

    private static void setLabel(Builder builder, Object value) {
      builder.label = (Label) value;
    }

    @SuppressWarnings("unchecked")
    private static void setRuleTags(Builder builder, Object value) {
      builder.ruleTags = (ImmutableSet<String>) value;
    }

    private static void setDeprecationWarning(Builder builder, Object value) {
      builder.deprecationWarning = (String) value;
    }

    private static void setTestTimeout(Builder builder, Object value) {
      builder.testTimeout = (TestTimeout) value;
    }
  }

  private static class NativeRuleClassBuilder extends Builder {
    private final ImmutableMap<String, RuleClass> ruleClassMap;
    private String ruleClassName;

    private NativeRuleClassBuilder(
        byte presenceMask, ImmutableMap<String, RuleClass> ruleClassMap) {
      super(presenceMask);
      this.ruleClassMap = ruleClassMap;
    }

    @Override
    RuleClassData getRuleClassData() {
      return ruleClassMap.get(ruleClassName);
    }

    private static void setRuleClassName(NativeRuleClassBuilder builder, Object value) {
      builder.ruleClassName = (String) value;
    }
  }

  private static class StarlarkRuleClassBuilder extends Builder {
    private StarlarkRuleClassData ruleClassData;

    private StarlarkRuleClassBuilder(byte presenceMask) {
      super(presenceMask);
    }

    @Override
    StarlarkRuleClassData getRuleClassData() {
      return ruleClassData;
    }

    private static void setRuleClassData(StarlarkRuleClassBuilder builder, Object value) {
      builder.ruleClassData = (StarlarkRuleClassData) value;
    }
  }

  private static void serializeRuleClassData(
      SerializationContext context, RuleClassData obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!obj.isStarlark()) {
      context.serialize(obj.getName(), codedOut);
      return;
    }

    // Handles the case of previously serialized Starlark rule data.
    if (obj instanceof StarlarkRuleClassData) {
      context.serialize(obj, codedOut);
      return;
    }

    // Serializes rule data for Starlark.
    context.serialize(
        new AutoValue_RuleDataCodec_StarlarkRuleClassData(
            obj.getName(),
            obj.getTargetKind(),
            obj.isDependencyResolutionRule(),
            obj.getAdvertisedProviders()),
        codedOut);
  }

  // TODO(b/297857068): to reduce possible value aliasing (which could happen when an instance of
  // this class co-resides on the same JVM as the actual Starlark RuleClass instance), use a .bzl
  // Starlark reference instead.
  @AutoValue
  abstract static class StarlarkRuleClassData implements RuleClassData {
    @Override
    public final boolean isStarlark() {
      return true;
    }
  }
}
