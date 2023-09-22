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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule.RuleData;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
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
final class RuleDataCodec implements ObjectCodec<RuleData> {
  private static final byte RULE_TAGS_MASK = 0b0000_0001;
  private static final byte DEPRECATION_WARNING_MASK = 0b0000_0010;
  private static final byte IS_TEST_ONLY_MASK = 0b0000_0100;
  private static final byte TEST_TIMEOUT_MASK = 0b0000_1000;

  @Override
  public Class<RuleData> getEncodedClass() {
    return RuleData.class;
  }

  @Override
  public void serialize(SerializationContext context, RuleData obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    serializeRuleClassData(context, obj.getRuleClassData(), codedOut);
    context.serialize(obj.getLocation(), codedOut);
    context.serialize(obj.getLabel(), codedOut);

    // There are quite a few fields here that are often null or empty. Writing the mask makes it so
    // that nulls and empty sets take 0 additional storage.
    byte presenceMask = 0;
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
  public RuleData deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    RuleClassData ruleClassData;
    Object deserializedRuleClassData = context.deserialize(codedIn);
    if (deserializedRuleClassData instanceof String) {
      ruleClassData =
          context
              .getDependency(RuleClassProvider.class)
              .getRuleClassMap()
              .get(deserializedRuleClassData);
    } else {
      ruleClassData = (RuleClassData) deserializedRuleClassData;
    }
    Location location = context.deserialize(codedIn);
    Label label = context.deserialize(codedIn);

    byte presenceMask = codedIn.readRawByte();
    ImmutableSet<String> ruleTags;
    if ((presenceMask & RULE_TAGS_MASK) != 0) {
      ruleTags = context.deserialize(codedIn);
    } else {
      ruleTags = ImmutableSet.of();
    }

    String deprecationWarning;
    if ((presenceMask & DEPRECATION_WARNING_MASK) != 0) {
      deprecationWarning = context.deserialize(codedIn);
    } else {
      deprecationWarning = null;
    }

    TestTimeout testTimeout;
    if ((presenceMask & TEST_TIMEOUT_MASK) != 0) {
      testTimeout = context.deserialize(codedIn);
    } else {
      testTimeout = null;
    }

    return new RuleData(
        ruleClassData,
        location,
        ruleTags,
        label,
        deprecationWarning,
        (presenceMask & IS_TEST_ONLY_MASK) != 0,
        testTimeout);
  }

  private static void serializeRuleClassData(
      SerializationContext context, RuleClassData obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!obj.isStarlark()) {
      context.serialize(obj.getName(), codedOut);
      return;
    }

    if (obj instanceof StarlarkRuleClassData) {
      context.serialize(obj, codedOut);
      return;
    }

    context.serialize(
        new AutoValue_RuleDataCodec_StarlarkRuleClassData(
            obj.getName(), obj.getTargetKind(), obj.getAdvertisedProviders()),
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
