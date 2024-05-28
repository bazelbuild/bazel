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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/** A non-rule configured target in the context of a Skyframe graph. */
@Immutable
@ThreadSafe
public final class NonRuleConfiguredTargetValue
    extends BaseRuleConfiguredTargetValue<ConfiguredTarget> implements ConfiguredTargetValue {

  NonRuleConfiguredTargetValue(
      ConfiguredTarget configuredTarget, @Nullable NestedSet<Package> transitivePackages) {
    super(configuredTarget, transitivePackages);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("configuredTarget", getConfiguredTarget())
        .toString();
  }

  @Keep // used reflectively
  private static final class Codec extends DeferredObjectCodec<NonRuleConfiguredTargetValue> {
    @Override
    public Class<NonRuleConfiguredTargetValue> getEncodedClass() {
      return NonRuleConfiguredTargetValue.class;
    }

    @Override
    public void serialize(
        SerializationContext context, NonRuleConfiguredTargetValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      // Reached via OutputFileConfiguredTarget.
      context.addExplicitlyAllowedClass(RuleConfiguredTarget.class);
      context.serialize(obj.getConfiguredTarget(), codedOut);
    }

    @Override
    public DeferredValue<NonRuleConfiguredTargetValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = new DeserializationBuilder();
      context.deserialize(codedIn, builder, DeserializationBuilder::setConfiguredTarget);
      return builder;
    }
  }

  private static final class DeserializationBuilder
      implements DeferredObjectCodec.DeferredValue<NonRuleConfiguredTargetValue> {
    private ConfiguredTarget configuredTarget;

    @Override
    public NonRuleConfiguredTargetValue call() {
      return new NonRuleConfiguredTargetValue(configuredTarget, /* transitivePackages= */ null);
    }

    private static void setConfiguredTarget(DeserializationBuilder builder, Object value) {
      builder.configuredTarget = (ConfiguredTarget) value;
    }
  }
}
