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
package com.google.devtools.build.lib.rules.java;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Strategy;
import javax.annotation.concurrent.Immutable;

/**
 * A {@link TransitiveInfoProvider} that aggregates {@link ProtoJavaApiInfoProvider} for propogation
 * in an aspect.
 */
@AutoCodec(strategy = Strategy.AUTO_VALUE_BUILDER)
@Immutable
@AutoValue
public abstract class ProtoJavaApiInfoAspectProvider implements TransitiveInfoProvider {

  public static Builder builder() {
    return new AutoValue_ProtoJavaApiInfoAspectProvider.Builder();
  }

  public static ProtoJavaApiInfoAspectProvider merge(
      Iterable<ProtoJavaApiInfoAspectProvider> providers) {
    ProtoJavaApiInfoAspectProvider.Builder protoBuilder = ProtoJavaApiInfoAspectProvider.builder();
    for (ProtoJavaApiInfoAspectProvider provider : providers) {
      protoBuilder.addTransitive(provider);
    }
    return protoBuilder.build();
  }

  public abstract NestedSet<ProtoJavaApiInfoProvider> getProviders();

  /** A builder for {@link ProtoJavaApiInfoProvider}. */
  @AutoValue.Builder
  public abstract static class Builder {

    private final NestedSetBuilder<ProtoJavaApiInfoProvider> providers =
        NestedSetBuilder.stableOrder();

    public Builder add(ProtoJavaApiInfoProvider provider) {
      providers.add(provider);
      return this;
    }

    public Builder addTransitive(ProtoJavaApiInfoAspectProvider provider) {
      providers.addTransitive(provider.getProviders());
      return this;
    }

    abstract Builder setProviders(NestedSet<ProtoJavaApiInfoProvider> providers);

    abstract ProtoJavaApiInfoAspectProvider autoBuild();

    public ProtoJavaApiInfoAspectProvider build() {
      return setProviders(providers.build()).autoBuild();
    }
  }
}
