// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

// Note: AutoValue v1.4-rc1 has AutoValue.CopyAnnotations which makes it work with Starlark. No need
// to un-AutoValue this class to expose it to Starlark.
/**
 * Specifies how to generate language-specific code from .proto files. Used by LANG_proto_library
 * rules.
 */
@AutoValue
@AutoCodec
public abstract class ProtoLangToolchainProvider implements TransitiveInfoProvider {
  public abstract String commandLine();

  @Nullable
  public abstract FilesToRunProvider pluginExecutable();

  @Nullable
  public abstract TransitiveInfoCollection runtime();

  /**
   * Returns a list of {@link ProtoSource}s that are already provided by the protobuf runtime (i.e.
   * for which {@code <lang>_proto_library} should not generate bindings.
   */
  public abstract ImmutableList<ProtoSource> providedProtoSources();

  /**
   * This makes the blacklisted_protos member available in the provider. It can be removed after
   * users are migrated and a sufficient time for Bazel rules to migrate has elapsed.
   */
  @Deprecated
  public NestedSet<Artifact> blacklistedProtos() {
    return forbiddenProtos();
  }

  // TODO(yannic): Remove after migrating all users to `providedProtoSources()`.
  @Deprecated
  public abstract NestedSet<Artifact> forbiddenProtos();

  @AutoCodec.Instantiator
  public static ProtoLangToolchainProvider createForDeserialization(
      String commandLine,
      FilesToRunProvider pluginExecutable,
      TransitiveInfoCollection runtime,
      ImmutableList<ProtoSource> providedProtoSources,
      NestedSet<Artifact> blacklistedProtos) {
    return new AutoValue_ProtoLangToolchainProvider(
        commandLine, pluginExecutable, runtime, providedProtoSources, blacklistedProtos);
  }

  public static ProtoLangToolchainProvider create(
      String commandLine,
      FilesToRunProvider pluginExecutable,
      TransitiveInfoCollection runtime,
      ImmutableList<ProtoSource> providedProtoSources) {
    NestedSetBuilder<Artifact> blacklistedProtos = NestedSetBuilder.stableOrder();
    for (ProtoSource protoSource : providedProtoSources) {
      blacklistedProtos.add(protoSource.getOriginalSourceFile());
    }
    return new AutoValue_ProtoLangToolchainProvider(
        commandLine, pluginExecutable, runtime, providedProtoSources, blacklistedProtos.build());
  }
}
