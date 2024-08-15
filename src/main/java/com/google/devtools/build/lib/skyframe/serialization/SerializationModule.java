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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactCodecs;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.DeferredNestedSetCodec;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.function.Supplier;

/** A {@link BlazeModule} to store Skyframe serialization lifecycle hooks. */
public class SerializationModule extends BlazeModule {

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    if (!directories.inWorkspace()) {
      // Serialization only works when the Bazel server is invoked from a workspace.
      // Counter-example: invoking the Bazel server outside of a workspace to generate/dump
      // documentation HTML.
      return;
    }

    // This is injected as a callback instead of evaluated eagerly to avoid forcing the somewhat
    // expensive AutoRegistry.get call on clients that don't require it.
    runtime.initAnalysisCodecRegistry(
        createAnalysisCodecRegistrySupplier(
            runtime,
            CommonSerializationConstants.makeReferenceConstants(
                directories,
                runtime.getRuleClassProvider(),
                directories.getWorkspace().getBaseName())));
  }

  /**
   * Initializes an {@link ObjectCodecRegistry} for analysis serialization.
   *
   * <p>This gets injected into {@link BlazeRuntime} and made available to clients via {@link
   * BlazeRuntime#getAnalysisCodecRegistry}.
   *
   * <p>TODO: move this to CommonSerializationConstants instead.
   */
  protected static Supplier<ObjectCodecRegistry> createAnalysisCodecRegistrySupplier(
      BlazeRuntime runtime, ImmutableList<Object> additionalReferenceConstants) {
    return () -> {
      ObjectCodecRegistry.Builder builder =
          AutoRegistry.get()
              .getBuilder()
              .addReferenceConstants(additionalReferenceConstants)
              .computeChecksum(false)
              .add(ArrayCodec.forComponentType(Artifact.class))
              .add(new DeferredNestedSetCodec())
              .add(Label.valueSharingCodec())
              .add(PackageIdentifier.valueSharingCodec())
              .add(ConfiguredTargetKey.valueSharingCodec());
      builder =
          CommonSerializationConstants.addStarlarkFunctionality(
              builder, runtime.getRuleClassProvider());
      ArtifactCodecs.VALUE_SHARING_CODECS.forEach(builder::add);
      return builder.build();
    };
  }
}
