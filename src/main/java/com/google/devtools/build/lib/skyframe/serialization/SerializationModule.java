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

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteAnalysisCaching.Code;
import com.google.devtools.build.lib.skyframe.serialization.analysis.FrontierSerializer;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.Optional;
import javax.annotation.Nullable;

/** A {@link BlazeModule} to store Skyframe serialization lifecycle hooks. */
public class SerializationModule extends BlazeModule {

  @Nullable private RemoteAnalysisCachingOptions options;

  // Retained only for analysis caching operations in afterCommand. Must be garbage collected once
  // it's no longer needed. See the javadoc for CommandEnvironment for more information.
  @Nullable private CommandEnvironment env;

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
        SerializationRegistrySetupHelpers.createAnalysisCodecRegistrySupplier(
            runtime,
            SerializationRegistrySetupHelpers.makeReferenceConstants(
                directories,
                runtime.getRuleClassProvider(),
                directories.getWorkspace().getBaseName())));
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    RemoteAnalysisCachingOptions options =
        env.getOptions().getOptions(RemoteAnalysisCachingOptions.class);
    if (options == null) {
      // not a supported command
      return;
    }

    if (!isNullOrEmpty(options.serializedFrontierProfile)) {
      this.options = options;
      this.env = env;
    }
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    if (options == null) {
      return;
    }

    try {
      if (!isNullOrEmpty(options.serializedFrontierProfile)) {
        Optional<FailureDetail> failureDetail =
            FrontierSerializer.dumpFrontierSerializationProfile(
                env, options.serializedFrontierProfile);
        if (failureDetail.isPresent()) {
          throw new AbruptExitException(DetailedExitCode.of(failureDetail.get()));
        }
      }
    } catch (InterruptedException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FrontierSerializer.createFailureDetail(
                  "frontier serializer was interrupted: " + e.getMessage(),
                  Code.SERIALIZED_FRONTIER_PROFILE_FAILED)));
    } finally {
      // Do not retain these objects between invocations.
      this.options = null;
      this.env = null;
      System.gc();
    }
  }
}
