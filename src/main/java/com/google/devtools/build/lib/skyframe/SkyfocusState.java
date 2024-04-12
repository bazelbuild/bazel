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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Skyfocus;
import com.google.devtools.build.lib.server.FailureDetails.Skyfocus.Code;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;

/**
 * An immutable record of the state of Skyfocus. This is recorded as a member in {@link
 * SkyframeExecutor}.
 */
public record SkyfocusState(
    boolean enabled,
    ImmutableSet<String> workingSet,
    ImmutableSet<SkyKey> verificationSet,
    @Nullable SkyfocusOptions options,
    @Nullable BuildConfigurationValue buildConfiguration,
    Request request) {

  /** Determines how (or if) Skyfocus should run during this build. */
  public enum Request {
    /**
     * Blaze has to reset the evaluator state and restart analysis before running Skyfocus. This is
     * usually due to Skyfocus having dropped nodes in a prior invocation, and there's no way to
     * recover from it. This can be expensive.
     */
    RERUN_ANALYSIS_THEN_RUN_FOCUS,

    /** Trigger Skyfocus. */
    RUN_FOCUS,

    DO_NOTHING
  }

  /** The canonical state to completely disable Skyfocus in the build. */
  public static final SkyfocusState DISABLED =
      new SkyfocusState(
          false, ImmutableSet.of(), ImmutableSet.of(), null, null, Request.DO_NOTHING);

  public SkyfocusState withEnabled(boolean val) {
    return new SkyfocusState(
        val, workingSet, verificationSet, options, buildConfiguration, request);
  }

  public SkyfocusState withWorkingSet(ImmutableSet<String> val) {
    return new SkyfocusState(enabled, val, verificationSet, options, buildConfiguration, request);
  }

  public SkyfocusState withVerificationSet(ImmutableSet<SkyKey> val) {
    return new SkyfocusState(enabled, workingSet, val, options, buildConfiguration, request);
  }

  public SkyfocusState withOptions(SkyfocusOptions val) {
    return new SkyfocusState(
        enabled, workingSet, verificationSet, val, buildConfiguration, request);
  }

  public SkyfocusState withBuildConfiguration(BuildConfigurationValue val) {
    return new SkyfocusState(enabled, workingSet, verificationSet, options, val, request);
  }

  public SkyfocusState withRequest(Request val) {
    return new SkyfocusState(
        enabled, workingSet, verificationSet, options, buildConfiguration, val);
  }

  public Request checkBuildConfigChanges(
      BuildConfigurationValue newConfig, Request originalRequest, EventHandler eventHandler)
      throws AbruptExitException {
    if (buildConfiguration() == null || buildConfiguration().equals(newConfig)) {
      return originalRequest;
    }

    return switch (options.handlingStrategy) {
      case WARN -> {
        eventHandler.handle(
            Event.warn(
                "Skyfocus: detected changes to the build configuration, will be discarding the"
                    + " analysis cache."));

        yield Request.RUN_FOCUS;
      }
      case STRICT ->
          throw new AbruptExitException(
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(
                          "Skyfocus: detected changes to the build configuration. This is not"
                              + " allowed in a focused build. Either clean to reset the build, or"
                              + " set --experimental_skyfocus_handling_strategy=warn to perform a"
                              + " full reanalysis instead of failing the build.")
                      .setSkyfocus(Skyfocus.newBuilder().setCode(Code.CONFIGURATION_CHANGE).build())
                      .build()));
    };
  }
}
