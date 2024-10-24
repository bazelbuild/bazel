// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.toolchains;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Toolchain.Code;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/** Base class for exceptions that happen during toolchain resolution. */
public abstract class ToolchainException extends Exception implements DetailedException {

  public ToolchainException(String message) {
    super(message);
  }

  public ToolchainException(Throwable cause) {
    super(cause);
  }

  public ToolchainException(String message, Throwable cause) {
    super(message, cause);
  }

  protected abstract Code getDetailedCode();

  @Override
  public DetailedExitCode getDetailedExitCode() {
    if (getCause() instanceof DetailedException) {
      return ((DetailedException) getCause()).getDetailedExitCode();
    }

    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(Strings.nullToEmpty(getMessage()))
            .setToolchain(FailureDetails.Toolchain.newBuilder().setCode(getDetailedCode()))
            .build());
  }

  /**
   * Attempt to find a {@link ConfiguredValueCreationException} in a {@link ToolchainException}, or
   * its causes.
   *
   * <p>If one cannot be found, make a new one.
   */
  public ConfiguredValueCreationException asConfiguredValueCreationException(
      TargetAndConfiguration targetAndConfiguration) {
    for (Throwable cause = getCause();
        cause != null && cause != cause.getCause();
        cause = cause.getCause()) {
      if (cause instanceof ConfiguredValueCreationException configuredValueCreationException) {
        return configuredValueCreationException;
      }
    }
    Cause cause =
        new AnalysisFailedCause(
            targetAndConfiguration.getLabel(),
            configurationIdMessage(targetAndConfiguration.getConfiguration()),
            createDetailedExitCode(
                String.format(
                    "While resolving toolchains for target %s: %s",
                    targetAndConfiguration.getLabel(), getMessage())));
    return new ConfiguredValueCreationException(
        targetAndConfiguration.getTarget(),
        targetAndConfiguration.getConfiguration().getEventId(),
        String.format(
            "While resolving toolchains for target %s: %s", targetAndConfiguration, getMessage()),
        NestedSetBuilder.create(Order.STABLE_ORDER, cause),
        getDetailedExitCode());
  }

  public static ConfigurationId configurationIdMessage(
      @Nullable BuildConfigurationValue configuration) {
    if (configuration == null) {
      return ConfigurationId.newBuilder().setId("none").build();
    }
    return ConfigurationId.newBuilder().setId(configuration.checksum()).build();
  }

  private static DetailedExitCode createDetailedExitCode(String message) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setAnalysis(
                Analysis.newBuilder().setCode(Analysis.Code.CONFIGURED_VALUE_CREATION_FAILED))
            .build());
  }
}
