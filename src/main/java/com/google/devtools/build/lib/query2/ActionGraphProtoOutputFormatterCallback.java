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
package com.google.devtools.build.lib.query2;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.output.AqueryOptions;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.actiongraph.ActionGraphDump;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.OutputStream;

/** Default output callback for aquery, prints proto output. */
public class ActionGraphProtoOutputFormatterCallback extends AqueryThreadsafeCallback {

  /** Defines the types of proto output this class can handle. */
  public enum OutputType {
    BINARY("proto"),
    TEXT("textproto");

    private final String formatName;

    OutputType(String formatName) {
      this.formatName = formatName;
    }

    public String formatName() {
      return formatName;
    }
  }

  private final OutputType outputType;
  private final ActionGraphDump actionGraphDump;
  private final AqueryActionFilter actionFilters;

  ActionGraphProtoOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      AqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTargetValue> accessor,
      OutputType outputType,
      AqueryActionFilter actionFilters) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
    this.outputType = outputType;
    this.actionFilters = actionFilters;
    this.actionGraphDump =
        new ActionGraphDump(
            options.includeCommandline, this.actionFilters, options.includeParamFiles);
  }

  @Override
  public String getName() {
    return outputType.formatName();
  }

  @Override
  public void processOutput(Iterable<ConfiguredTargetValue> partialResult)
      throws IOException, InterruptedException {
    try {
      // Enabling includeParamFiles should enable includeCommandline by default.
      options.includeCommandline |= options.includeParamFiles;

      for (ConfiguredTargetValue configuredTargetValue : partialResult) {
        actionGraphDump.dumpConfiguredTarget(configuredTargetValue);
        if (options.useAspects) {
          if (configuredTargetValue.getConfiguredTarget() instanceof RuleConfiguredTarget) {
            for (AspectValue aspectValue : accessor.getAspectValues(configuredTargetValue)) {
              actionGraphDump.dumpAspect(aspectValue, configuredTargetValue);
            }
          }
        }
      }
    } catch (CommandLineExpansionException e) {
      throw new IOException(e.getMessage());
    }
  }

  @Override
  public void close(boolean failFast) throws IOException {
    if (!failFast && printStream != null) {
      ActionGraphContainer actionGraphContainer = actionGraphDump.build();

      // Write the data.
      switch (outputType) {
        case BINARY:
          actionGraphContainer.writeTo(printStream);
          break;
        case TEXT:
          TextFormat.print(actionGraphContainer, printStream);
          break;
        default:
          throw new IllegalStateException("Unknown outputType " + outputType.formatName());
      }
    }
  }

  @VisibleForTesting
  public AnalysisProtos.ActionGraphContainer getProtoResult() {
    return actionGraphDump.build();
  }
}
