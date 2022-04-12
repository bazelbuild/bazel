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
package com.google.devtools.build.lib.query2.aquery;

import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.ActionGraphDump;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.MonolithicOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.StreamedOutputHandler;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

/** Default output callback for aquery, prints proto output. */
public class ActionGraphProtoOutputFormatterCallback extends AqueryThreadsafeCallback {

  private final OutputType outputType;
  private final ActionGraphDump actionGraphDump;
  private final AqueryActionFilter actionFilters;
  private final AqueryOutputHandler aqueryOutputHandler;

  /**
   * Pseudo-arbitrarily chosen buffer size for output. Chosen to be large enough to fit a handful of
   * messages without needing to flush to the underlying output, which may not be buffered.
   */
  private static final int OUTPUT_BUFFER_SIZE = 16384;

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
    this.aqueryOutputHandler = constructAqueryOutputHandler(outputType, out, printStream);
    this.actionGraphDump =
        new ActionGraphDump(
            options.includeCommandline,
            options.includeArtifacts,
            this.actionFilters,
            options.includeParamFiles,
            options.deduplicateDepsets,
            aqueryOutputHandler);
  }

  public static AqueryOutputHandler constructAqueryOutputHandler(
      OutputType outputType, OutputStream out, PrintStream printStream) {
    switch (outputType) {
      case BINARY:
      case TEXT:
        return new StreamedOutputHandler(
            outputType, CodedOutputStream.newInstance(out, OUTPUT_BUFFER_SIZE), printStream);
      case JSON:
        return new MonolithicOutputHandler(printStream);
    }
    // The above cases are exhaustive.
    throw new AssertionError("Wrong output type: " + outputType);
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
        if (!(configuredTargetValue instanceof RuleConfiguredTargetValue)) {
          // We have to include non-rule values in the graph to visit their dependencies, but they
          // don't have any actions to print out.
          continue;
        }
        actionGraphDump.dumpConfiguredTarget((RuleConfiguredTargetValue) configuredTargetValue);
        if (options.useAspects) {
          for (AspectValue aspectValue : accessor.getAspectValues(configuredTargetValue)) {
            actionGraphDump.dumpAspect(aspectValue, configuredTargetValue);
          }
        }
      }
    } catch (CommandLineExpansionException e) {
      throw new IOException(e.getMessage());
    }
  }

  @Override
  public void close(boolean failFast) throws IOException {
    if (!failFast) {
      aqueryOutputHandler.close();
    }
  }
}
