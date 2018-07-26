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
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.output.AqueryOptions;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.actiongraph.ActionGraphDump;
import java.io.IOException;
import java.io.OutputStream;

/** Default output callback for aquery, prints proto output. */
public class ActionGraphProtoOutputFormatterCallback extends AqueryThreadsafeCallback {

  final ActionGraphDump actionGraphDump;

  ActionGraphProtoOutputFormatterCallback(
      Reporter reporter,
      AqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTargetValue> accessor) {
    super(reporter, options, out, skyframeExecutor, accessor);
    // TODO(twerth): Allow users to include action command lines.
    actionGraphDump = new ActionGraphDump(/* includeActionCmdLine */ false);
  }

  @Override
  public String getName() {
    return "proto";
  }

  @Override
  public void processOutput(Iterable<ConfiguredTargetValue> partialResult) throws IOException {
    try {
      for (ConfiguredTargetValue configuredTargetValue : partialResult) {
        actionGraphDump.dumpConfiguredTarget(configuredTargetValue);
      }
    } catch (CommandLineExpansionException e) {
      throw new IOException(e.getMessage());
    }
  }

  @Override
  public void close(boolean failFast) throws IOException {
    if (!failFast && printStream != null) {
      ActionGraphContainer actionGraphContainer = actionGraphDump.build();
      actionGraphContainer.writeTo(printStream);
    }
  }

  @VisibleForTesting
  public AnalysisProtos.ActionGraphContainer getProtoResult() {
    return actionGraphDump.build();
  }
}
