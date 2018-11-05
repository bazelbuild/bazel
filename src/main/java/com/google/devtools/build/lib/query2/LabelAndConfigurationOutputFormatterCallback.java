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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.output.CqueryOptions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.OutputStream;

/** Default Output callback for cquery. Prints a label and configuration pair per result. */
public class LabelAndConfigurationOutputFormatterCallback extends CqueryThreadsafeCallback {

  LabelAndConfigurationOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
  }

  @Override
  public String getName() {
    return "label";
  }

  @Override
  public void processOutput(Iterable<ConfiguredTarget> partialResult) {
    for (ConfiguredTarget configuredTarget : partialResult) {
      BuildConfiguration config =
          skyframeExecutor.getConfiguration(eventHandler, configuredTarget.getConfigurationKey());
      StringBuilder output =
          new StringBuilder()
              .append(configuredTarget.getLabel())
              .append(" (")
              .append(config != null && config.isHostConfiguration() ? "HOST" : config)
              .append(")");
      addResult(output.toString());
    }
  }
}