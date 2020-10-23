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
package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.OutputStream;

/** Default Output callback for cquery. Prints a label and configuration pair per result. */
public class LabelAndConfigurationOutputFormatterCallback extends CqueryThreadsafeCallback {
  private final boolean showKind;

  public LabelAndConfigurationOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor,
      boolean showKind) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
    this.showKind = showKind;
  }

  @Override
  public String getName() {
    return this.showKind ? "label_kind" : "label";
  }

  @Override
  public void processOutput(Iterable<ConfiguredTarget> partialResult) {
    for (ConfiguredTarget configuredTarget : partialResult) {
      StringBuilder output = new StringBuilder();
      if (showKind) {
        Target actualTarget = accessor.getTargetFromConfiguredTarget(configuredTarget);
        output = output.append(actualTarget.getTargetKind()).append(" ");
      }
      output =
          output
              .append(configuredTarget.getOriginalLabel())
              .append(" (")
              .append(shortId(getConfiguration(configuredTarget.getConfigurationKey())))
              .append(")");

      if (options.showRequiredConfigFragments != IncludeConfigFragmentsEnum.OFF) {
        RequiredConfigFragmentsProvider configFragmentsProvider =
            configuredTarget.getProvider(RequiredConfigFragmentsProvider.class);
        String requiredFragmentsOutput =
            configFragmentsProvider != null
                ? String.join(", ", configFragmentsProvider.getRequiredConfigFragments())
                : "";
        output.append(" [").append(requiredFragmentsOutput).append("]");
      }

      addResult(output.toString());
    }
  }
}
