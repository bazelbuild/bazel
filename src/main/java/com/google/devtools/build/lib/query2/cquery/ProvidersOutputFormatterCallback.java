// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.OutputStream;

/** Provider name output formatter for cquery results */
public class ProvidersOutputFormatterCallback extends CqueryThreadsafeCallback {

  ProvidersOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options, OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
  }

  @Override
  public String getName() {
    return "providers";
  }

  @Override
  public void processOutput(Iterable<ConfiguredTarget> partialResult) {
    for (ConfiguredTarget configuredTarget : partialResult) {
      if (configuredTarget instanceof RuleConfiguredTarget) {
        RuleConfiguredTarget ruleConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
        StringBuilder output = new StringBuilder();
        output
            .append(configuredTarget.getLabel())
            .append(" ")
            .append(
                ruleConfiguredTarget.getStarlarkProviderKeyStrings(/*showOutputGroupInfo=*/true));
        addResult(output.toString());
      }
    }
  }
}
