// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.rules.MakeVariableProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/** Implementation for the {@code java_runtime_suite} rule. */
public class JavaRuntimeSuite implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    TransitiveInfoCollection runtime =
        ruleContext.getPrerequisiteMap("runtimes").get(ruleContext.getConfiguration().getCpu());
    if (runtime == null) {
      runtime = ruleContext.getPrerequisite("default", Mode.TARGET);
    }

    if (runtime == null) {
      ruleContext.throwWithRuleError(
          "could not resolve runtime for cpu " + ruleContext.getConfiguration().getCpu());
    }

    MakeVariableProvider makeVariableProvider = runtime.getProvider(MakeVariableProvider.class);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(JavaRuntimeProvider.class, runtime.getProvider(JavaRuntimeProvider.class))
        .addNativeDeclaredProvider(runtime.get(JavaRuntimeProvider.SKYLARK_CONSTRUCTOR.getKey()))
        .addProvider(RunfilesProvider.class, runtime.getProvider(RunfilesProvider.class))
        .addProvider(MiddlemanProvider.class, runtime.getProvider(MiddlemanProvider.class))
        .addProvider(makeVariableProvider)
        .addNativeDeclaredProvider(makeVariableProvider)
        .setFilesToBuild(runtime.getProvider(FileProvider.class).getFilesToBuild())
        .build();
  }
}
