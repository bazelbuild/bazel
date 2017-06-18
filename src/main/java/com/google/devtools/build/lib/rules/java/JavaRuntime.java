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

import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.CompilationHelper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;

/** Implementation for the {@code java_runtime} rule. */
public class JavaRuntime implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    NestedSet<Artifact> filesToBuild =
        PrerequisiteArtifacts.nestedSet(ruleContext, "srcs", Mode.TARGET);
    NestedSet<Artifact> middleman =
        CompilationHelper.getAggregatingMiddleman(
            ruleContext, Actions.escapeLabel(ruleContext.getLabel()), filesToBuild);
    // TODO(cushon): clean up uses of java_runtime in data deps and remove this
    Runfiles runfiles =
        new Runfiles.Builder(ruleContext.getWorkspaceName())
            .addTransitiveArtifacts(filesToBuild)
            .build();
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(runfiles))
        .setFilesToBuild(filesToBuild)
        .addProvider(JavaRuntimeProvider.class, JavaRuntimeProvider.create(filesToBuild))
        .addProvider(MiddlemanProvider.class, new MiddlemanProvider(middleman))
        .build();
  }
}
