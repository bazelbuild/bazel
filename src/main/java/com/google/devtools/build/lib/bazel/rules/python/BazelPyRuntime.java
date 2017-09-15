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

package com.google.devtools.build.lib.bazel.rules.python;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Implementation for the {@code py_runtime} rule.
 */
public final class BazelPyRuntime implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    NestedSet<Artifact> files =
        PrerequisiteArtifacts.nestedSet(ruleContext, "files", Mode.TARGET);
    Artifact interpreter =
        ruleContext.getPrerequisiteArtifact("interpreter", Mode.TARGET);
    String interpreterPath =
        ruleContext.attributes().get("interpreter_path", Type.STRING);

    NestedSet<Artifact> all = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(files)
        .build();

    if (interpreter != null && !interpreterPath.isEmpty()) {
      ruleContext.ruleError("interpreter and interpreter_path cannot be set at the same time.");
    }

    if (interpreter == null && interpreterPath.isEmpty()) {
      ruleContext.ruleError("interpreter and interpreter_path cannot be empty at the same time.");
    }

    if (!interpreterPath.isEmpty() && !PathFragment.create(interpreterPath).isAbsolute()) {
      ruleContext.attributeError("interpreter_path", "must be an absolute path.");
    }

    if (!interpreterPath.isEmpty() && !files.isEmpty()) {
      ruleContext.ruleError("interpreter with an absolute path requires files to be empty.");
    }

    if (ruleContext.hasErrors()) {
      return null;
    }

    BazelPyRuntimeProvider provider = BazelPyRuntimeProvider
        .create(files, interpreter, interpreterPath);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .setFilesToBuild(all)
        .addProvider(BazelPyRuntimeProvider.class, provider)
        .build();
  }

}
