// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.sh;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Implementation for sh_test rules.
 */
public class ShTest extends ShBinary implements RuleConfiguredTargetFactory {

  @Override
  protected Artifact getExecutableScript(RuleContext ruleContext, Artifact src) {
    if (ruleContext.attributes().get("bash_version", Type.STRING)
        .equals(BazelShRuleClasses.SYSTEM_BASH_VERSION)) {
      return src;
    }

    // What *will* this script run with the wrapper?
    PathFragment newOutput = src.getRootRelativePath().getParentDirectory().getRelative(
        ruleContext.getLabel().getName() + "_runner.sh");
    Artifact testRunner = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
        newOutput, ruleContext.getConfiguration().getBinDirectory());

    String bashPath = BazelShRuleClasses.BASH_BINARY_BINDINGS
        .get(BazelShRuleClasses.SYSTEM_BASH_VERSION).execPath;

    // Generate the runner contents.
    String runnerContents =
        "#!/bin/bash\n"
        + bashPath + " \"" + src.getRootRelativePath().getPathString() + "\" \"$@\"\n";

    ruleContext.registerAction(
        new FileWriteAction(ruleContext.getActionOwner(), testRunner, runnerContents, true));
    return testRunner;
  }
}
