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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder.StatelessRunfilesProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.RunfilesSupport;
import com.google.devtools.build.lib.view.actions.ExecutableSymlinkAction;

import java.util.Collection;

/**
 * Implementation for the sh_binary rule.
 */
public class ShBinary implements RuleConfiguredTargetFactory {

  @Override
  public RuleConfiguredTarget create(RuleContext ruleContext) {
    ImmutableList<Artifact> srcs = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET);
    if (srcs.size() != 1) {
      ruleContext.attributeError("srcs", "you must specify exactly one file in 'srcs'");
      return null;
    }

    Artifact symlink = ruleContext.createOutputArtifact();
    Artifact src = srcs.get(0);
    Artifact executableScript = getExecutableScript(ruleContext, src);
    // The interpretation of this deceptively simple yet incredibly generic rule is complicated
    // by the distinction between targets and (not properly encapsulated) artifacts. It depends
    // on the notion of other rule's "files-to-build" sets, which are undocumented, making it
    // impossible to give a precise definition of what this rule does in all cases (e.g. what
    // happens when srcs = ['x', 'y'] but 'x' is an empty filegroup?). This is a pervasive
    // problem in Blaze.
    ruleContext.registerAction(
        new ExecutableSymlinkAction(ruleContext.getActionOwner(), executableScript, symlink));

    NestedSet<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .add(src)
        .add(executableScript) // May be the same as src, in which case set semantics apply.
        .add(symlink)
        .build();
    Runfiles runfiles = new Runfiles.Builder()
        .addArtifacts(filesToBuild)
        .build();
    RunfilesSupport runfilesSupport = RunfilesSupport.withExecutable(
        ruleContext, runfiles, symlink);
    return new GenericRuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .setRunfilesSupport(runfilesSupport)
        .addProvider(RunfilesProvider.class, new StatelessRunfilesProvider(runfiles))
        .build();
  }

  /**
   * Hook for sh_test to provide the executable.
   *
   * @param ruleContext
   * @param src
   */
  protected Artifact getExecutableScript(RuleContext ruleContext, Artifact src) {
    return src;
  }

  /**
   * Checks that the given string attribute has a valid value. Returns true if valid
   * or undefined, false otherwise.
   */
  private boolean validateStringInput(RuleContext ruleContext, String attribute,
      Collection<String> validValues) {
    if (ruleContext.getRule().isAttrDefined(attribute, Type.STRING)) {
      String s = ruleContext.attributes().get(attribute, Type.STRING);
      if (!validValues.contains(s)) {
        ruleContext.attributeError(attribute, "invalid '" + attribute + "' value: " + s);
        return false;
      }
    }
    return true;
  }
}
