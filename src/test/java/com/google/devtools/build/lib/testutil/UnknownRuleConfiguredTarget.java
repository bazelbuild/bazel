// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.testutil;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.server.FailureDetails.FailAction.Code;
import javax.annotation.Nullable;

/**
 * A null implementation of ConfiguredTarget for rules we don't know how to build.
 */
public class UnknownRuleConfiguredTarget implements RuleConfiguredTargetFactory {

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext context)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    // TODO(bazel-team): (2009) why isn't this an error?  It would stop the build more promptly...
    context.ruleWarning("cannot build " + context.getRule().getRuleClass() + " rules");

    ImmutableList<Artifact> outputArtifacts = context.getOutputArtifacts();
    NestedSet<Artifact> filesToBuild;
    if (outputArtifacts.isEmpty()) {
      // Gotta build *something*...
      filesToBuild = NestedSetBuilder.create(Order.STABLE_ORDER,
          context.createOutputArtifact());
    } else {
      filesToBuild = NestedSetBuilder.wrap(Order.STABLE_ORDER, outputArtifacts);
    }

    Rule rule = context.getRule();
    context.registerAction(
        new FailAction(
            context.getActionOwner(),
            filesToBuild.toList(),
            "cannot build " + rule.getRuleClass() + " rules such as " + rule.getLabel(),
            Code.FAIL_ACTION_UNKNOWN));
    return new RuleConfiguredTargetBuilder(context)
        .setFilesToBuild(filesToBuild)
        .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
