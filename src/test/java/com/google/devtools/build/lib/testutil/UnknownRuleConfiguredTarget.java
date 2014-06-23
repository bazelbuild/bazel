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

package com.google.devtools.build.lib.testutil;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder.StatelessRunfilesProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;

/**
 * A null implementation of ConfiguredTarget for rules we don't know how to build.
 */
public class UnknownRuleConfiguredTarget implements RuleConfiguredTargetFactory {

  @Override
  public RuleConfiguredTarget create(RuleContext context)  {
    // TODO(blaze-team): (2009) why isn't this an error?  It would stop the build more promptly...
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
    context.getAnalysisEnvironment().registerAction(new FailAction(context.getActionOwner(),
        filesToBuild, "cannot build " + rule.getRuleClass() + " rules such as " + rule.getLabel()));
    return new GenericRuleConfiguredTargetBuilder(context)
        .setFilesToBuild(filesToBuild)
        .add(RunfilesProvider.class, new StatelessRunfilesProvider(Runfiles.EMPTY))
        .build();
  }
}
