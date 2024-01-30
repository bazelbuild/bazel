// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import javax.annotation.Nullable;

/**
 * Implementation for the environment rule.
 */
public class Environment implements RuleConfiguredTargetFactory {

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    // The main analysis work to do here is to simply fill in SupportedEnvironmentsProvider to
    // pass the environment itself to depending rules.
    Label label = ruleContext.getLabel();

    EnvironmentGroup group;
    try {
      group = ConstraintSemantics.getEnvironmentGroup(ruleContext.getRule());
    } catch (ConstraintSemantics.EnvironmentLookupException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    }

    EnvironmentCollection env =
        new EnvironmentCollection.Builder().put(group.getEnvironmentLabels(), label).build();
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(
            SupportedEnvironmentsProvider.class,
            SupportedEnvironments.create(env, env, ImmutableMap.of()))
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .build();
  }
}
