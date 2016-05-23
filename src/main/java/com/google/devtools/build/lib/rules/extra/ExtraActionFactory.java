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

package com.google.devtools.build.lib.rules.extra;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MakeVariableExpander;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;

import java.util.List;

/**
 * Factory for 'extra_action'.
 */
public final class ExtraActionFactory implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext context) throws RuleErrorException {
    // This rule doesn't produce any output when listed as a build target.
    // Only when used via the --experimental_action_listener flag,
    // this rule instructs the build system to add additional outputs.
    List<Artifact> resolvedData = Lists.newArrayList();

    Iterable<? extends TransitiveInfoCollection> tools =
        context.getPrerequisites("tools", Mode.HOST);
    CommandHelper commandHelper =
        new CommandHelper(context, tools, ImmutableMap.<Label, Iterable<Artifact>>of());

    resolvedData.addAll(context.getPrerequisiteArtifacts("data", Mode.DATA).list());
    List<String>outputTemplates =
        context.attributes().get("out_templates", Type.STRING_LIST);

    String command = commandHelper.resolveCommandAndExpandLabels(false, true);
    // This is a bit of a hack. We want to run the MakeVariableExpander first, so we expand $ on
    // variables that are expanded below with $$, which gets reverted to $ by the
    // MakeVariableExpander. This allows us to expand package-specific make variables in the
    // package where the extra action is defined, and then later replace the owner-specific make
    // variables when the extra action is instantiated.
    command = command.replace("$(EXTRA_ACTION_FILE)", "$$(EXTRA_ACTION_FILE)");
    command = command.replace("$(ACTION_ID)", "$$(ACTION_ID)");
    command = command.replace("$(OWNER_LABEL_DIGEST)", "$$(OWNER_LABEL_DIGEST)");
    command = command.replace("$(output ", "$$(output ");
    try {
      command = MakeVariableExpander.expand(
          command, new ConfigurationMakeVariableContext(
              context.getTarget().getPackage(), context.getConfiguration()));
    } catch (MakeVariableExpander.ExpansionException e) {
      context.ruleError(String.format("Unable to expand make variables: %s",
          e.getMessage()));
    }

    boolean requiresActionOutput =
        context.attributes().get("requires_action_output", Type.BOOLEAN);

    ExtraActionSpec spec = new ExtraActionSpec(
        commandHelper.getResolvedTools(),
        commandHelper.getRemoteRunfileManifestMap(),
        resolvedData,
        outputTemplates,
        command,
        context.getLabel(),
        TargetUtils.getExecutionInfo(context.getRule()),
        requiresActionOutput);
    return new RuleConfiguredTargetBuilder(context)
        .addProvider(ExtraActionSpec.class, spec)
        .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
