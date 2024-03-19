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

import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.extra.ExtraActionMapProvider;
import com.google.devtools.build.lib.analysis.extra.ExtraActionSpec;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import com.google.devtools.build.lib.packages.Types;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Implementation for the 'action_listener' rule.
 */
public final class ActionListener implements RuleConfiguredTargetFactory {
  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    // This rule doesn't produce any output when listed as a build target.
    // Only when used via the --experimental_action_listener flag,
    // this rule instructs the build system to add additional outputs.

    List<ExtraActionSpec> extraActions;

    Multimap<String, ExtraActionSpec> extraActionMap;

    Set<String> mnemonics =
        Sets.newHashSet(ruleContext.attributes().get("mnemonics", Types.STRING_LIST));
    extraActions = retrieveAndValidateExtraActions(ruleContext);
    ImmutableSortedKeyListMultimap.Builder<String, ExtraActionSpec>
        extraActionMapBuilder = ImmutableSortedKeyListMultimap.builder();
    for (String mnemonic : mnemonics) {
      extraActionMapBuilder.putAll(mnemonic, extraActions);
    }
    extraActionMap = extraActionMapBuilder.build();
    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .add(ExtraActionMapProvider.class, new ExtraActionMapProvider(extraActionMap))
        .build();
  }

  /**
   * Loads the targets listed in the 'extra_actions' attribute of this rule.
   * Validates these targets to be extra_actions indeed. And checks if the
   * blaze version number is in the range of the blaze_version restrictions on the rule.
   */
  private List<ExtraActionSpec> retrieveAndValidateExtraActions(RuleContext ruleContext) {
    List<ExtraActionSpec> extraActions = new ArrayList<>();
    for (TransitiveInfoCollection prerequisite : ruleContext.getPrerequisites("extra_actions")) {
      ExtraActionSpec spec = prerequisite.getProvider(ExtraActionSpec.class);
      if (spec == null) {
        ruleContext.attributeError("extra_actions", String.format("target %s is not an "
            + "extra_action rule", prerequisite.getLabel().toString()));
      } else {
        extraActions.add(spec);
      }
    }
    if (extraActions.isEmpty()) {
      ruleContext.attributeWarning(
          "extra_actions", "No extra_action is specified for this version of bazel.");
    }
    return extraActions;
  }
}
