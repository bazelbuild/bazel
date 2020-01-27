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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.extra.ExtraActionMapProvider;
import com.google.devtools.build.lib.analysis.extra.ExtraActionSpec;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import java.util.List;
import java.util.Set;

/**
 * A collection of static methods related to ExtraActions.
 */
class ExtraActionUtils {
  /**
   * Scans {@code action_listeners} associated with this build to see if any
   * {@code extra_actions} should be added to this configured target. If any
   * action_listeners are present, a partial visit of the artifact/action graph
   * is performed (for as long as actions found are owned by this {@link
   * ConfiguredTarget}). Any actions that match the {@code action_listener}
   * get an {@code extra_action} associated. The output artifacts of the
   * extra_action are reported to the {@link AnalysisEnvironment} for
   * bookkeeping.
   */
  static ExtraActionArtifactsProvider createExtraActionProvider(
      Set<ActionAnalysisMetadata> actionsWithoutExtraAction, RuleContext ruleContext) {
    BuildConfiguration configuration = ruleContext.getConfiguration();
    if (configuration.isToolConfiguration()) {
      return ExtraActionArtifactsProvider.EMPTY;
    }

    ImmutableList<Artifact.DerivedArtifact> extraActionArtifacts = ImmutableList.of();
    NestedSetBuilder<Artifact.DerivedArtifact> builder = NestedSetBuilder.stableOrder();

    List<Label> actionListenerLabels = configuration.getActionListeners();
    if (!actionListenerLabels.isEmpty()
        && ruleContext.attributes().getAttributeDefinition(":action_listener") != null) {
      ExtraActionsVisitor visitor =
          new ExtraActionsVisitor(ruleContext, computeMnemonicsToExtraActionMap(ruleContext));

      // The action list is modified within the body of the loop by the maybeAddExtraAction() call,
      // thus the copy
      for (ActionAnalysisMetadata action :
          ImmutableList.copyOf(ruleContext.getAnalysisEnvironment().getRegisteredActions())) {
        if (!actionsWithoutExtraAction.contains(action)) {
          visitor.maybeAddExtraAction(action);
        }
      }

      extraActionArtifacts = visitor.getAndResetExtraArtifacts();
      if (!extraActionArtifacts.isEmpty()) {
        builder.addAll(extraActionArtifacts);
      }
    }

    // Add extra action artifacts from dependencies
    for (ExtraActionArtifactsProvider provider : AnalysisUtils.getProviders(
        ruleContext.getConfiguredTargetMap().values(), ExtraActionArtifactsProvider.class)) {
      builder.addTransitive(provider.getTransitiveExtraActionArtifacts());
    }

    return ExtraActionArtifactsProvider.create(
        NestedSetBuilder.<Artifact.DerivedArtifact>stableOrder()
            .addAll(extraActionArtifacts)
            .build(),
        builder.build());
  }

  /**
   * Populates the configuration specific mnemonicToExtraActionMap
   * based on all action_listers selected by the user (via the blaze option
   * {@code --experimental_action_listener=<target>}).
   */
  private static Multimap<String, ExtraActionSpec> computeMnemonicsToExtraActionMap(
      RuleContext ruleContext) {
    // We copy the multimap here every time. This could be expensive.
    Multimap<String, ExtraActionSpec> mnemonicToExtraActionMap = HashMultimap.create();
    for (TransitiveInfoCollection actionListener :
        ruleContext.getPrerequisites(":action_listener", Mode.HOST)) {
      ExtraActionMapProvider provider = actionListener.getProvider(ExtraActionMapProvider.class);
      if (provider == null) {
        ruleContext.ruleError(String.format(
            "Unable to match experimental_action_listeners to this rule. "
            + "Specified target %s is not an action_listener rule",
            actionListener.getLabel().toString()));
      } else {
        mnemonicToExtraActionMap.putAll(provider.getExtraActionMap());
      }
    }
    return mnemonicToExtraActionMap;
  }
}
