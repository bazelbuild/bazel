// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Configured target factory for {@link NinjaBuildRule}.
 *
 * Does not create any actions, just selects artifacts created by actions in ninja_graph,
 * and puts them in user-defined file groups.
 */
public class NinjaBuild implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    NinjaGraphProvider graphProvider =
        ruleContext.getPrerequisite("ninja_graph", Mode.TARGET, NinjaGraphProvider.class);
    Preconditions.checkNotNull(graphProvider);

    AttributeMap attributes = ruleContext.attributes();
    List<String> targets = attributes.get("targets", Type.STRING_LIST);
    Preconditions.checkNotNull(targets);
    Map<String, List<String>> outputGroupsMap = Preconditions.checkNotNull(attributes
        .get("output_groups", Type.STRING_LIST_DICT));

    // Include paths both from 'targets' and 'output_groups' into the files-to-build.
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
    TreeMap<String, NestedSet<Artifact>> groups = Maps.newTreeMap();
    for (Map.Entry<String, List<String>> entry : outputGroupsMap.entrySet()) {
      NestedSet<Artifact> artifacts = getArtifacts(ruleContext, entry.getValue(), graphProvider);
      groups.put(entry.getKey(), artifacts);
      filesToBuild.addTransitive(artifacts);
    }

    filesToBuild.addTransitive(getArtifacts(ruleContext, targets, graphProvider));

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild.build())
        .setRunfilesSupport(null, null)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY);
    if (!outputGroupsMap.isEmpty()) {
      builder.addOutputGroups(groups);
    }
    return builder.build();
  }

  private static NestedSet<Artifact> getArtifacts(
      RuleContext ruleContext, List<String> targets,
      NinjaGraphProvider graphProvider) throws RuleErrorException {
    NestedSetBuilder<Artifact> nestedSetBuilder = NestedSetBuilder.stableOrder();
    for (String target : targets) {
      PathFragment path = PathFragment.create(target);
      NestedSet<Artifact> artifacts = graphProvider.getPhonyArtifacts(path);
      if (artifacts != null) {
        nestedSetBuilder.addTransitive(artifacts);
      } else {
        Artifact usualArtifact = graphProvider.getUsualArtifact(path);
        if (usualArtifact == null) {
          ruleContext.throwWithRuleError(
              String.format("Required target '%s' is not created in ninja_graph.", path));
        }
        nestedSetBuilder.add(usualArtifact);
      }
    }
    return nestedSetBuilder.build();
  }
}
