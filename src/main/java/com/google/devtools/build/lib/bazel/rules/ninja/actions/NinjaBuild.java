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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor.VisitedState;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

/** Configured target factory for {@link NinjaBuildRule}. */
public class NinjaBuild implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    Map<String, List<String>> outputGroupsMap =
        ruleContext.attributes().get("output_groups", Type.STRING_LIST_DICT);
    NinjaGraphProvider graphProvider =
        ruleContext.getPrerequisite("ninja_graph", TransitionMode.TARGET, NinjaGraphProvider.class);
    Preconditions.checkNotNull(graphProvider);
    List<PathFragment> pathsToBuild =
        outputGroupsMap.values().stream()
            .flatMap(List::stream)
            .map(PathFragment::create)
            .collect(Collectors.toList());
    ImmutableSortedMap.Builder<PathFragment, Artifact> depsMapBuilder =
        ImmutableSortedMap.naturalOrder();
    ImmutableSortedMap.Builder<PathFragment, Artifact> symlinksMapBuilder =
        ImmutableSortedMap.naturalOrder();
    createDepsMap(
        ruleContext, graphProvider.getWorkingDirectory(), depsMapBuilder, symlinksMapBuilder);

    ImmutableSortedMap<PathFragment, Artifact> depsMap = depsMapBuilder.build();

    NinjaGraphArtifactsHelper artifactsHelper =
        new NinjaGraphArtifactsHelper(
            ruleContext,
            graphProvider.getOutputRoot(),
            graphProvider.getWorkingDirectory(),
            symlinksMapBuilder.build(),
            graphProvider.getOutputRootSymlinks());
    if (ruleContext.hasErrors()) {
      return null;
    }

    try {
      symlinkDepsMappings(ruleContext, artifactsHelper, depsMap);

      PhonyTargetArtifacts phonyTargetArtifacts =
          new PhonyTargetArtifacts(graphProvider.getPhonyTargetsMap(), artifactsHelper);
      ImmutableSet<PathFragment> symlinks =
          ImmutableSet.<PathFragment>builder()
              .addAll(graphProvider.getOutputRootInputsSymlinks())
              .addAll(depsMap.keySet())
              .build();

      new NinjaActionsHelper(
              ruleContext,
              artifactsHelper,
              graphProvider.getTargetsMap(),
              graphProvider.getPhonyTargetsMap(),
              phonyTargetArtifacts,
              pathsToBuild,
              symlinks)
          .createNinjaActions();

      if (!checkOrphanArtifacts(ruleContext)) {
        return null;
      }

      NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
      TreeMap<String, NestedSet<Artifact>> groups = Maps.newTreeMap();
      for (Map.Entry<String, List<String>> entry : outputGroupsMap.entrySet()) {
        NestedSet<Artifact> artifacts =
            getGroupArtifacts(
                ruleContext,
                entry.getValue(),
                graphProvider.getPhonyTargetsMap(),
                phonyTargetArtifacts,
                artifactsHelper);
        groups.put(entry.getKey(), artifacts);
        filesToBuild.addTransitive(artifacts);
      }

      if (ruleContext.hasErrors()) {
        return null;
      }

      return new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
          .setFilesToBuild(filesToBuild.build())
          .addOutputGroups(groups)
          .build();
    } catch (GenericParsingException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    }
  }

  private static void symlinkDepsMappings(
      RuleContext ruleContext,
      NinjaGraphArtifactsHelper artifactsHelper,
      ImmutableSortedMap<PathFragment, Artifact> depsMap)
      throws GenericParsingException {
    for (Map.Entry<PathFragment, Artifact> entry : depsMap.entrySet()) {
      PathFragment depPath = entry.getKey();
      Artifact destinationArtifact = entry.getValue();
      Artifact outputArtifact = artifactsHelper.createOutputArtifact(depPath);

      SymlinkAction symlinkAction =
          SymlinkAction.toArtifact(
              ruleContext.getActionOwner(),
              destinationArtifact,
              outputArtifact,
              String.format(
                  "Symlinking deps_mapping entry '%s' to '%s'",
                  destinationArtifact.getExecPath(), outputArtifact.getExecPath()));
      ruleContext.registerAction(symlinkAction);
    }
  }

  private static boolean checkOrphanArtifacts(RuleContext ruleContext) {
    ImmutableSet<Artifact> orphanArtifacts =
        ruleContext.getAnalysisEnvironment().getOrphanArtifacts();
    if (!orphanArtifacts.isEmpty()) {
      List<String> paths =
          orphanArtifacts.stream().map(Artifact::getExecPathString).collect(Collectors.toList());
      ruleContext.ruleError(
          "The following artifacts do not have a generating action in Ninja file: "
              + String.join(", ", paths));
      return false;
    }
    return true;
  }

  private static NestedSet<Artifact> getGroupArtifacts(
      RuleContext ruleContext,
      List<String> targets,
      ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap,
      PhonyTargetArtifacts phonyTargetsArtifacts,
      NinjaGraphArtifactsHelper artifactsHelper)
      throws GenericParsingException {
    NestedSetBuilder<Artifact> nestedSetBuilder = NestedSetBuilder.stableOrder();
    for (String target : targets) {
      PathFragment path = PathFragment.create(target);
      if (phonyTargetsMap.containsKey(path)) {
        NestedSet<Artifact> artifacts = phonyTargetsArtifacts.getPhonyTargetArtifacts(path);
        nestedSetBuilder.addTransitive(artifacts);
      } else {
        Artifact outputArtifact = artifactsHelper.createOutputArtifact(path);
        if (outputArtifact == null) {
          ruleContext.ruleError(
              String.format("Required target '%s' is not created in ninja_graph.", path));
          return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
        }
        nestedSetBuilder.add(outputArtifact);
      }
    }
    return nestedSetBuilder.build();
  }

  private static void createDepsMap(
      RuleContext ruleContext,
      PathFragment workingDirectory,
      ImmutableSortedMap.Builder<PathFragment, Artifact> depsMapBuilder,
      ImmutableSortedMap.Builder<PathFragment, Artifact> symlinksMapBuilder)
      throws InterruptedException {
    FileProvider fileProvider =
        ruleContext.getPrerequisite("ninja_graph", TransitionMode.TARGET, FileProvider.class);
    Preconditions.checkNotNull(fileProvider);
    new NestedSetVisitor<Artifact>(
            a -> {
              symlinksMapBuilder.put(a.getExecPath().relativeTo(workingDirectory), a);
            },
            new VisitedState<>())
        .visit(fileProvider.getFilesToBuild());

    Map<String, TransitiveInfoCollection> mapping = ruleContext.getPrerequisiteMap("deps_mapping");
    for (Map.Entry<String, TransitiveInfoCollection> entry : mapping.entrySet()) {
      NestedSet<Artifact> filesToBuild =
          entry.getValue().getProvider(FileProvider.class).getFilesToBuild();
      if (!filesToBuild.isSingleton()) {
        ruleContext.attributeError(
            "deps_mapping",
            String.format(
                "'%s' contains more than one output. "
                    + "deps_mapping should only contain targets, producing a single output file.",
                entry.getValue().getLabel().getCanonicalForm()));
        return;
      }
      depsMapBuilder.put(PathFragment.create(entry.getKey()), filesToBuild.getSingleton());
    }
  }
}
