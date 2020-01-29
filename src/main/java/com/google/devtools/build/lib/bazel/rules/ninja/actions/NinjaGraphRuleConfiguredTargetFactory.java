// Copyright 2019 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.pipeline.NinjaPipeline;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.repository.ExternalPackageUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.stream.Collectors;

/**
 * Configured target factory for {@link NinjaGraphRule}.
 */
public class NinjaGraphRuleConfiguredTargetFactory implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    if (!ruleContext.getAnalysisEnvironment().getSkylarkSemantics().experimentalNinjaActions()) {
      throw new RuleErrorException(""
          + "Usage of ninja_graph is only allowed with --experimental_ninja_actions flag");
    }
    Artifact mainArtifact = ruleContext.getPrerequisiteArtifact("main", Mode.TARGET);

    ImmutableList<Artifact> ninjaSrcs = ruleContext.getPrerequisiteArtifacts("ninja_srcs", Mode.TARGET).list();
    List<Path> childNinjaFiles = ninjaSrcs.stream().map(Artifact::getPath).collect(Collectors.toList());

    String outputRoot = ruleContext.attributes().get("output_root", Type.STRING);
    Preconditions.checkNotNull(outputRoot);

    String workingDirectory = ruleContext.attributes().get("working_directory", Type.STRING);
    Preconditions.checkNotNull(workingDirectory);

    List<String> outputRootInputs = ruleContext.attributes()
        .get("output_root_inputs", Type.STRING_LIST);
    Preconditions.checkNotNull(outputRootInputs);

    Path workspace = Preconditions.checkNotNull(ruleContext.getConfiguration())
        .getDirectories().getWorkspace();
    String ownerTargetName = ruleContext.getLabel().getName();
    List<NinjaTarget> ninjaTargets = readNinjaGraph(mainArtifact,
        childNinjaFiles, workingDirectory, workspace, ownerTargetName);

    ImmutableSortedMap.Builder<PathFragment, NinjaTarget> usualTargetsBuilder =
        ImmutableSortedMap.naturalOrder();
    ImmutableSortedMap.Builder<PathFragment, NinjaTarget> phonyTargetsBuilder =
        ImmutableSortedMap.naturalOrder();
    separatePhonyTargets(ninjaTargets, usualTargetsBuilder, phonyTargetsBuilder);
    ImmutableSortedMap<PathFragment, NinjaTarget> usualTargets = usualTargetsBuilder.build();
    ImmutableSortedMap<PathFragment, NestedSet<PathFragment>> phonyTargetsMap;

    Root sourceRoot = mainArtifact.getRoot().getRoot();
    NestedSet<Artifact> filesToBuild;
    try {
      ImmutableSortedMap<PathFragment, NinjaTarget> phonyTargets = phonyTargetsBuilder.build();
      phonyTargetsMap = new NinjaPhonyTargetsUtil(phonyTargets).getPhonyPathsMap();
      NinjaActionsFactory ninjaActionsFactory = new NinjaActionsFactory(ruleContext, sourceRoot,
          PathFragment.create(outputRoot),
          PathFragment.create(workingDirectory), outputRootInputs, usualTargets, phonyTargetsMap);
      ninjaActionsFactory.process();
      filesToBuild = ninjaActionsFactory.getFilesToBuild();
    } catch (GenericParsingException e) {
      throw new RuleErrorException(e.getMessage());
    }

    NinjaGraphProvider graphProvider = new NinjaGraphProvider(outputRoot, workingDirectory,
        phonyTargetsMap);

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    builder.setFilesToBuild(filesToBuild);
    builder.addProvider(NinjaGraphProvider.class, graphProvider)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY);

    return builder.build();
  }

  private void separatePhonyTargets(List<NinjaTarget> ninjaTargets,
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> usualTargetsBuilder,
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> phonyTargetsBuilder)
      throws RuleErrorException {
    for (NinjaTarget target : ninjaTargets) {
      if ("phony".equals(target.getRuleName())) {
        if (target.getAllOutputs().size() != 1) {
          String allOutputs = target.getAllOutputs().stream()
              .map(PathFragment::getPathString).collect(Collectors.joining(" "));
          throw new RuleErrorException(
              String.format("Ninja phony alias can only be used for single output, but found '%s'.",
              allOutputs));
        }
        phonyTargetsBuilder.put(Iterables.getOnlyElement(target.getAllOutputs()), target);
      } else {
        for (PathFragment output : target.getAllOutputs()) {
          usualTargetsBuilder.put(output, target);
        }
      }
    }
  }

  private static List<NinjaTarget> readNinjaGraph(
      Artifact mainArtifact,
      Collection<Path> childNinjaFiles,
      String workingDirectory,
      Path workspace,
      String ownerTargetName) throws InterruptedException, RuleErrorException {
    ThreadFactory threadFactory = new ThreadFactoryBuilder()
        .setNameFormat(NinjaGraphRuleConfiguredTargetFactory.class.getSimpleName() + "-%d")
        .build();
    int numThreads = Math.min(25, Runtime.getRuntime().availableProcessors() - 1);
    ListeningExecutorService service =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads, threadFactory));
    try {
      NinjaPipeline pipeline = new NinjaPipeline(workspace.getRelative(workingDirectory), service,
          childNinjaFiles, ownerTargetName);
      return pipeline.pipeline(mainArtifact.getPath());
    } catch (GenericParsingException | IOException e) {
      throw new RuleErrorException(e.getMessage());
    } finally {
      ExecutorUtil.interruptibleShutdown(service);
    }
  }
}
