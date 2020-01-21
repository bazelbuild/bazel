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
import com.google.common.collect.Maps;
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
import java.util.Collection;
import java.util.List;
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

    ImmutableList<Artifact> srcs = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();
    List<Path> childNinjaFiles = srcs.stream().map(Artifact::getPath).collect(Collectors.toList());

    // TODO(ichern): add check against non-symlinked roots.
    String outputRoot = ruleContext.attributes().get("output_root", Type.STRING);
    Preconditions.checkNotNull(outputRoot);

    String workingDirectory = ruleContext.attributes().get("working_directory", Type.STRING);
    Preconditions.checkNotNull(workingDirectory);

    List<String> outputRootInputs = ruleContext.attributes()
        .get("output_root_inputs", Type.STRING_LIST);
    Preconditions.checkNotNull(outputRootInputs);

    ImmutableList<Artifact> symlinkedArtifacts = createSymlinksForOutputRoot(
        ruleContext,
        mainArtifact.getRoot().getRoot(),
        outputRoot, outputRootInputs);

    Path workspace = Preconditions.checkNotNull(ruleContext.getConfiguration())
        .getDirectories().getWorkspace();
    String ownerTargetName = ruleContext.getLabel().getName();
    NinjaGraphProvider graphProvider = createGraphProvider(mainArtifact, childNinjaFiles,
        workingDirectory, outputRoot, workspace, ownerTargetName, symlinkedArtifacts);

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    builder.setFilesToBuild(NestedSetBuilder.wrap(Order.STABLE_ORDER, symlinkedArtifacts));
    builder.addProvider(NinjaGraphProvider.class, graphProvider)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY);

    return builder.build();
  }

  private ImmutableList<Artifact> createSymlinksForOutputRoot(
      RuleContext ruleContext,
      Root sourceRoot,
      String outputRoot,
      List<String> outputRootInputs) {
    if (outputRootInputs.isEmpty()) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    PathFragment outputRootPath = PathFragment.create(outputRoot);
    Path execRoot = Preconditions.checkNotNull(ruleContext.getConfiguration()).getDirectories()
        .getExecRoot(ruleContext.getWorkspaceName());
    ArtifactRoot derivedRoot = ArtifactRoot.asDerivedRoot(execRoot, outputRootPath);

    for (String input : outputRootInputs) {
      DerivedArtifact derivedArtifact = ruleContext
          .getDerivedArtifact(PathFragment.create(input), derivedRoot);
      builder.add(derivedArtifact);

      SymlinkAction symlinkAction = SymlinkAction.toAbsolutePath(ruleContext.getActionOwner(),
          sourceRoot.asPath().getRelative(outputRootPath).getRelative(PathFragment.create(input)).asFragment(), derivedArtifact,
          String.format("Symlinking %s under <execroot>/%s", input, outputRoot));
      ruleContext.getAnalysisEnvironment().registerAction(symlinkAction);
    }
    return builder.build();
 }

  private static NinjaGraphProvider createGraphProvider(
      Artifact mainArtifact,
      Collection<Path> childNinjaFiles,
      String workingDirectory,
      String outputRoot,
      Path workspace,
      String ownerTargetName,
      ImmutableList<Artifact> symlinkedArtifacts) throws InterruptedException, RuleErrorException {
    ThreadFactory threadFactory = new ThreadFactoryBuilder()
        .setNameFormat(NinjaGraphRuleConfiguredTargetFactory.class.getSimpleName() + "-%d")
        .build();
    int numThreads = Math.min(25, Runtime.getRuntime().availableProcessors() - 1);
    ListeningExecutorService service =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads, threadFactory));
    try {
      NinjaPipeline pipeline = new NinjaPipeline(workspace.getRelative(workingDirectory), service,
          childNinjaFiles, ownerTargetName);
      Pair<NinjaScope, List<NinjaTarget>> pipelineResult =
          pipeline.pipeline(mainArtifact.getPath());
      TreeMap<PathFragment, NinjaTarget> targetsMap = Maps.newTreeMap();
      NinjaTarget previous;
      for (NinjaTarget target : Preconditions.checkNotNull(pipelineResult.getSecond())) {
        for (PathFragment fragment : target.getOutputs()) {
          if ((previous = targetsMap.put(fragment, target)) != null) {
            throw new RuleErrorException(
                String.format("Two Ninja build statements are producing the same output '%s':\n"
                    + "%s\nand\n%s\n", fragment.getPathString(), previous.prettyPrint(),
                    target.prettyPrint()));
          }
        }
      }
      return new NinjaGraphProvider(outputRoot, workingDirectory, pipelineResult.getFirst(),
          ImmutableSortedMap.copyOf(targetsMap), symlinkedArtifacts);
    } catch (GenericParsingException | IOException e) {
      throw new RuleErrorException(e.getMessage());
    } finally {
      ExecutorUtil.interruptibleShutdown(service);
    }
  }
}
