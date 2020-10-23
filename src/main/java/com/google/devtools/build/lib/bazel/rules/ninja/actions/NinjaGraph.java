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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.pipeline.NinjaPipelineImpl;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

/** Configured target factory for {@link NinjaGraphRule}. */
public class NinjaGraph implements RuleConfiguredTargetFactory {
  /**
   * Use a thread pool with the heuristically determined maximum number of threads, but the policy
   * to not consume threads when there are no active tasks.
   */
  private static final ThreadPoolExecutor NINJA_POOL =
      ExecutorUtil.newSlackPool(
          Math.max(2, Runtime.getRuntime().availableProcessors() / 2),
          NinjaGraph.class.getSimpleName());

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    if (!ruleContext
        .getAnalysisEnvironment()
        .getStarlarkSemantics()
        .getBool(BuildLanguageOptions.EXPERIMENTAL_NINJA_ACTIONS)) {
      throw ruleContext.throwWithRuleError(
          "Usage of ninja_graph is only allowed with --experimental_ninja_actions flag");
    }
    Artifact mainArtifact = ruleContext.getPrerequisiteArtifact("main");
    ImmutableList<Artifact> ninjaSrcs = ruleContext.getPrerequisiteArtifacts("ninja_srcs").list();
    PathFragment outputRoot =
        PathFragment.create(ruleContext.attributes().get("output_root", Type.STRING));
    PathFragment workingDirectory =
        PathFragment.create(ruleContext.attributes().get("working_directory", Type.STRING));
    List<String> outputRootInputs =
        ruleContext.attributes().get("output_root_inputs", Type.STRING_LIST);

    Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    establishDependencyOnNinjaFiles(env, mainArtifact, ninjaSrcs);
    checkDirectoriesAttributes(ruleContext, outputRoot, workingDirectory);

    if (env.valuesMissing() || ruleContext.hasErrors()) {
      return null;
    }

    Root sourceRoot = mainArtifact.getRoot().getRoot();
    NinjaGraphArtifactsHelper artifactsHelper =
        new NinjaGraphArtifactsHelper(
            ruleContext,
            outputRoot,
            workingDirectory,
            ImmutableSortedMap.of(),
            ImmutableSortedSet.of());
    if (ruleContext.hasErrors()) {
      return null;
    }

    try {
      TargetsPreparer targetsPreparer = new TargetsPreparer();
      List<Path> childNinjaFiles =
          ninjaSrcs.stream().map(Artifact::getPath).collect(Collectors.toList());
      Path workspace =
          Preconditions.checkNotNull(ruleContext.getConfiguration())
              .getDirectories()
              .getWorkspace();
      String ownerTargetName = ruleContext.getLabel().getName();
      List<NinjaTarget> ninjaTargets =
          new NinjaPipelineImpl(
                  workspace.getRelative(workingDirectory),
                  MoreExecutors.listeningDecorator(NINJA_POOL),
                  childNinjaFiles,
                  ownerTargetName)
              .pipeline(mainArtifact.getPath());
      targetsPreparer.prepareTargets(ninjaTargets);

      NestedSet<Artifact> outputRootInputsSymlinks =
          createSymlinkActions(ruleContext, sourceRoot, outputRootInputs, artifactsHelper);
      if (ruleContext.hasErrors()) {
        return null;
      }

      ImmutableSet<PathFragment> outputRootInputsSymlinksPathFragments =
          outputRootInputsSymlinks.toList().stream()
              .map(Artifact::getExecPath)
              .collect(toImmutableSet());

      NinjaGraphProvider ninjaGraphProvider =
          new NinjaGraphProvider(
              outputRoot,
              workingDirectory,
              targetsPreparer.getTargetsMap(),
              targetsPreparer.getPhonyTargetsMap(),
              targetsPreparer.getSymlinkOutputs(),
              outputRootInputsSymlinksPathFragments);

      return new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
          .addProvider(NinjaGraphProvider.class, ninjaGraphProvider)
          .setFilesToBuild(outputRootInputsSymlinks)
          .build();
    } catch (GenericParsingException | IOException e) {
      // IOException is possible with reading Ninja file, describing the action graph.
      ruleContext.ruleError(e.getMessage());
      return null;
    }
  }

  private NestedSet<Artifact> createSymlinkActions(
      RuleContext ruleContext,
      Root sourceRoot,
      List<String> outputRootInputs,
      NinjaGraphArtifactsHelper artifactsHelper)
      throws GenericParsingException {

    if (outputRootInputs.isEmpty()) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    NestedSetBuilder<Artifact> symlinks = NestedSetBuilder.stableOrder();
    Path outputRootInSources =
        Preconditions.checkNotNull(sourceRoot.asPath())
            .getRelative(artifactsHelper.getOutputRootPath());

    for (String input : outputRootInputs) {
      // output_root_inputs are relative to the output_root directory, and we should
      // pass inside createOutputArtifact() paths, relative to working directory.
      DerivedArtifact derivedArtifact =
          artifactsHelper.createOutputArtifact(
              artifactsHelper
                  .getOutputRootPath()
                  .getRelative(input)
                  .relativeTo(artifactsHelper.getWorkingDirectory()));
      symlinks.add(derivedArtifact);
      // This method already expects the path relative to output_root.
      PathFragment absolutePath =
          outputRootInSources.getRelative(PathFragment.create(input)).asFragment();
      SymlinkAction symlinkAction =
          SymlinkAction.toAbsolutePath(
              ruleContext.getActionOwner(),
              absolutePath,
              derivedArtifact,
              String.format(
                  "Symlinking %s under <execroot>/%s", input, artifactsHelper.getOutputRootPath()));
      ruleContext.registerAction(symlinkAction);
    }
    return symlinks.build();
  }

  private static class TargetsPreparer {
    private ImmutableSortedMap<PathFragment, NinjaTarget> targetsMap;
    private ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap;
    private ImmutableSet<PathFragment> symlinkOutputs;

    public ImmutableSortedMap<PathFragment, NinjaTarget> getTargetsMap() {
      return targetsMap;
    }

    public ImmutableSortedMap<PathFragment, PhonyTarget> getPhonyTargetsMap() {
      return phonyTargetsMap;
    }

    public ImmutableSet<PathFragment> getSymlinkOutputs() {
      return symlinkOutputs;
    }

    void prepareTargets(List<NinjaTarget> ninjaTargets) throws GenericParsingException {
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> targetsMapBuilder =
          ImmutableSortedMap.naturalOrder();
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> phonyTargetsBuilder =
          ImmutableSortedMap.naturalOrder();
      ImmutableSet.Builder<PathFragment> symlinkOutputsBuilder = ImmutableSet.builder();
      categorizeTargetsAndOutputs(
          ninjaTargets, targetsMapBuilder, phonyTargetsBuilder, symlinkOutputsBuilder);
      targetsMap = targetsMapBuilder.build();
      phonyTargetsMap = NinjaPhonyTargetsUtil.getPhonyPathsMap(phonyTargetsBuilder.build());
      symlinkOutputs = symlinkOutputsBuilder.build();
    }

    /**
     * Iterate over all parsed Ninja targets into phony and non-phony targets in a single pass, and
     * run validations along the way. For non-phony targets, also extract the symlink_outputs for
     * registering symlink artifacts in actions later on, in ninja_build.
     *
     * @param ninjaTargets list of all parsed Ninja targets
     * @param targetsBuilder builder for map of path fragments to the non-phony targets
     * @param phonyTargetsBuilder builder for map of path fragments to the phony targets
     * @param symlinkOutputsBuilder builder for set of declared symlink outputs
     */
    private static void categorizeTargetsAndOutputs(
        List<NinjaTarget> ninjaTargets,
        ImmutableSortedMap.Builder<PathFragment, NinjaTarget> targetsBuilder,
        ImmutableSortedMap.Builder<PathFragment, NinjaTarget> phonyTargetsBuilder,
        ImmutableSet.Builder<PathFragment> symlinkOutputsBuilder)
        throws GenericParsingException {
      for (NinjaTarget target : ninjaTargets) {
        if ("phony".equals(target.getRuleName())) {
          if (target.getAllOutputs().size() != 1) {
            String allOutputs =
                target.getAllOutputs().stream()
                    .map(PathFragment::getPathString)
                    .collect(Collectors.joining(" "));
            throw new GenericParsingException(
                String.format(
                    "Ninja phony alias can only be used for single output, but found '%s'.",
                    allOutputs));
          }
          phonyTargetsBuilder.put(Iterables.getOnlyElement(target.getAllOutputs()), target);
        } else {
          for (PathFragment output : target.getAllOutputs()) {
            targetsBuilder.put(output, target);
          }
          symlinkOutputsBuilder.addAll(target.getAllSymlinkOutputs());
        }
      }
    }
  }

  private void checkDirectoriesAttributes(
      RuleContext ruleContext, PathFragment outputRoot, PathFragment workingDirectory)
      throws InterruptedException {
    Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    ImmutableSortedSet<String> notSymlinkedDirs =
        BazelSkyframeExecutorConstants.EXTERNAL_PACKAGE_HELPER.getNotSymlinkedInExecrootDirectories(
            env);
    if (env.valuesMissing()) {
      return;
    }

    // We can compare strings because notSymlinkedDirs contains normalized directory names
    if (!notSymlinkedDirs.contains(outputRoot.getPathString())) {
      ruleContext.attributeError(
          "output_root",
          String.format(
              "Ninja output root directory '%s' must be declared"
                  + " using global workspace function toplevel_output_directories().",
              outputRoot.getPathString()));
    }

    if (!workingDirectory.isEmpty() && !workingDirectory.equals(outputRoot)) {
      ruleContext.attributeError(
          "working_directory",
          String.format(
              "Ninja working directory '%s' is restricted to be either empty (or not defined),"
                  + " or be the same as output root '%s'.",
              workingDirectory.getPathString(), outputRoot.getPathString()));
    }
  }

  /**
   * As Ninja files describe the action graph, we must establish the dependency between Ninja files
   * and the Ninja graph configured target for the SkyFrame. We are doing it by computing all
   * related FileValue SkyValues.
   */
  private static void establishDependencyOnNinjaFiles(
      Environment env, Artifact mainFile, ImmutableList<Artifact> ninjaSrcs)
      throws InterruptedException {
    ArrayList<SkyKey> depKeys = Lists.newArrayList();
    depKeys.add(getArtifactRootedPath(mainFile));
    for (Artifact artifact : ninjaSrcs) {
      depKeys.add(getArtifactRootedPath(artifact));
    }
    env.getValues(depKeys);
  }

  private static SkyKey getArtifactRootedPath(Artifact artifact) {
    return FileValue.key(
        RootedPath.toRootedPath(artifact.getRoot().getRoot(), artifact.getRootRelativePath()));
  }
}
