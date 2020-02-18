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
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.pipeline.NinjaPipeline;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.repository.ExternalPackageUtil;
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
    if (!ruleContext.getAnalysisEnvironment().getSkylarkSemantics().experimentalNinjaActions()) {
      throw new RuleErrorException(
          "Usage of ninja_graph is only allowed with --experimental_ninja_actions flag");
    }
    Artifact mainArtifact = ruleContext.getPrerequisiteArtifact("main", Mode.TARGET);
    ImmutableList<Artifact> ninjaSrcs =
        ruleContext.getPrerequisiteArtifacts("ninja_srcs", Mode.TARGET).list();
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
            ImmutableSortedMap.of(),
            ImmutableSortedMap.of());
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
          new NinjaPipeline(
                  workspace.getRelative(workingDirectory),
                  MoreExecutors.listeningDecorator(NINJA_POOL),
                  childNinjaFiles,
                  ownerTargetName)
              .pipeline(mainArtifact.getPath());
      targetsPreparer.process(ninjaTargets);

      NinjaGraphProvider ninjaGraphProvider =
          new NinjaGraphProvider(
              outputRoot,
              workingDirectory,
              targetsPreparer.getUsualTargets(),
              targetsPreparer.getPhonyTargetsMap());

      NestedSet<Artifact> filesToBuild =
          createSymlinkActions(
              ruleContext, sourceRoot, outputRoot, outputRootInputs, artifactsHelper);
      if (ruleContext.hasErrors()) {
        return null;
      }

      return new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
          .addProvider(NinjaGraphProvider.class, ninjaGraphProvider)
          .setFilesToBuild(filesToBuild)
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
      PathFragment outputRootPath,
      List<String> outputRootInputs,
      NinjaGraphArtifactsHelper artifactsHelper)
      throws GenericParsingException {
    if (outputRootInputs.isEmpty()) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
    Path outputRootInSources =
        Preconditions.checkNotNull(sourceRoot.asPath()).getRelative(outputRootPath);
    for (String input : outputRootInputs) {
      // output_root_inputs are relative to the output_root directory, and we should
      // pass inside createOutputArtifact() paths, relative to working directory.
      DerivedArtifact derivedArtifact =
          artifactsHelper.createOutputArtifact(
              artifactsHelper
                  .getOutputRootPath()
                  .getRelative(input)
                  .relativeTo(artifactsHelper.getWorkingDirectory()));
      filesToBuild.add(derivedArtifact);
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
    return filesToBuild.build();
  }

  private static class TargetsPreparer {
    private ImmutableSortedMap<PathFragment, NinjaTarget> usualTargets;
    private ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap;

    public ImmutableSortedMap<PathFragment, NinjaTarget> getUsualTargets() {
      return usualTargets;
    }

    public ImmutableSortedMap<PathFragment, PhonyTarget> getPhonyTargetsMap() {
      return phonyTargetsMap;
    }

    void process(List<NinjaTarget> ninjaTargets) throws GenericParsingException {
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> usualTargetsBuilder =
          ImmutableSortedMap.naturalOrder();
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> phonyTargetsBuilder =
          ImmutableSortedMap.naturalOrder();
      separatePhonyTargets(ninjaTargets, usualTargetsBuilder, phonyTargetsBuilder);
      usualTargets = usualTargetsBuilder.build();
      phonyTargetsMap = NinjaPhonyTargetsUtil.getPhonyPathsMap(phonyTargetsBuilder.build());
    }

    private static void separatePhonyTargets(
        List<NinjaTarget> ninjaTargets,
        ImmutableSortedMap.Builder<PathFragment, NinjaTarget> usualTargetsBuilder,
        ImmutableSortedMap.Builder<PathFragment, NinjaTarget> phonyTargetsBuilder)
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
            usualTargetsBuilder.put(output, target);
          }
        }
      }
    }
  }

  private void checkDirectoriesAttributes(
      RuleContext ruleContext, PathFragment outputRoot, PathFragment workingDirectory)
      throws InterruptedException {
    Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    ImmutableSortedSet<String> notSymlinkedDirs =
        ExternalPackageUtil.getNotSymlinkedInExecrootDirectories(env);
    // We can compare strings because notSymlinkedDirs contains normalized directory names
    if (!notSymlinkedDirs.contains(outputRoot.getPathString())) {
      ruleContext.attributeError(
          "output_root",
          String.format(
              "Ninja output root directory '%s' must be declared"
                  + " using global workspace function dont_symlink_directories_in_execroot().",
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
