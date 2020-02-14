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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
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
import java.util.Map;
import java.util.TreeMap;
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
    Map<String, List<String>> outputGroupsMap =
        ruleContext.attributes().get("output_groups", Type.STRING_LIST_DICT);

    Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    establishDependencyOnNinjaFiles(env, mainArtifact, ninjaSrcs);
    checkDirectoriesAttributes(ruleContext, outputRoot, workingDirectory);

    if (env.valuesMissing() || ruleContext.hasErrors()) {
      return null;
    }

    Root sourceRoot = mainArtifact.getRoot().getRoot();
    List<PathFragment> pathsToBuild =
        outputGroupsMap.values().stream()
            .flatMap(List::stream)
            .map(PathFragment::create)
            .collect(Collectors.toList());
    NinjaGraphArtifactsHelper artifactsHelper =
        new NinjaGraphArtifactsHelper(
            ruleContext,
            sourceRoot,
            outputRoot,
            workingDirectory,
            createSrcsMap(ruleContext),
            createDepsMap(ruleContext),
            pathsToBuild);
    if (ruleContext.hasErrors()) {
      return null;
    }
    TargetsPreparer targetsPreparer = new TargetsPreparer();

    try {
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
      targetsPreparer.process(ninjaTargets, artifactsHelper);
      new NinjaActionsHelper(
              ruleContext,
              artifactsHelper,
              outputRootInputs,
              targetsPreparer.getUsualTargets(),
              targetsPreparer.getPhonyTargetsMap())
          .process();
    } catch (GenericParsingException | IOException e) {
      // IOException is possible with reading Ninja file, describing the action graph.
      ruleContext.ruleError(e.getMessage());
      return null;
    }

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
              targetsPreparer.getPhonyTargetsMap(),
              artifactsHelper.getOutputsMap());
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

  private static class TargetsPreparer {
    private ImmutableSortedMap<PathFragment, NinjaTarget> usualTargets;
    private ImmutableSortedMap<PathFragment, PhonyTarget<Artifact>> phonyTargetsMap;

    public ImmutableSortedMap<PathFragment, NinjaTarget> getUsualTargets() {
      return usualTargets;
    }

    public ImmutableSortedMap<PathFragment, PhonyTarget<Artifact>> getPhonyTargetsMap() {
      return phonyTargetsMap;
    }

    void process(List<NinjaTarget> ninjaTargets, NinjaGraphArtifactsHelper artifactsHelper)
        throws GenericParsingException {
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> usualTargetsBuilder =
          ImmutableSortedMap.naturalOrder();
      ImmutableSortedMap.Builder<PathFragment, NinjaTarget> phonyTargetsBuilder =
          ImmutableSortedMap.naturalOrder();
      separatePhonyTargets(ninjaTargets, usualTargetsBuilder, phonyTargetsBuilder);
      usualTargets = usualTargetsBuilder.build();
      phonyTargetsMap =
          NinjaPhonyTargetsUtil.getPhonyPathsMap(
              phonyTargetsBuilder.build(), artifactsHelper::getInputArtifact);
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

  private static NestedSet<Artifact> getGroupArtifacts(
      RuleContext ruleContext,
      List<String> targets,
      ImmutableSortedMap<PathFragment, PhonyTarget<Artifact>> phonyTargetsMap,
      ImmutableSortedMap<PathFragment, Artifact> outputsMap) {
    NestedSetBuilder<Artifact> nestedSetBuilder = NestedSetBuilder.stableOrder();
    for (String target : targets) {
      PathFragment path = PathFragment.create(target);
      PhonyTarget<Artifact> phonyTarget = phonyTargetsMap.get(path);
      if (phonyTarget != null) {
        nestedSetBuilder.addTransitive(phonyTarget.getInputs());
      } else {
        Artifact usualArtifact = outputsMap.get(path);
        if (usualArtifact == null) {
          ruleContext.ruleError(
              String.format("Required target '%s' is not created in ninja_graph.", path));
          return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
        }
        nestedSetBuilder.add(usualArtifact);
      }
    }
    return nestedSetBuilder.build();
  }

  private static ImmutableSortedMap<PathFragment, Artifact> createSrcsMap(RuleContext ruleContext) {
    ImmutableList<Artifact> srcs = ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();
    ImmutableSortedMap.Builder<PathFragment, Artifact> inputsMapBuilder =
        ImmutableSortedMap.naturalOrder();
    srcs.forEach(a -> inputsMapBuilder.put(a.getRootRelativePath(), a));
    return inputsMapBuilder.build();
  }

  private static ImmutableSortedMap<PathFragment, Artifact> createDepsMap(RuleContext ruleContext) {
    Map<String, TransitiveInfoCollection> mapping = ruleContext.getPrerequisiteMap("deps_mapping");
    ImmutableSortedMap.Builder<PathFragment, Artifact> builder = ImmutableSortedMap.naturalOrder();
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
        return ImmutableSortedMap.of();
      }
      builder.put(PathFragment.create(entry.getKey()), filesToBuild.getSingleton());
    }
    return builder.build();
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
