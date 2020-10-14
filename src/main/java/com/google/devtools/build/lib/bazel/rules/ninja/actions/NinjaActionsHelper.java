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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRuleVariable;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget.InputKind;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Helper class for creating {@link NinjaAction} for each {@link NinjaTarget}, and linking the file
 * under the output_root into corresponding directory in execroot. See output_root property in the
 * {@link NinjaGraphRule}.
 */
public class NinjaActionsHelper {

  private final RuleContext ruleContext;
  private final RuleConfiguredTargetBuilder ruleConfiguredTargetBuilder;
  private final ImmutableSortedMap<PathFragment, NinjaTarget> targetsMap;
  private final ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargets;

  private final NinjaGraphArtifactsHelper artifactsHelper;

  private final PathFragment shellExecutable;
  private final ImmutableSortedMap<String, String> executionInfo;
  private final PhonyTargetArtifacts phonyTargetArtifacts;
  private final List<PathFragment> pathsToBuild;
  private final ImmutableSet<PathFragment> outputRootInputsSymlinks;

  /**
   * Constructor
   *
   * @param ruleContext parent NinjaGraphRule rule context
   * @param artifactsHelper helper object to create artifacts
   * @param targetsMap mapping of outputs to all non-phony Ninja targets from Ninja file
   * @param phonyTargets mapping of names to all phony Ninja actions from Ninja file
   * @param phonyTargetArtifacts helper class for computing transitively included artifacts of phony
   *     targets
   * @param pathsToBuild paths requested by the user to be build (in output_groups attribute)
   */
  NinjaActionsHelper(
      RuleContext ruleContext,
      RuleConfiguredTargetBuilder ruleConfiguredTargetBuilder,
      NinjaGraphArtifactsHelper artifactsHelper,
      ImmutableSortedMap<PathFragment, NinjaTarget> targetsMap,
      ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargets,
      PhonyTargetArtifacts phonyTargetArtifacts,
      List<PathFragment> pathsToBuild,
      ImmutableSet<PathFragment> outputRootInputsSymlinks) {
    this.ruleContext = ruleContext;
    this.artifactsHelper = artifactsHelper;
    this.targetsMap = targetsMap;
    this.phonyTargets = phonyTargets;
    this.shellExecutable = ShToolchain.getPathOrError(ruleContext);
    this.executionInfo = createExecutionInfo(ruleContext);
    this.phonyTargetArtifacts = phonyTargetArtifacts;
    this.pathsToBuild = pathsToBuild;
    this.outputRootInputsSymlinks = outputRootInputsSymlinks;
    this.ruleConfiguredTargetBuilder = ruleConfiguredTargetBuilder;
  }

  void createNinjaActions() throws GenericParsingException {
    // Traverse the action graph starting from the targets, specified by the user.
    // Only create the required actions.
    Set<PathFragment> visitedPaths = Sets.newHashSet();
    Set<NinjaTarget> visitedTargets = Sets.newHashSet();
    visitedPaths.addAll(pathsToBuild);
    ArrayDeque<PathFragment> queue = new ArrayDeque<>(pathsToBuild);
    Consumer<Collection<PathFragment>> enqueuer =
        paths -> {
          for (PathFragment input : paths) {
            if (visitedPaths.add(input)) {
              queue.add(input);
            }
          }
        };
    while (!queue.isEmpty()) {
      PathFragment fragment = queue.remove();
      NinjaTarget target = targetsMap.get(fragment);
      if (target != null) {
        // If the output is already created by a symlink action created from specifying that
        // file in output_root_inputs attribute of the ninja_graph rule, do not create other
        // actions that output that same file, since that will result in an action conflict.
        if (!outputRootInputsSymlinks.contains(fragment)) {
          if (visitedTargets.add(target)) {
            createNinjaAction(target);
          }
          enqueuer.accept(target.getAllInputs());
        } else {
          // Verify that the Ninja action we're skipping (because its outputs are already
          // being symlinked using output_root_inputs) has only symlink outputs specified in
          // output_root_inputs. Otherwise we might skip some other outputs.
          List<PathFragment> outputsInOutputRootInputsSymlinks = new ArrayList<>();
          List<PathFragment> outputsNotInOutputRootInputsSymlinks = new ArrayList<>();
          for (PathFragment output : target.getAllOutputs()) {
            if (outputRootInputsSymlinks.contains(output)) {
              outputsInOutputRootInputsSymlinks.add(output);
            } else {
              outputsNotInOutputRootInputsSymlinks.add(output);
            }
          }
          if (!outputsNotInOutputRootInputsSymlinks.isEmpty()) {
            throw new GenericParsingException(
                "Ninja target "
                    + target.getRuleName()
                    + " has "
                    + "outputs in output_root_inputs and other outputs not in output_root_inputs:\n"
                    + "Outputs in output_root_inputs:\n  "
                    + Joiner.on("  \n").join(outputsInOutputRootInputsSymlinks)
                    + "\nOutputs not in output_root_inputs:\n  "
                    + Joiner.on("  \n").join(outputsNotInOutputRootInputsSymlinks));
          }
        }
      } else {
        PhonyTarget phonyTarget = phonyTargets.get(fragment);
        // Phony target can be null, if the path in neither regular or phony target,
        // but the source file.
        if (phonyTarget != null) {
          phonyTarget.visitExplicitInputs(phonyTargets, enqueuer::accept);
        }
      }
    }
  }

  private void createNinjaAction(NinjaTarget target) throws GenericParsingException {
    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> orderOnlyInputsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> validationInputs = NestedSetBuilder.stableOrder();
    ImmutableList.Builder<Artifact> outputsBuilder = ImmutableList.builder();
    boolean isAlwaysDirty =
        fillArtifacts(
            target, inputsBuilder, orderOnlyInputsBuilder, validationInputs, outputsBuilder);

    ImmutableSortedMap<NinjaRuleVariable, String> resolvedMap = target.computeRuleVariables();
    String command = resolvedMap.get(NinjaRuleVariable.COMMAND);
    maybeCreateRspFile(target.getRuleName(), inputsBuilder, resolvedMap);

    if (!artifactsHelper.getWorkingDirectory().isEmpty()) {
      command = String.format("cd %s && ", artifactsHelper.getWorkingDirectory()) + command;
    }
    CommandLines commandLines =
        CommandLines.of(ImmutableList.of(shellExecutable.getPathString(), "-c", command));
    Artifact depFile = getDepfile(resolvedMap);
    if (depFile != null) {
      outputsBuilder.add(depFile);
    }

    List<Artifact> outputs = outputsBuilder.build();

    if (!validationInputs.isEmpty()) {
      ruleConfiguredTargetBuilder.addOutputGroup(
          OutputGroupInfo.VALIDATION, validationInputs.build());
    }

    ruleContext.registerAction(
        new NinjaAction(
            ruleContext.getActionOwner(),
            artifactsHelper.getSourceRoot(),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            inputsBuilder.build(),
            orderOnlyInputsBuilder.build(),
            outputs,
            commandLines,
            Preconditions.checkNotNull(ruleContext.getConfiguration()).getActionEnvironment(),
            executionInfo,
            createProgressMessage(target, resolvedMap, outputs),
            EmptyRunfilesSupplier.INSTANCE,
            isAlwaysDirty,
            depFile,
            artifactsHelper.getDerivedOutputRoot()));
  }

  /**
   * Create a progress message for the ninja action.
   *
   * <ul>
   *   <li>If the target has a "description" variable, use that. It has been expanded at parse time
   *       with file variables.
   *   <li>If the rule for the target has a description, use that. It has been expanded with rule,
   *       build and file variables.
   *   <li>Else, generate a pretty-printed progress message at runtime, using the rule name and
   *       output filenames for a general idea on what the action is doing, without printing the
   *       full command line (which can be surfaced with --subcommands, anyway).
   */
  private static String createProgressMessage(
      NinjaTarget target,
      ImmutableSortedMap<NinjaRuleVariable, String> ruleVariables,
      List<Artifact> outputs) {
    String ruleDescription = ruleVariables.get(NinjaRuleVariable.DESCRIPTION);
    if (ruleDescription != null) {
      return ruleDescription;
    }

    String ruleName = target.getRuleName();
    StringBuilder messageBuilder = new StringBuilder();
    if (!ruleName.isEmpty()) {
      messageBuilder.append("[rule ").append(ruleName).append("] ");
    }
    messageBuilder.append("Outputs: ");
    messageBuilder.append(outputs.stream().map(Artifact::getFilename).collect(joining(", ")));
    return messageBuilder.toString();
  }

  /** Returns true if the action should be marked as always dirty. */
  private boolean fillArtifacts(
      NinjaTarget target,
      NestedSetBuilder<Artifact> inputsBuilder,
      NestedSetBuilder<Artifact> orderOnlyInputsBuilder,
      NestedSetBuilder<Artifact> validationInputsBuilder,
      ImmutableList.Builder<Artifact> outputsBuilder)
      throws GenericParsingException {

    boolean isAlwaysDirty = false;
    for (Map.Entry<InputKind, PathFragment> entry : target.getAllInputsAndKind()) {

      InputKind kind = entry.getKey();
      NestedSetBuilder<Artifact> builder;
      if (kind == InputKind.ORDER_ONLY) {
        builder = orderOnlyInputsBuilder;
      } else if (kind == InputKind.VALIDATION) {
        // Note that validation inputs are specific to AOSP's Ninja implementation.
        builder = validationInputsBuilder;
      } else {
        builder = inputsBuilder;
      }

      PathFragment input = entry.getValue();
      PhonyTarget phonyTarget = this.phonyTargets.get(input);
      if (phonyTarget != null) {
        builder.addTransitive(phonyTargetArtifacts.getPhonyTargetArtifacts(input));
        isAlwaysDirty |= (phonyTarget.isAlwaysDirty() && kind != InputKind.ORDER_ONLY);
      } else {
        Artifact artifact = artifactsHelper.getInputArtifact(input);
        builder.add(artifact);
      }
    }

    for (PathFragment output : target.getAllOutputs()) {
      outputsBuilder.add(artifactsHelper.createOutputArtifact(output));
    }
    return isAlwaysDirty;
  }

  @Nullable
  private Artifact getDepfile(ImmutableSortedMap<NinjaRuleVariable, String> ruleVariables)
      throws GenericParsingException {
    String depfileName = ruleVariables.get(NinjaRuleVariable.DEPFILE);
    if (depfileName != null) {
      if (!depfileName.trim().isEmpty()) {
        return artifactsHelper.createOutputArtifact(PathFragment.create(depfileName));
      }
    }
    return null;
  }

  private void maybeCreateRspFile(
      String ruleName,
      NestedSetBuilder<Artifact> inputsBuilder,
      ImmutableSortedMap<NinjaRuleVariable, String> ruleVariables)
      throws GenericParsingException {
    String fileName = ruleVariables.get(NinjaRuleVariable.RSPFILE);
    String contentString = ruleVariables.get(NinjaRuleVariable.RSPFILE_CONTENT);

    if (fileName == null && contentString == null) {
      return;
    }
    if (fileName == null || contentString == null) {
      ruleContext.ruleError(
          String.format(
              "Both rspfile and rspfile_content should be defined for rule '%s'.", ruleName));
      return;
    }

    if (!fileName.trim().isEmpty()) {
      DerivedArtifact rspArtifact =
          artifactsHelper.createOutputArtifact(PathFragment.create(fileName));
      FileWriteAction fileWriteAction =
          FileWriteAction.create(ruleContext, rspArtifact, contentString, false);
      ruleContext.registerAction(fileWriteAction);
      inputsBuilder.add(rspArtifact);
    }
  }

  private static ImmutableSortedMap<String, String> createExecutionInfo(RuleContext ruleContext) {
    ImmutableSortedMap.Builder<String, String> builder = ImmutableSortedMap.naturalOrder();
    builder.putAll(TargetUtils.getExecutionInfo(ruleContext.getRule()));
    builder.put("local", "");
    ImmutableSortedMap<String, String> map = builder.build();
    Preconditions.checkNotNull(ruleContext.getConfiguration())
        .modifyExecutionInfo(map, "NinjaRule");
    return map;
  }
}
