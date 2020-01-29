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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRule;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRuleVariable;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaVariableValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor.VisitedState;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.List;
import java.util.stream.Collectors;

public class NinjaActionsHelper {
  private final RuleContext ruleContext;
  private final PathFragment outputRootPath;
  private final PathFragment workingDirectory;
  private final List<String> outputRootInputs;
  private final ImmutableSortedMap<PathFragment, NinjaTarget> allUsualTargets;
  private final ImmutableSortedMap<PathFragment, NestedSet<PathFragment>> phonyTargets;

  private final NinjaGraphArtifactsHelper artifactsHelper;

  private final PathFragment shellExecutable;
  private final ImmutableSortedMap<String, String> executionInfo;

  NinjaActionsHelper(RuleContext ruleContext,
      Root sourceRoot,
      PathFragment outputRootPath, PathFragment workingDirectory,
      List<String> outputRootInputs,
      ImmutableSortedMap<PathFragment, NinjaTarget> allUsualTargets,
      ImmutableSortedMap<PathFragment, NestedSet<PathFragment>> phonyTargets) {
    this.ruleContext = ruleContext;
    this.outputRootPath = outputRootPath;
    this.workingDirectory = workingDirectory;
    this.outputRootInputs = outputRootInputs;
    this.allUsualTargets = allUsualTargets;
    this.phonyTargets = phonyTargets;
    this.artifactsHelper = new NinjaGraphArtifactsHelper(ruleContext,
        sourceRoot, outputRootPath, workingDirectory);
    this.shellExecutable = ShToolchain.getPathOrError(ruleContext);
    this.executionInfo = createExecutionInfo(ruleContext);
  }

  void process() throws GenericParsingException {
    this.artifactsHelper.createInputsMap();
    createSymlinkActions();
    createNinjaActions();
  }

  private void createSymlinkActions() throws GenericParsingException {
    if (this.outputRootInputs.isEmpty()) {
      return;
    }
    for (String input : this.outputRootInputs) {
      PathFragment inputPathFragment = PathFragment.create(input);
      DerivedArtifact derivedArtifact =
          artifactsHelper.createDerivedArtifactUnderOutputRoot(inputPathFragment);
      PathFragment absolutePath = artifactsHelper.createAbsolutePathUnderOutputRoot(inputPathFragment);
      SymlinkAction symlinkAction = SymlinkAction.toAbsolutePath(ruleContext.getActionOwner(),
          absolutePath, derivedArtifact,
          String.format("Symlinking %s under <execroot>/%s", input, this.outputRootPath));
      ruleContext.registerAction(symlinkAction);
    }
  }

  private void createNinjaActions() throws GenericParsingException {
    for (NinjaTarget target : allUsualTargets.values()) {
      createNinjaAction(target);
    }
  }

  private void createNinjaAction(NinjaTarget target) throws GenericParsingException {
    NinjaRule rule = getNinjaRule(target);

    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    ImmutableList.Builder<Artifact> outputsBuilder = ImmutableList.builder();
    fillArtifacts(target, inputsBuilder, outputsBuilder);

    NinjaScope targetScope = createTargetScope(target);
    int targetOffset = target.getOffset();
    maybeCreateRspFile(rule, targetScope, targetOffset, inputsBuilder);

    String command = targetScope
        .getExpandedValue(targetOffset, rule.getVariables().get(NinjaRuleVariable.COMMAND));
    if (!this.workingDirectory.isEmpty()) {
      command = String.format("cd %s && ", this.workingDirectory) + command;
    }
    CommandLines commandLines = CommandLines
        .of(ImmutableList.of(shellExecutable.getPathString(), "-c", command));
    ruleContext.registerAction(new NinjaGenericAction(
        ruleContext.getActionOwner(),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        inputsBuilder.build(),
        outputsBuilder.build(),
        commandLines,
        Preconditions.checkNotNull(ruleContext.getConfiguration()).getActionEnvironment(),
        executionInfo,
        EmptyRunfilesSupplier.INSTANCE
    ));
  }

  private void fillArtifacts(NinjaTarget target,
      NestedSetBuilder<Artifact> inputsBuilder,
      ImmutableList.Builder<Artifact> outputsBuilder) throws GenericParsingException {
    VisitedState<PathFragment> visitedState = new VisitedState<>();
    for (PathFragment input : target.getAllInputs()) {
      NestedSet<PathFragment> nestedSet = this.phonyTargets.get(input);
      if (nestedSet != null) {
        List<String> problems = Lists.newArrayList();
        new NestedSetVisitor<>(path -> {
          try {
            inputsBuilder.add(artifactsHelper.createInputArtifact(path));
          } catch (GenericParsingException e) {
            problems.add(e.getMessage());
          }
        }, visitedState).visit(nestedSet);
        if (!problems.isEmpty()) {
          throw new GenericParsingException(String.join("\n", problems));
        }
      } else {
        inputsBuilder.add(artifactsHelper.createInputArtifact(input));
      }
    }

    for (PathFragment output : target.getAllOutputs()) {
      outputsBuilder.add(artifactsHelper.createDerivedArtifactUnderOutputRoot(output));
    }
  }

  private void maybeCreateRspFile(NinjaRule rule, NinjaScope targetScope, int targetOffset,
      NestedSetBuilder<Artifact> inputsBuilder) throws GenericParsingException {
    NinjaVariableValue value = rule.getVariables().get(NinjaRuleVariable.RSPFILE);
    NinjaVariableValue content = rule.getVariables().get(NinjaRuleVariable.RSPFILE_CONTENT);
    if (value == null && content == null) {
      return;
    }
    if (value == null || content == null) {
      throw new GenericParsingException(
          String.format("Both rspfile and rspfile_content should be defined for rule '%s'.",
              rule.getName()));
    }
    String fileName = targetScope.getExpandedValue(targetOffset, value);
    String contentString = targetScope.getExpandedValue(targetOffset, content);
    if (!fileName.trim().isEmpty()) {
      DerivedArtifact derivedArtifact =
          artifactsHelper.createDerivedArtifactUnderOutputRoot(PathFragment.create(fileName));
      FileWriteAction fileWriteAction = FileWriteAction
          .create(ruleContext, derivedArtifact, contentString, false);
      ruleContext.registerAction(fileWriteAction);
      inputsBuilder.add(derivedArtifact);
    }
  }

  private NinjaScope createTargetScope(NinjaTarget target) {
    ImmutableSortedMap.Builder<String, List<Pair<Integer, String>>> builder =
        ImmutableSortedMap.naturalOrder();
    target.getVariables()
        .forEach((key, value) -> builder.put(key, ImmutableList.of(Pair.of(0, value))));
    String inNewline = target.getUsualInputs().stream().map(PathFragment::getPathString)
        .collect(Collectors.joining("\n"));
    String out = target.getOutputs().stream().map(PathFragment::getPathString)
        .collect(Collectors.joining(" "));
    builder.put("in", ImmutableList.of(Pair.of(0, inNewline.replace("\n", " "))));
    builder.put("in_newline", ImmutableList.of(Pair.of(0, inNewline)));
    builder.put("out", ImmutableList.of(Pair.of(0, out)));

    return target.getScope().createTargetsScope(builder.build());
  }

  private NinjaRule getNinjaRule(NinjaTarget target) throws GenericParsingException {
    String ruleName = target.getRuleName();
    Preconditions.checkState(!"phony".equals(ruleName));
    NinjaRule rule = target.getScope().findRule(target.getOffset(), ruleName);
    if (rule == null) {
      throw new GenericParsingException(
          String.format("Unknown Ninja rule: '%s'", ruleName));
    }
    return rule;
  }

  private static ImmutableSortedMap<String, String> createExecutionInfo(RuleContext ruleContext) {
    ImmutableSortedMap.Builder<String, String> builder = ImmutableSortedMap.naturalOrder();
    builder.putAll(TargetUtils.getExecutionInfo(ruleContext.getRule()));
    builder.put("local", "");
    ImmutableSortedMap<String, String> map = builder.build();
    Preconditions.checkNotNull(ruleContext.getConfiguration()).modifyExecutionInfo(map, "NinjaRule");
    return map;
  }

  public NestedSet<Artifact> getFilesToBuild() {
    return this.artifactsHelper.getOutputFiles();
  }
}
